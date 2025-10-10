from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from torch.utils.data import DataLoader

from main.controlnet.pretrained import get_pretrained_controlnet_model
from main.utils import log_wandb_audio_batch, log_wandb_audio_spectrogram

# ================================================================================================
# MODEL
# ================================================================================================


class Model(pl.LightningModule):
    """ControlNet model for chord-conditioned audio generation.

    Supports both OneHot and Embedding representations for chord conditioning.
    Memory-optimized for 24GB VRAM training.
    """

    def __init__(
        self,
        # Optimizer parameters
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        # Model parameters
        depth_factor: float,
        cfg_dropout_prob: float,
        # Chord representation parameters
        chord_layer_type: str = "onehot",  # "onehot" or "embedding"
        chord_embed_dim: int = 16,  # 8, 16, 32 (memory-efficient default)
    ):
        super().__init__()

        # === Optimizer Configuration ===
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay

        # === Chord Representation Configuration ===
        self.chord_layer_type = chord_layer_type
        self.chord_embed_dim = chord_embed_dim

        # === Diffusion Configuration ===
        self.timestep_sampler = "logit_normal"
        self.diffusion_objective = "v"
        self.cfg_dropout_prob = cfg_dropout_prob
        model, model_config = get_pretrained_controlnet_model(
            "stabilityai/stable-audio-open-1.0",
            controlnet_types=["chord"],
            depth_factor=depth_factor,
        )
        self.model_config = model_config
        self.sample_size = model_config["sample_size"]
        self.sample_rate = model_config["sample_rate"]

        self.model = model
        self.model.model.model.requires_grad_(False)
        self.model.conditioner.requires_grad_(False)
        self.model.conditioner.eval()
        self.model.pretransform.requires_grad_(False)
        self.model.pretransform.eval()

        # Enable gradient checkpointing for long sequences (47+ seconds)
        if hasattr(self.model.model, 'enable_gradient_checkpointing'):
            self.model.model.enable_gradient_checkpointing()

        # Initialize chord embedding layers if using embedding mode
        if self.chord_layer_type == "embedding":
            self.root_embedding = torch.nn.Embedding(13, self.chord_embed_dim)  # 0-11 + N
            self.quality_embedding = torch.nn.Embedding(10, self.chord_embed_dim)  # 0-8 + N
            # inversion_embedding removed for memory efficiency

            # Ultra-lightweight combination layer with non-linearity
            # Non-linearity allows learning complex interactions between root and quality
            self.chord_combiner = torch.nn.Sequential(
                torch.nn.Linear(self.chord_embed_dim * 2, self.chord_embed_dim),
                torch.nn.SiLU(),  # SiLU (Swish) is memory-efficient and smooth
            )

            # Initialize embeddings
            self._init_chord_embeddings()

    def configure_optimizers(self):
        params = list(self.model.model.controlnet.parameters())

        # Add embedding parameters if using embedding mode
        if self.chord_layer_type == "embedding":
            params.extend(list(self.root_embedding.parameters()))
            params.extend(list(self.quality_embedding.parameters()))
            # inversion_embedding removed
            params.extend(list(self.chord_combiner.parameters()))

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    # === Embedding Initialization ===

    def _init_chord_embeddings(self):
        """Initialize embeddings with music theory principles."""
        self._init_root_embeddings()
        self._init_quality_embeddings()

    def _init_root_embeddings(self):
        """Initialize root embeddings using circle of fifths.

        Circle of fifths from C: C(0) -> G(7) -> D(2) -> A(9) -> E(4) -> B(11)
        -> F#(6) -> C#(1) -> G#(8) -> D#(3) -> A#(10) -> F(5) -> back to C
        """
        # Circle of fifths starting from C=0
        circle_of_fifths_from_c = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

        for position, note_index in enumerate(circle_of_fifths_from_c):
            angle = 2 * torch.pi * position / 12
            if self.chord_embed_dim >= 2:
                init_vec = torch.zeros(self.chord_embed_dim)
                init_vec[0] = torch.cos(torch.tensor(angle)) * 0.1
                init_vec[1] = torch.sin(torch.tensor(angle)) * 0.1
                if self.chord_embed_dim > 2:
                    # Small random noise for higher dimensions
                    init_vec[2:] = torch.randn(self.chord_embed_dim - 2) * 0.01
                self.root_embedding.weight.data[note_index] = init_vec

        # No chord (N) initialization - zero vector
        self.root_embedding.weight.data[12] = torch.zeros(self.chord_embed_dim)

    def _init_quality_embeddings(self):
        """Initialize quality embeddings with musical meanings."""
        if self.chord_embed_dim >= 2:
            # Musical quality meanings: [brightness, tension]
            quality_meanings = torch.tensor([
                [1.0, 0.0],   # major - bright, stable
                [-1.0, 0.0],  # minor - dark, stable
                [0.0, 1.0],   # diminished - neutral, tense
                [0.0, -1.0],  # augmented - neutral, unstable
                [0.5, 0.5],   # major7 - bright, sophisticated
                [-0.5, 0.5],  # minor7 - dark, jazzy
                [0.8, 0.2],   # dominant7 - very bright, bluesy
                [0.2, 0.8],   # half-dim - slightly bright, complex
                [0.0, 0.0],   # sus - neutral
            ])

            self.quality_embedding.weight.data[:9, :2] = quality_meanings * 0.1
            if self.chord_embed_dim > 2:
                self.quality_embedding.weight.data[:9, 2:] = torch.randn(9, self.chord_embed_dim - 2) * 0.01
        else:
            torch.nn.init.normal_(self.quality_embedding.weight, std=0.02)

        # No chord quality
        self.quality_embedding.weight.data[9] = torch.zeros(self.chord_embed_dim)

    # === Chord Representation Conversion ===

    def _chord_to_embedding(self, chord_batch: torch.Tensor) -> torch.Tensor:
        """
        Convert chord batch to embedding representation.

        Args:
            chord_batch: (B, T_frames, 3) [root, quality, inversion]
                        inversion is ignored for memory efficiency

        Returns:
            torch.Tensor: (B, chord_embed_dim, T_frames)
        """
        B, T, _ = chord_batch.shape
        device = chord_batch.device

        # Validation
        if chord_batch.numel() == 0:
            raise ValueError("Empty chord_batch provided")

        assert chord_batch.shape[-1] == 3, f"Expected chord_batch shape (B, T, 3), got {chord_batch.shape}"

        # Normalize indices (inversion is ignored)
        root = chord_batch[..., 0].clone()
        qual = chord_batch[..., 1].clone()
        # inv = chord_batch[..., 2].clone()  # Not used

        # Map -1 (no chord) to special indices: 12 for root, 9 for quality
        root_idx = torch.where(root >= 0, root, torch.full_like(root, 12))  # 0..11, 12 for N
        qual_idx = torch.where(qual >= 0, qual, torch.full_like(qual, 9))   # 0..8, 9 for N

        # Validate index ranges
        assert root_idx.max() <= 12 and root_idx.min() >= 0, f"Invalid root indices: [{root_idx.min()}, {root_idx.max()}]"
        assert qual_idx.max() <= 9 and qual_idx.min() >= 0, f"Invalid quality indices: [{qual_idx.min()}, {qual_idx.max()}]"

        # Memory-efficient embedding processing
        with torch.cuda.amp.autocast(enabled=True):
            # Embedding lookup (inversion removed)
            root_emb = self.root_embedding(root_idx.long())      # (B, T, embed_dim)
            qual_emb = self.quality_embedding(qual_idx.long())   # (B, T, embed_dim)

            # Combine embeddings (2 inputs only)
            combined = torch.cat([root_emb, qual_emb], dim=-1)  # (B, T, 2*embed_dim)
            chord_emb = self.chord_combiner(combined)  # (B, T, embed_dim)

        # Transpose to (B, embed_dim, T) for interpolation
        return chord_emb.transpose(1, 2)

    def _chord_to_onehot(self, chord_batch: torch.Tensor) -> torch.Tensor:
        """
        Convert chord batch to one-hot representation.

        Args:
            chord_batch: (B, T_frames, 3) [root, quality, inversion]
                        inversion is ignored for memory efficiency

        Returns:
            torch.Tensor: (B, 23, T_frames) [13 root + 10 quality channels]
        """
        B, T, _ = chord_batch.shape
        device = chord_batch.device

        root = chord_batch[..., 0].clone()
        qual = chord_batch[..., 1].clone()
        # inv = chord_batch[..., 2].clone()  # Not used

        # Map -1 to last index in its group
        root_idx = torch.where(
            root >= 0, root, torch.full_like(root, 12)
        )  # 0..11, 12 for N
        qual_idx = torch.where(
            qual >= 0, qual, torch.full_like(qual, 9)
        )  # 0..8, 9 for N

        C_root, C_qual = 13, 10  # inversion removed
        C_total = C_root + C_qual  # 31 -> 23
        out = torch.zeros((B, C_total, T), device=device, dtype=torch.float32)

        # scatter for each group
        # root
        out_root = out[:, 0:C_root]
        out_root.zero_()
        out_root.scatter_(1, root_idx.long().unsqueeze(1), 1.0)

        # quality
        out_qual = out[:, C_root : C_root + C_qual]
        out_qual.zero_()
        out_qual.scatter_(1, qual_idx.long().unsqueeze(1), 1.0)

        return out

    def step(self, batch):
        # Support both collate variants: mix (5-tuple) and conditional (6-tuple)
        if len(batch) == 5:
            x, prompts, start_seconds, total_seconds, chord_batch = batch
        elif len(batch) == 6:
            x, _y_in, prompts, start_seconds, total_seconds, chord_batch = batch
        else:
            raise ValueError("Unexpected batch format for chord training")

        diffusion_input = self.model.pretransform.encode(x)

        # if self.timestep_sampler == "uniform":
        #     # Draw uniformly distributed continuous timesteps
        #     # t = self.rng.draw(x.shape[0])[:, 0]
        if self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(x.shape[0]))
        else:
            raise ValueError(f"Unknown time step sampler: {self.timestep_sampler}")

        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        else:
            raise ValueError("Diffusion objective not supported")

        alphas = alphas[:, None, None].to(self.device)
        sigmas = sigmas[:, None, None].to(self.device)

        noise = torch.randn_like(diffusion_input).to(self.device)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas

        # === Chord Conditioning Processing ===
        B, T_samples = x.shape[0], x.shape[-1]

        if chord_batch.numel() == 0:
            raise ValueError("Empty chord_batch provided")

        # Convert and interpolate chord representation
        with torch.cuda.amp.autocast(enabled=True):
            if self.chord_layer_type == "embedding":
                chord_representation = self._chord_to_embedding(chord_batch)  # (B, chord_embed_dim, T_frames)
                chord_rescaled = F.interpolate(chord_representation, size=T_samples, mode="linear", align_corners=False)
            elif self.chord_layer_type == "onehot":
                chord_representation = self._chord_to_onehot(chord_batch)  # (B, 23, T_frames)
                chord_rescaled = F.interpolate(chord_representation, size=T_samples, mode="nearest")
            else:
                raise ValueError(f"Unsupported chord_layer_type: {self.chord_layer_type}. Supported types are 'onehot' and 'embedding'.")

        output = self.model(
            x=noised_inputs,
            t=t.to(self.device),
            cond=self.model.conditioner(
                [
                    {
                        "prompt": prompts[i],
                        "seconds_start": start_seconds[i],
                        "seconds_total": total_seconds[i],
                        "chord": chord_rescaled[i : i + 1],
                    }
                    for i in range(B)
                ],
                device=self.device,
            ),
            cfg_dropout_prob=self.cfg_dropout_prob,
        )
        loss = torch.nn.functional.mse_loss(output, targets).mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("valid_loss", loss)
        return loss


# ================================================================================================
# DATAMODULE
# ================================================================================================


class WebDatasetDatamodule(pl.LightningDataModule):
    """WebDataset-based DataModule for chord conditioning training."""
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size_train: int,
        batch_size_val: int,
        num_workers: int,
        pin_memory: bool,
        shuffle_size: int,
        collate_fn=None,
        drop_last: bool = True,
        persistent_workers: bool = True,
        multiprocessing_context: str = "spawn",
    ) -> None:
        super().__init__()
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_size = shuffle_size
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context

        train_dataset = train_dataset.shuffle(self.shuffle_size)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
        )


# ================================================================================================
# CALLBACKS
# ================================================================================================


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    print("WandbLogger not found.")
    return None


class SampleLogger(Callback):
    """Callback for logging generated audio samples during validation."""

    def __init__(
        self, sampling_steps: List[int], cfg_scale: float, num_samples: int = 1
    ) -> None:
        self.sampling_steps = sampling_steps
        self.cfg_scale = cfg_scale
        self.num_samples = num_samples
        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        wandb_logger = get_wandb_logger(trainer).experiment
        # batch may be 5-tuple (mix) or 6-tuple (conditional)
        if len(batch) == 5:
            x, prompts, start_seconds, total_seconds, chord_batch = batch
        else:
            x, _y_in, prompts, start_seconds, total_seconds, chord_batch = batch
        x = torch.clip(x, -1, 1)

        num_samples = min(self.num_samples, x.shape[0])

        # === Chord Conditioning for Sampling ===
        with torch.cuda.amp.autocast(enabled=True):
            if pl_module.chord_layer_type == "embedding":
                chord_representation = pl_module._chord_to_embedding(chord_batch.to(pl_module.device))
                chord_rescaled = F.interpolate(chord_representation, size=x.shape[-1], mode="linear", align_corners=False)
            elif pl_module.chord_layer_type == "onehot":
                chord_representation = pl_module._chord_to_onehot(chord_batch.to(pl_module.device))
                chord_rescaled = F.interpolate(chord_representation, size=x.shape[-1], mode="nearest")
            else:
                raise ValueError(f"Unsupported chord_layer_type: {pl_module.chord_layer_type}. Supported types are 'onehot' and 'embedding'.")

        conditioning = [
            {
                "prompt": prompts[i],
                "seconds_start": start_seconds[i],
                "seconds_total": total_seconds[i],
                "chord": chord_rescaled[i : i + 1],
            }
            for i in range(num_samples)
        ]

        for i in range(num_samples):
            log_wandb_audio_batch(
                logger=wandb_logger,
                id=f"true_{i}",
                samples=x[i : i + 1],
                sampling_rate=pl_module.sample_rate,
                caption=f"Prompt: {prompts[i]}",
            )
            log_wandb_audio_spectrogram(
                logger=wandb_logger,
                id=f"true_{i}",
                samples=x[i : i + 1],
                sampling_rate=pl_module.sample_rate,
                caption=f"Prompt: {prompts[i]}",
            )

        for steps in self.sampling_steps:
            output = generate_diffusion_cond(
                pl_module.model,
                batch_size=num_samples,
                steps=steps,
                cfg_scale=self.cfg_scale,
                conditioning=conditioning,
                sample_size=pl_module.sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=pl_module.device,
            )
            for i in range(num_samples):
                log_wandb_audio_batch(
                    logger=wandb_logger,
                    id=f"sample_x_{i}",
                    samples=output[i : i + 1],
                    sampling_rate=pl_module.sample_rate,
                    caption=f"Sampled in {steps} steps.",
                )
                log_wandb_audio_spectrogram(
                    logger=wandb_logger,
                    id=f"sample_x_{i}",
                    samples=output[i : i + 1],
                    sampling_rate=pl_module.sample_rate,
                    caption=f"Sampled in {steps} steps.",
                )

        if is_train:
            pl_module.train()
