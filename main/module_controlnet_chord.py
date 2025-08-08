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

""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        depth_factor: float,
        cfg_dropout_prob: float,
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay

        self.timestep_sampler = "logit_normal"
        self.diffusion_objective = "v"
        model, model_config = get_pretrained_controlnet_model(
            "stabilityai/stable-audio-open-1.0",
            controlnet_types=["chord"],
            depth_factor=depth_factor,
        )
        self.model_config = model_config
        self.sample_size = model_config["sample_size"]
        self.sample_rate = model_config["sample_rate"]

        self.cfg_dropout_prob = cfg_dropout_prob

        self.model = model
        self.model.model.model.requires_grad_(False)
        self.model.conditioner.requires_grad_(False)
        self.model.conditioner.eval()
        self.model.pretransform.requires_grad_(False)
        self.model.pretransform.eval()

    def configure_optimizers(self):
        params = list(self.model.model.controlnet.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def _chord_to_onehot(self, chord_batch: torch.Tensor) -> torch.Tensor:
        """
        chord_batch: (B, T_frames, 3) with values [root, quality, inversion]
          root: -1 (N) or 0..11
          quality: -1 (N) or 0..8
          inversion: 0..6 or others (treated as unknown)
        returns: (B, 31, T_frames) one-hot channels [13 root, 10 quality, 8 inversion]
        """
        B, T, _ = chord_batch.shape
        device = chord_batch.device

        root = chord_batch[..., 0].clone()
        qual = chord_batch[..., 1].clone()
        inv = chord_batch[..., 2].clone()

        # Map -1 to last index in its group
        root_idx = torch.where(
            root >= 0, root, torch.full_like(root, 12)
        )  # 0..11, 12 for N
        qual_idx = torch.where(
            qual >= 0, qual, torch.full_like(qual, 9)
        )  # 0..8, 9 for N
        # inversion: 0..6 valid, others -> 7
        inv_idx = torch.where((inv >= 0) & (inv <= 6), inv, torch.full_like(inv, 7))

        C_root, C_qual, C_inv = 13, 10, 8
        C_total = C_root + C_qual + C_inv
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

        # inversion
        out_inv = out[:, C_root + C_qual :]
        out_inv.zero_()
        out_inv.scatter_(1, inv_idx.long().unsqueeze(1), 1.0)

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

        # Prepare chord conditioning: (B, T_frames, 3) -> (B, 31, T_frames) -> upsample to T_samples
        B = x.shape[0]
        T_samples = x.shape[-1]
        if chord_batch.numel() == 0:
            raise ValueError("Empty chord_batch provided")
        chord_onehot = self._chord_to_onehot(chord_batch)  # (B, 31, T_frames)
        # Interpolate to sample length
        chord_rescaled = F.interpolate(chord_onehot, size=T_samples, mode="nearest")

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


""" Datamodule """


class WebDatasetDatamodule(pl.LightningDataModule):
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


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    print("WandbLogger not found.")
    return None


class SampleLogger(Callback):
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

        # Prepare chord conditioning for logging
        chord_onehot = pl_module._chord_to_onehot(chord_batch.to(pl_module.device))
        chord_rescaled = F.interpolate(chord_onehot, size=x.shape[-1], mode="nearest")

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
                cfg_scale=7.0,
                conditioning=conditioning,
                sample_size=pl_module.sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device="cuda",
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
