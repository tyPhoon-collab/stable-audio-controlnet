import typing as tp

import torch
import torch.nn.functional as F
from stable_audio_tools.models.conditioners import Conditioner
from torch import nn


class EmbeddingMelodyConditioner(Conditioner):
    """Simple melody conditioner using note embeddings.

    Expected input per item (in the list passed to forward):
    {
        "data": Tensor[T] or Tensor[K, T],  # per-frame note indices (0..127) or -1 for rest/unvoiced
        "target_size": int,                 # output temporal length to match diffusion latent
    }

    Strategy:
    - Map note indices to 129 tokens: [0..127] for MIDI notes, 128 for rest (-1)
    - If multiple sequences are provided (K,T), embed each and mean-pool across K -> (T, embed_dim)
    - Conv1d -> proj_out -> temporal interpolation to target_size
    - Return (cond, attention_mask) where cond has shape (1, output_dim, target_size)
    """

    def __init__(
        self,
        output_dim: int = 64,
        embed_dim: int = 128,
        conv_channels: int = 32,
    ) -> None:
        super().__init__(conv_channels, output_dim)
        self.embed_dim = embed_dim
        self.num_tokens = 129  # 0..127 notes + 128 for rest/unvoiced

        self.note_embedding = nn.Embedding(self.num_tokens, embed_dim)
        self.conv = nn.Conv1d(embed_dim, conv_channels, kernel_size=3, padding=1)

    def forward(self, x, device: tp.Union[torch.device, str]):  # type: ignore[override]
        # Move internal modules to device
        self.to(device)

        # Expect x as a list with one dict: {"data": Tensor[T], "target_size": int}
        mel_data: torch.Tensor = x[0]["data"]  # (T,) or (K, T)
        target_size: int = int(x[0]["target_size"])  # scalar

        # Map input notes to token ids in [0..128]; -1 (rest/unvoiced) -> 128
        if mel_data.dim() == 1:
            # (T,)
            token_ids = mel_data.clone().long()
            token_ids = torch.where(
                token_ids < 0, torch.full_like(token_ids, 128), token_ids
            )
            token_ids = torch.clamp(token_ids, 0, 128)

            x = self.note_embedding(token_ids)  # (T, embed_dim)
        else:
            # (K, T)
            token_ids = mel_data.clone().long()
            token_ids = torch.where(
                token_ids < 0, torch.full_like(token_ids, 128), token_ids
            )
            token_ids = torch.clamp(token_ids, 0, 128)

            # Embed each channel then mean-pool across K
            x_embed = self.note_embedding(token_ids)  # (K, T, embed_dim)
            x = x_embed.mean(dim=0)  # (T, embed_dim)

        x = x.transpose(0, 1).unsqueeze(0)  # (1, embed_dim, T)
        x = self.conv(x)  # (1, conv_channels, T_frames)
        x = x.transpose(1, 2).squeeze(0)  # (T_frames, conv_channels)

        x = self.proj_out(x)  # (T_frames, output_dim)
        x = x.transpose(0, 1)  # (output_dim, T_frames)

        # Interpolate over time to match diffusion latent length
        x = F.interpolate(
            x.unsqueeze(0),
            size=target_size,
            mode="linear",
            align_corners=False,
        )  # (1, output_dim, target_size)

        attention_mask = torch.ones(1, target_size, device=device)
        return x, attention_mask


if __name__ == "__main__":
    # Quick self-check
    conditioner = EmbeddingMelodyConditioner(output_dim=768, embed_dim=32)
    # Sequence of notes with rests: [C4=60, D4=62, rest, G4=67]
    mel = [
        {
            "data": torch.tensor([60, 62, -1, 67], dtype=torch.long),
            "target_size": 128,
        }
    ]
    cond, mask = conditioner(x=mel, device="cpu")
    print(cond.shape, mask.shape)
