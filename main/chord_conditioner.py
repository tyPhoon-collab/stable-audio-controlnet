import typing as tp

import torch
import torch.nn.functional as F
from stable_audio_tools.models.conditioners import Conditioner
from torch import nn


class EmbeddingChordConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int = 64,
        embed_dim: int = 128,
        conv_channels: int = 32,
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
    ):
        super().__init__(conv_channels, output_dim)

        self.embed_dim = embed_dim
        self.num_chords = 25  # 12 maj + 12 min + 1 N

        # 直接25種類のコードをEmbedding
        self.chord_embedding = nn.Embedding(self.num_chords, embed_dim)
        self.conv = nn.Conv1d(
            embed_dim, conv_channels, kernel_size=conv_kernel_size, padding=conv_padding
        )

    def forward(self, chords: tp.Any, device: tp.Union[torch.device, str]) -> tp.Any:
        """
        chords: [(T_frames, 3)] [root, quality, inversion]
                root: 0-11 (C-B) or -1 (N)
                quality: 0 (maj), 1 (min) or -1 (N)
                inversion: not used
        """
        self.to(device)

        chords_data = chords[0]["data"]
        target_size = chords[0]["target_size"]

        # chordsを1つのidに変換
        # 0-11: C-B (maj), 12-23: C-B (min), 24: N
        chord_ids = chords_data[:, 0] + (chords_data[:, 1] == 1) * 12
        chord_ids[chords_data[:, 0] == -1] = 24  # N

        x = self.chord_embedding(chord_ids)  # (T_frames, embed_dim)
        x = x.transpose(0, 1).unsqueeze(0)  # (1, embed_dim, T_frames)
        x = self.conv(x)  # (1, conv_channels, T_frames)
        x = x.transpose(1, 2).squeeze(0)  # (T_frames, conv_channels)

        x = self.proj_out(x)  # (T_frames, output_dim)

        x = x.transpose(0, 1)  # (output_dim, T_frames)

        x = F.interpolate(
            x.unsqueeze(0),
            size=target_size,
            mode="linear",
            align_corners=False,
        )  # (1, output_dim, target_size)
        attention_mask = torch.ones(1, target_size, device=device)  # (1, target_size)

        return x, attention_mask


if __name__ == "__main__":
    # テストコード
    conditioner = EmbeddingChordConditioner(output_dim=768, embed_dim=16)
    chords = [
        torch.tensor(
            [
                [0, 0, 0],  # C maj
                [4, 1, 0],  # E min
                [7, 0, 0],  # G maj
                [-1, -1, 0],  # N
                [11, 1, 0],  # B min
            ]
        )
    ]
    output, mask = conditioner(chords, device="cpu")
    print(output.shape)
