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
        conv_kernel_size: int = 7,
        conv_padding: int = 3,
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


class SeparatedEmbeddingChordConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int = 64,
        embed_dim: int = 128,
        conv_channels: int = 32,
        conv_kernel_size: int = 7,
        conv_padding: int = 3,
    ):
        super().__init__(conv_channels, output_dim)

        # root: 0-11 + 1 (for N)
        # quality: 0=maj,1=min,2=N
        self.root_embedding = nn.Embedding(13, embed_dim // 2)
        self.quality_embedding = nn.Embedding(3, embed_dim // 2)

        self.conv = nn.Conv1d(
            embed_dim, conv_channels, kernel_size=conv_kernel_size, padding=conv_padding
        )

    def encode_chords(self, chords_data: torch.Tensor):
        """
        chords_data: (T_frames, 3) -> [root, quality, inversion]
        root: 0-11 or -1 (N)
        quality: 0 (maj), 1 (min), or -1 (N)
        """
        # root: map -1 -> 12
        roots = chords_data[:, 0].clone()
        roots[roots == -1] = 12
        # quality: map -1 -> 2
        qualities = chords_data[:, 1].clone()
        qualities[qualities == -1] = 2
        return roots.long(), qualities.long()

    def forward(self, chords: tp.Any, device: tp.Union[torch.device, str]) -> tp.Any:
        chords_data = chords[0]["data"]
        target_size = chords[0]["target_size"]

        roots, qualities = self.encode_chords(chords_data)

        # 埋め込み結合
        root_emb = self.root_embedding(roots)
        quality_emb = self.quality_embedding(qualities)
        x = torch.cat([root_emb, quality_emb], dim=-1)  # (T_frames, embed_dim)

        x = x.transpose(0, 1).unsqueeze(0)  # (1, embed_dim, T_frames)
        x = self.conv(x)  # (1, conv_channels, T_frames)
        x = x.transpose(1, 2).squeeze(0)  # (T_frames, conv_channels)

        x = self.proj_out(x)  # (T_frames, output_dim)
        x = x.transpose(0, 1)  # (output_dim, T_frames)

        # target_size にリサイズ
        x = F.interpolate(
            x.unsqueeze(0),
            size=target_size,
            mode="linear",
            align_corners=False,
        )  # (1, output_dim, target_size)

        attention_mask = torch.ones(1, target_size, device=device)

        return x, attention_mask
