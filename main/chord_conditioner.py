import typing as tp

import torch
import torch.nn.functional as F
from stable_audio_tools.models.conditioners import Conditioner
from torch import nn

from main.data.annotation import (
    NUM_CHORD_QUALITIES,
    QUALITY_CHROMA_INTERVALS,
    QUALITY_NAMES,
)


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


class RotatedChromaChordConditioner(Conditioner):
    """
    Coco-Mullaの 25次元 (Root 12 + Chroma 12 + NoChord 1) 表現を使用。
    ただし、クロマ部分はルートに基づいて回転される。
    入力: (T_frames, 3) [root, quality_idx, inversion]
    """

    def __init__(
        self,
        output_dim: int = 64,
        embed_dim: int = 128,
        conv_channels: int = 32,
        conv_kernel_size: int = 7,
        conv_padding: int = 3,
    ):
        super().__init__(conv_channels, output_dim)

        # 25次元 (Root 12 + Chroma 12 + NoChord 1)
        self.input_dim = 25

        # 25次元ベクトルをembed_dimに射影する層
        self.input_proj = nn.Linear(self.input_dim, embed_dim)

        self.conv = nn.Conv1d(
            embed_dim, conv_channels, kernel_size=conv_kernel_size, padding=conv_padding
        )

        # --- 和音構成音 (Chroma) のルックアップテーブルを作成 ---
        quality_map_tensor = torch.zeros(NUM_CHORD_QUALITIES, 12)

        for i, name in enumerate(QUALITY_NAMES):
            intervals = QUALITY_CHROMA_INTERVALS.get(name)
            if intervals:
                # 構成音のインデックス (タプル) をリストに変換して設定
                quality_map_tensor[i, list(intervals)] = 1.0

        # モデルのバッファとして登録 (GPU対応)
        self.register_buffer("chord_quality_map", quality_map_tensor.float())
        # ----------------------------------------------------

    def encode_chords(self, chords_data: torch.Tensor, device: torch.device):
        """
        [root, quality_idx, inversion] から 25次元のCoco-Mulla表現 (c_i) を構築。

        Args:
            chords_data (torch.Tensor): (T_frames, 3)
                [root, quality_idx, inversion]
                root: 0-11 (N/A=-1)
                quality_idx: 0-13 (N/A=-1) (QUALITY_NAMESのインデックス)
        Returns:
            torch.Tensor: (T_frames, 25)
        """
        T_frames = chords_data.shape[0]
        c = torch.zeros(T_frames, self.input_dim, device=device)

        # 'no-chord' (root == -1) のフレームを特定
        no_chord_mask = chords_data[:, 0] == -1
        chord_mask = ~no_chord_mask

        # 1. No-Chord フレーム (論文 Eq. 1 の 'otherwise' ケース)
        c[no_chord_mask, 24] = 1.0  # no-chord フラグ (最後の次元) を立てる

        # 2. Chord フレーム
        if torch.any(chord_mask):
            roots = chords_data[chord_mask, 0].long()
            quality_idx = chords_data[chord_mask, 1].long()

            # Root (0-11)
            c[chord_mask, 0:12] = F.one_hot(roots, num_classes=12).float()

            # Chroma (12-23)
            # (rootを基準とした構成音)
            chord_quality_map = tp.cast(torch.Tensor, self.chord_quality_map)
            base_chroma = chord_quality_map[quality_idx]
            pitch_classes = torch.arange(12, device=device)
            roll_indices = (pitch_classes.unsqueeze(0) - roots.unsqueeze(1)) % 12
            rotated_chroma = torch.gather(base_chroma, 1, roll_indices)
            c[chord_mask, 12:24] = rotated_chroma

        return c

    def forward(self, chords: tp.Any, device: tp.Union[torch.device, str]) -> tp.Any:  # type: ignore[override]
        """
        chords (dict):
            "data" (Tensor): (T_frames, 3) [root, quality_idx, inversion]
            "target_size" (int): F.interpolate の目標フレーム数
        """
        # create_chord_tensor からの出力を想定
        target_device = torch.device(device) if isinstance(device, str) else device

        chords_data = chords[0]["data"].to(target_device)
        target_size = chords[0]["target_size"]

        # 1. Coco-Mulla 表現 (25-dim) にエンコード
        # (T_frames, 25)
        x = self.encode_chords(chords_data, target_device)

        # 2. embed_dim に射影
        # (T_frames, embed_dim)
        x = self.input_proj(x)

        # 3. Conv1d + Interpolate 処理 (元コードと同じ)
        x = x.transpose(0, 1).unsqueeze(0)  # (1, embed_dim, T_frames)
        x = self.conv(x)  # (1, conv_channels, T_frames)
        x = x.transpose(1, 2).squeeze(0)  # (T_frames, conv_channels)

        x = self.proj_out(x)  # (T_frames, output_dim)
        x = x.transpose(0, 1)  # (output_dim, T_frames)

        # 4. target_size にリサイズ
        x = F.interpolate(
            x.unsqueeze(0),
            size=target_size,
            mode="linear",
            align_corners=False,
        )  # (1, output_dim, target_size)

        attention_mask = torch.ones(1, target_size, device=target_device)

        return x, attention_mask


class ChromaChordConditioner(Conditioner):
    """
    Coco-Mullaの 25次元 (Root 12 + Chroma 12 + NoChord 1) 表現を使用。
    入力: (T_frames, 3) [root, quality_idx, inversion]
    """

    def __init__(
        self,
        output_dim: int = 64,
        embed_dim: int = 128,
        conv_channels: int = 32,
        conv_kernel_size: int = 7,
        conv_padding: int = 3,
    ):
        super().__init__(conv_channels, output_dim)

        # 25次元 (Root 12 + Chroma 12 + NoChord 1)
        self.input_dim = 25

        # 25次元ベクトルをembed_dimに射影する層
        self.input_proj = nn.Linear(self.input_dim, embed_dim)

        self.conv = nn.Conv1d(
            embed_dim, conv_channels, kernel_size=conv_kernel_size, padding=conv_padding
        )

        # --- 和音構成音 (Chroma) のルックアップテーブルを作成 ---
        quality_map_tensor = torch.zeros(NUM_CHORD_QUALITIES, 12)

        for i, name in enumerate(QUALITY_NAMES):
            intervals = QUALITY_CHROMA_INTERVALS.get(name)
            if intervals:
                # 構成音のインデックス (タプル) をリストに変換して設定
                quality_map_tensor[i, list(intervals)] = 1.0

        # モデルのバッファとして登録 (GPU対応)
        self.register_buffer("chord_quality_map", quality_map_tensor.float())
        # ----------------------------------------------------

    def encode_chords(self, chords_data: torch.Tensor, device: torch.device):
        """
        [root, quality_idx, inversion] から 25次元のCoco-Mulla表現 (c_i) を構築。

        Args:
            chords_data (torch.Tensor): (T_frames, 3)
                [root, quality_idx, inversion]
                root: 0-11 (N/A=-1)
                quality_idx: 0-13 (N/A=-1) (QUALITY_NAMESのインデックス)
        Returns:
            torch.Tensor: (T_frames, 25)
        """
        T_frames = chords_data.shape[0]
        c = torch.zeros(T_frames, self.input_dim, device=device)

        # 'no-chord' (root == -1) のフレームを特定
        no_chord_mask = chords_data[:, 0] == -1
        chord_mask = ~no_chord_mask

        # 1. No-Chord フレーム (論文 Eq. 1 の 'otherwise' ケース)
        c[no_chord_mask, 24] = 1.0  # no-chord フラグ (最後の次元) を立てる

        # 2. Chord フレーム
        if torch.any(chord_mask):
            roots = chords_data[chord_mask, 0].long()
            quality_idx = chords_data[chord_mask, 1].long()

            # Root (0-11)
            c[chord_mask, 0:12] = F.one_hot(roots, num_classes=12).float()

            # Chroma (12-23)
            # (rootを基準とした構成音)
            c[chord_mask, 12:24] = self.chord_quality_map[quality_idx]

        return c

    def forward(self, chords: tp.Any, device: tp.Union[torch.device, str]) -> tp.Any:
        """
        chords (dict):
            "data" (Tensor): (T_frames, 3) [root, quality_idx, inversion]
            "target_size" (int): F.interpolate の目標フレーム数
        """
        # create_chord_tensor からの出力を想定
        chords_data = chords[0]["data"].to(device)
        target_size = chords[0]["target_size"]

        # 1. Coco-Mulla 表現 (25-dim) にエンコード
        # (T_frames, 25)
        x = self.encode_chords(chords_data, device)

        # 2. embed_dim に射影
        # (T_frames, embed_dim)
        x = self.input_proj(x)

        # 3. Conv1d + Interpolate 処理 (元コードと同じ)
        x = x.transpose(0, 1).unsqueeze(0)  # (1, embed_dim, T_frames)
        x = self.conv(x)  # (1, conv_channels, T_frames)
        x = x.transpose(1, 2).squeeze(0)  # (T_frames, conv_channels)

        x = self.proj_out(x)  # (T_frames, output_dim)
        x = x.transpose(0, 1)  # (output_dim, T_frames)

        # 4. target_size にリサイズ
        x = F.interpolate(
            x.unsqueeze(0),
            size=target_size,
            mode="linear",
            align_corners=False,
        )  # (1, output_dim, target_size)

        attention_mask = torch.ones(1, target_size, device=device)

        return x, attention_mask
