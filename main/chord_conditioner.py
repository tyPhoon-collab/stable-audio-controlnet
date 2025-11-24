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


class CocoMullaChordConditioner(Conditioner):
    """
    柔軟な和音表現コンディショナー

    入力: (T_frames, 3) [root, quality_idx, bass_note]
        root: 0-11 (C-B) or -1 (N)
        quality_idx: 0-(NUM_CHORD_QUALITIES-1) or -1 (N)
        bass_note: 0-11 (C-B) or -1 (no bass/N)

    出力次元数:
        - with_bass=True:  37 (Root 12 + Bass 12 + Chroma 12 + NoChord 1)
        - with_bass=False: 25 (Root 12 + Chroma 12 + NoChord 1)

    パラメータ:
        output_dim: 最終出力次元 (デフォルト: 64)
        embed_dim: 内部埋め込み次元 (デフォルト: 128)
        conv_channels: Conv1d出力チャネル (デフォルト: 32)
        conv_kernel_size: Conv1dカーネルサイズ (デフォルト: 7)
        conv_padding: Conv1dパディング (デフォルト: 3)
        with_bass: bass noteを含める (デフォルト: True)
        rotate_chroma: chromaをrootベースで回転させる (デフォルト: False)
    """

    def __init__(
        self,
        output_dim: int = 64,
        embed_dim: int = 128,
        conv_channels: int = 32,
        conv_kernel_size: int = 7,
        conv_padding: int = 3,
        with_bass: bool = True,
        rotate_chroma: bool = False,
    ):
        super().__init__(conv_channels, output_dim)

        self.with_bass = with_bass
        self.rotate_chroma = rotate_chroma

        # 入力次元を決定
        if with_bass:
            self.input_dim = 37  # Root 12 + Bass 12 + Chroma 12 + NoChord 1
        else:
            self.input_dim = 25  # Root 12 + Chroma 12 + NoChord 1

        # 入力射影層
        self.input_proj = nn.Linear(self.input_dim, embed_dim)

        self.conv = nn.Conv1d(
            embed_dim, conv_channels, kernel_size=conv_kernel_size, padding=conv_padding
        )

        # --- 和音構成音 (Chroma) のルックアップテーブルを作成 ---
        quality_map_tensor = torch.zeros(NUM_CHORD_QUALITIES, 12)

        for i, name in enumerate(QUALITY_NAMES):
            intervals = QUALITY_CHROMA_INTERVALS.get(name)
            if intervals:
                # intervalをmodulo 12してchroma spaceに射影
                chroma_indices = [interval % 12 for interval in intervals]
                quality_map_tensor[i, chroma_indices] = 1.0

        # モデルのバッファとして登録 (GPU対応)
        self.register_buffer("chord_quality_map", quality_map_tensor)

    def encode_chords(
        self, chords_data: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        [root, quality_idx, bass_note] から 37/25次元のCoco-Mulla表現を構築。

        Args:
            chords_data (torch.Tensor): (T_frames, 3) or (batch_size, T_frames, 3)
                [root, quality_idx, bass_note]
            device (torch.device): 処理デバイス

        Returns:
            torch.Tensor: (T_frames, 37 or 25) or (batch_size, T_frames, 37 or 25)
        """
        # バッチ次元の有無を確認
        if len(chords_data.shape) == 2:
            # (T_frames, 3) の場合、バッチ次元を追加
            chords_data = chords_data.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, T_frames, _ = chords_data.shape
        c = torch.zeros(batch_size, T_frames, self.input_dim, device=device)

        # 'no-chord' (root == -1) のフレームを特定
        no_chord_mask = chords_data[:, :, 0] == -1  # (batch_size, T_frames)
        chord_mask = ~no_chord_mask

        # 1. No-Chord フレーム
        no_chord_idx = 36 if self.with_bass else 24
        c[no_chord_mask, no_chord_idx] = 1.0

        # 2. Chord フレーム
        if torch.any(chord_mask):
            # 全データを一度にlong型に変換
            roots = chords_data[:, :, 0].long()  # (batch_size, T_frames)
            quality_idx = chords_data[:, :, 1].long()  # (batch_size, T_frames)
            bass_notes = chords_data[:, :, 2].long()  # (batch_size, T_frames)

            # Root one-hot encoding (0-11)
            # マスク部分は-1なので、clampして0-11の範囲にしてからone-hot化
            roots_safe = roots.clamp(0, 11)
            root_one_hot = F.one_hot(
                roots_safe, num_classes=12
            ).float()  # (batch_size, T_frames, 12)
            # chord_maskの部分のみ適用
            c[:, :, 0:12] = torch.where(
                chord_mask.unsqueeze(-1), root_one_hot, c[:, :, 0:12]
            )

            if self.with_bass:
                # Bass one-hot encoding (12-23)
                bass_safe = bass_notes.clamp(0, 11)
                bass_one_hot = F.one_hot(bass_safe, num_classes=12).float()
                c[:, :, 12:24] = torch.where(
                    chord_mask.unsqueeze(-1), bass_one_hot, c[:, :, 12:24]
                )

                # Chroma (24-35)
                self._encode_chroma_batch(
                    c, chord_mask, roots, quality_idx, start_idx=24
                )
            else:
                # Chroma (12-23)
                self._encode_chroma_batch(
                    c, chord_mask, roots, quality_idx, start_idx=12
                )

        if squeeze_output:
            c = c.squeeze(0)

        return c

    def _encode_chroma_batch(
        self,
        c: torch.Tensor,
        mask: torch.Tensor,
        roots: torch.Tensor,
        quality_idx: torch.Tensor,
        start_idx: int,
    ) -> None:
        """
        Chroma情報をCoco-Mulla表現に追加（バッチ対応版）。

        Args:
            c: 出力ベクトル (batch_size, T_frames, input_dim)
            mask: Chordマスク (batch_size, T_frames)
            roots: ルート (batch_size, T_frames)
            quality_idx: 質インデックス (batch_size, T_frames)
            start_idx: Chromaの開始インデックス (12 or 24)
        """
        # chord_quality_mapを同じデバイスに移動
        chord_quality_map = tp.cast(torch.Tensor, self.chord_quality_map).to(c.device)

        # quality_idxを安全な範囲にclamp
        quality_idx_safe = quality_idx.clamp(0, NUM_CHORD_QUALITIES - 1)
        base_chroma = chord_quality_map[quality_idx_safe]  # (batch_size, T_frames, 12)

        if self.rotate_chroma:
            # rootベースで回転
            pitch_classes = torch.arange(12, device=c.device)  # (12,)
            # (batch_size, T_frames, 12) の形状で計算
            roll_indices = (
                pitch_classes.unsqueeze(0).unsqueeze(0) - roots.unsqueeze(-1)
            ) % 12
            rotated_chroma = torch.gather(
                base_chroma, 2, roll_indices
            )  # (batch_size, T_frames, 12)
            # maskの部分のみ適用
            c[:, :, start_idx : start_idx + 12] = torch.where(
                mask.unsqueeze(-1), rotated_chroma, c[:, :, start_idx : start_idx + 12]
            )
        else:
            # 回転しない (絶対的なchroma)
            c[:, :, start_idx : start_idx + 12] = torch.where(
                mask.unsqueeze(-1), base_chroma, c[:, :, start_idx : start_idx + 12]
            )

    def forward(self, chords: tp.Any, device: tp.Union[torch.device, str]) -> tp.Any:
        """
        Args:
            chords (dict): {"data": Tensor (T_frames, 3) or (batch_size, T_frames, 3), "target_size": int}
            device: 処理デバイス

        Returns:
            Tuple[Tensor, Tensor]: (conditioning, attention_mask)
        """
        target_device = torch.device(device) if isinstance(device, str) else device

        chords_data = chords[0]["data"].to(target_device)
        target_size = chords[0]["target_size"]

        # バッチ次元の有無を確認
        if len(chords_data.shape) == 2:
            # (T_frames, 3) の場合、バッチ次元を追加
            chords_data = chords_data.unsqueeze(0)

        batch_size, T_frames, _ = chords_data.shape

        # 1. バッチ全体を一度にCoco-Mulla表現にエンコード（ループなし）
        # (batch_size, T_frames, 3) -> (batch_size, T_frames, input_dim)
        x = self.encode_chords(chords_data, target_device)

        # 2. embed_dim に射影: (batch_size, T_frames, input_dim) -> (batch_size, T_frames, embed_dim)
        x = self.input_proj(x)

        # 3. Conv1d処理のため形状変換: (batch_size, T_frames, embed_dim) -> (batch_size, embed_dim, T_frames)
        x = x.transpose(1, 2)

        # 4. Conv1d + projection: (batch_size, embed_dim, T_frames) -> (batch_size, conv_channels, T_frames)
        x = self.conv(x)

        # 5. 形状変換とprojection: (batch_size, conv_channels, T_frames) -> (batch_size, T_frames, conv_channels)
        x = x.transpose(1, 2)
        x = self.proj_out(x)  # (batch_size, T_frames, output_dim)

        # 6. 最終形状変換: (batch_size, T_frames, output_dim) -> (batch_size, output_dim, T_frames)
        x = x.transpose(1, 2)

        # 7. target_sizeにリサイズ: (batch_size, output_dim, T_frames) -> (batch_size, output_dim, target_size)
        x = F.interpolate(
            x,
            size=target_size,
            mode="linear",
            align_corners=False,
        )

        attention_mask = torch.ones(batch_size, target_size, device=target_device)

        return x, attention_mask
