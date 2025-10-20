"""
Chord Representation Module

和音をニューラルネットワークで処理可能な形式に変換するモジュール。
異なる表現方法（OneHot、Embedding、簡略版など）を統一的なインターフェースで提供。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChordRepresentation(ABC, nn.Module):
    """
    和音表現の抽象基底クラス

    すべての和音表現実装は、このインターフェースに従う必要があります。
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_output_channels(self) -> int:
        """出力チャンネル数を返す"""
        pass

    @abstractmethod
    def forward(self, chord_batch: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        和音バッチを表現ベクトルに変換

        Args:
            chord_batch: (B, T_frames, 3) [root, quality, inversion]
            target_length: リサンプリング後の時間長（Noneの場合はT_framesのまま）

        Returns:
            torch.Tensor: (B, channels, target_length or T_frames)
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict:
        """設定情報を辞書で返す（ログ・デバッグ用）"""
        pass


class OneHotChordRepresentation(ChordRepresentation):
    """
    One-Hot エンコーディングによる和音表現

    - Root: 13チャンネル (C, C#, ..., B, N)
    - Quality: 10チャンネル (maj, min, maj7, min7, 7, dim, aug, sus4, sus2, N)
    - 合計: 23チャンネル
    """

    def __init__(self):
        super().__init__()
        self.num_roots = 13  # 0-11 + N
        self.num_qualities = 10  # 0-8 + N
        self.output_channels = self.num_roots + self.num_qualities

    def get_output_channels(self) -> int:
        return self.output_channels

    def forward(self, chord_batch: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        和音バッチをOne-Hot表現に変換

        Args:
            chord_batch: (B, T_frames, 3) [root, quality, inversion]
                        inversion は無視される
            target_length: リサンプリング後の時間長（Noneの場合はT_framesのまま）

        Returns:
            torch.Tensor: (B, 23, target_length or T_frames)
        """
        B, T, _ = chord_batch.shape
        device = chord_batch.device

        if chord_batch.numel() == 0:
            raise ValueError("Empty chord_batch provided")

        assert chord_batch.shape[-1] == 3, \
            f"Expected chord_batch shape (B, T, 3), got {chord_batch.shape}"

        root = chord_batch[..., 0].clone()
        qual = chord_batch[..., 1].clone()
        # inversion は使用しない

        # -1 (無和音) を最後のインデックスにマップ
        root_idx = torch.where(
            root >= 0, root, torch.full_like(root, 12)
        )  # 0..11, 12 for N
        qual_idx = torch.where(
            qual >= 0, qual, torch.full_like(qual, 9)
        )  # 0..8, 9 for N

        # One-Hotテンソルを作成
        out = torch.zeros(
            (B, self.output_channels, T),
            device=device,
            dtype=torch.float32
        )

        # Root部分
        out_root = out[:, 0:self.num_roots]
        out_root.scatter_(1, root_idx.long().unsqueeze(1), 1.0)

        # Quality部分
        out_qual = out[:, self.num_roots:self.num_roots + self.num_qualities]
        out_qual.scatter_(1, qual_idx.long().unsqueeze(1), 1.0)

        # target_lengthが指定されていればリサンプリング
        if target_length is not None and target_length != T:
            out = F.interpolate(out, size=target_length, mode='nearest')

        return out

    def get_config(self) -> Dict:
        return {
            "type": "onehot",
            "num_roots": self.num_roots,
            "num_qualities": self.num_qualities,
            "output_channels": self.output_channels,
        }


class SimpleOneHotChordRepresentation(ChordRepresentation):
    """
    シンプルなOne-Hotエンコーディングによる和音表現

    - 12音 × 2質(maj, min) + 1(N) = 25チャンネル
    - メジャー: 0-11 (C, C#, ..., B)
    - マイナー: 12-23 (Cm, C#m, ..., Bm)
    - 無和音: 24 (N)

    データセット側で和音を(maj, min, N)に簡略化済みであることを想定。
    """

    def __init__(self):
        super().__init__()
        self.num_roots = 12  # C-B
        self.num_qualities = 2  # maj, min (Nは別)
        self.output_channels = self.num_roots * self.num_qualities + 1  # 25

    def get_output_channels(self) -> int:
        return self.output_channels

    def forward(self, chord_batch: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        和音バッチをシンプルなOne-Hot表現に変換

        Args:
            chord_batch: (B, T_frames, 3) [root, quality, inversion]
                        root: 0-11 (C-B) or -1 (N)
                        quality: 0 (maj), 1 (min) or -1 (N)
                        inversion: 無視
            target_length: リサンプリング後の時間長（Noneの場合はT_framesのまま）

        Returns:
            torch.Tensor: (B, 25, target_length or T_frames)
        """
        B, T, _ = chord_batch.shape
        device = chord_batch.device

        if chord_batch.numel() == 0:
            raise ValueError("Empty chord_batch provided")

        assert chord_batch.shape[-1] == 3, \
            f"Expected chord_batch shape (B, T, 3), got {chord_batch.shape}"

        root = chord_batch[..., 0].clone().long()  # (B, T)
        qual = chord_batch[..., 1].clone().long()  # (B, T)

        # One-Hotテンソルを作成
        out = torch.zeros(
            (B, self.output_channels, T),
            device=device,
            dtype=torch.float32
        )

        # 無和音判定: root == -1 or qual == -1
        is_no_chord = (root == -1) | (qual == -1)

        # 有効な和音のマスク
        valid_mask = ~is_no_chord

        # 有効な和音のインデックス計算
        # maj: root (0-11)
        # min: 12 + root (12-23)
        chord_idx = torch.where(
            qual == 0,  # maj
            root,
            12 + root   # min
        )

        # 有効な和音をセット
        if valid_mask.any():
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, T)[valid_mask]
            time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T)[valid_mask]
            chord_indices = chord_idx[valid_mask]
            out[batch_indices, chord_indices, time_indices] = 1.0

        # 無和音をセット (インデックス24)
        if is_no_chord.any():
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, T)[is_no_chord]
            time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T)[is_no_chord]
            out[batch_indices, 24, time_indices] = 1.0

        # target_lengthが指定されていればリサンプリング
        if target_length is not None and target_length != T:
            out = F.interpolate(out, size=target_length, mode='nearest')

        return out

    def get_config(self) -> Dict:
        return {
            "type": "onehot",
            "num_roots": self.num_roots,
            "num_qualities": self.num_qualities,
            "output_channels": self.output_channels,
            "description": "12 roots × 2 qualities (maj, min) + 1 (N) = 25 channels"
        }


class SimpleEmbeddingChordRepresentation(ChordRepresentation):
    """
    シンプルなEmbeddingによる和音表現

    - 25種類のコードを直接Embeddingで表現
    - メジャー: 0-11 (C, C#, ..., B)
    - マイナー: 12-23 (Cm, C#m, ..., Bm)
    - 無和音: 24 (N)
    - 音楽理論に基づいた初期化（Circle of Fifths、明暗）
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_chords = 25  # 12 maj + 12 min + 1 N

        # 直接25種類のコードをEmbedding
        self.chord_embedding = nn.Embedding(self.num_chords, embed_dim)

        # 音楽理論に基づく初期化
        self._init_embeddings()

    def get_output_channels(self) -> int:
        return self.embed_dim

    def _init_embeddings(self):
        """音楽理論に基づいた初期化"""
        # Circle of Fifths: C(0) -> G(7) -> D(2) -> ...
        circle_of_fifths = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

        if self.embed_dim >= 3:
            for position, root in enumerate(circle_of_fifths):
                angle = 2 * torch.pi * position / 12

                # メジャー (0-11): 明るさ +1
                maj_vec = torch.zeros(self.embed_dim)
                maj_vec[0] = torch.cos(torch.tensor(angle)) * 0.1
                maj_vec[1] = torch.sin(torch.tensor(angle)) * 0.1
                maj_vec[2] = 0.1  # 明るさ
                if self.embed_dim > 3:
                    maj_vec[3:] = torch.randn(self.embed_dim - 3) * 0.01
                self.chord_embedding.weight.data[root] = maj_vec

                # マイナー (12-23): 明るさ -1
                min_vec = torch.zeros(self.embed_dim)
                min_vec[0] = torch.cos(torch.tensor(angle)) * 0.1
                min_vec[1] = torch.sin(torch.tensor(angle)) * 0.1
                min_vec[2] = -0.1  # 暗さ
                if self.embed_dim > 3:
                    min_vec[3:] = torch.randn(self.embed_dim - 3) * 0.01
                self.chord_embedding.weight.data[12 + root] = min_vec
        else:
            nn.init.normal_(self.chord_embedding.weight, std=0.02)

        # 無和音 (24) は零ベクトル
        self.chord_embedding.weight.data[24] = torch.zeros(self.embed_dim)

    def forward(self, chord_batch: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        和音バッチをシンプルなEmbedding表現に変換

        Args:
            chord_batch: (B, T_frames, 3) [root, quality, inversion]
                        root: 0-11 (C-B) or -1 (N)
                        quality: 0 (maj), 1 (min) or -1 (N)
                        inversion: 無視
            target_length: リサンプリング後の時間長（Noneの場合はT_framesのまま）

        Returns:
            torch.Tensor: (B, embed_dim, target_length or T_frames)
        """
        B, T, _ = chord_batch.shape

        if chord_batch.numel() == 0:
            raise ValueError("Empty chord_batch provided")

        assert chord_batch.shape[-1] == 3, \
            f"Expected chord_batch shape (B, T, 3), got {chord_batch.shape}"

        root = chord_batch[..., 0].clone().long()  # (B, T)
        qual = chord_batch[..., 1].clone().long()  # (B, T)

        # 無和音判定: root == -1 or qual == -1
        is_no_chord = (root == -1) | (qual == -1)

        # コードインデックス計算
        # maj: root (0-11)
        # min: 12 + root (12-23)
        # N: 24
        chord_idx = torch.where(
            is_no_chord,
            torch.full_like(root, 24),  # N
            torch.where(
                qual == 0,  # maj
                root,
                12 + root   # min
            )
        )

        # インデックス範囲チェック
        assert chord_idx.max() <= 24 and chord_idx.min() >= 0, \
            f"Invalid chord indices: [{chord_idx.min()}, {chord_idx.max()}]"

        # Embedding lookup
        with torch.cuda.amp.autocast(enabled=True):
            chord_emb = self.chord_embedding(chord_idx)  # (B, T, embed_dim)

        # (B, embed_dim, T) に転置
        out = chord_emb.transpose(1, 2)

        # target_lengthが指定されていればリサンプリング（linear補間）
        if target_length is not None and target_length != T:
            out = F.interpolate(out, size=target_length, mode='linear', align_corners=False)

        return out

    def get_config(self) -> Dict:
        return {
            "type": "embedding",
            "embed_dim": self.embed_dim,
            "num_chords": self.num_chords,
            "output_channels": self.embed_dim,
            "description": "25 chords (12 maj + 12 min + 1 N) with embedding"
        }


class EmbeddingChordRepresentation(ChordRepresentation):
    """
    学習可能なEmbeddingによる和音表現

    - Root Embedding: 13種類 (C-B + N)
    - Quality Embedding: 10種類 (maj, min, ..., N)
    - 音楽理論に基づいた初期化（Circle of Fifths、明暗・緊張度）
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_roots = 13
        self.num_qualities = 10

        # Embedding層
        self.root_embedding = nn.Embedding(self.num_roots, embed_dim)
        self.quality_embedding = nn.Embedding(self.num_qualities, embed_dim)

        # 結合層（Root + Quality → 統合表現）
        self.chord_combiner = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),  # Swish activation
        )

        # 音楽理論に基づく初期化
        self._init_embeddings()

    def get_output_channels(self) -> int:
        return self.embed_dim

    def _init_embeddings(self):
        """音楽理論に基づいた初期化"""
        self._init_root_embeddings()
        self._init_quality_embeddings()

    def _init_root_embeddings(self):
        """Circle of Fifthsに基づくRoot初期化"""
        # 5度圏: C(0) -> G(7) -> D(2) -> A(9) -> E(4) -> B(11)
        #      -> F#(6) -> C#(1) -> G#(8) -> D#(3) -> A#(10) -> F(5)
        circle_of_fifths = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

        for position, note_index in enumerate(circle_of_fifths):
            angle = 2 * torch.pi * position / 12
            if self.embed_dim >= 2:
                init_vec = torch.zeros(self.embed_dim)
                init_vec[0] = torch.cos(torch.tensor(angle)) * 0.1
                init_vec[1] = torch.sin(torch.tensor(angle)) * 0.1
                if self.embed_dim > 2:
                    # 高次元はランダムノイズ
                    init_vec[2:] = torch.randn(self.embed_dim - 2) * 0.01
                self.root_embedding.weight.data[note_index] = init_vec

        # 無和音 (N) は零ベクトル
        self.root_embedding.weight.data[12] = torch.zeros(self.embed_dim)

    def _init_quality_embeddings(self):
        """音楽的意味に基づくQuality初期化"""
        if self.embed_dim >= 2:
            # [明るさ, 緊張度] の2次元で初期化
            quality_meanings = torch.tensor([
                [1.0, 0.0],    # maj - 明るい、安定
                [-1.0, 0.0],   # min - 暗い、安定
                [0.0, 1.0],    # maj7 - 中立、洗練
                [0.0, -1.0],   # min7 - 中立、ジャジー
                [0.5, 0.5],    # 7 (dom7) - やや明るい、ブルージー
                [-0.5, 0.5],   # dim - やや暗い、緊張
                [0.8, 0.2],    # aug - 明るい、不安定
                [0.2, 0.8],    # sus4 - やや明るい、浮遊感
                [0.0, 0.0],    # sus2 - 中立
            ])

            self.quality_embedding.weight.data[:9, :2] = quality_meanings * 0.1
            if self.embed_dim > 2:
                self.quality_embedding.weight.data[:9, 2:] = \
                    torch.randn(9, self.embed_dim - 2) * 0.01
        else:
            nn.init.normal_(self.quality_embedding.weight, std=0.02)

        # 無和音 quality は零ベクトル
        self.quality_embedding.weight.data[9] = torch.zeros(self.embed_dim)

    def forward(self, chord_batch: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        和音バッチをEmbedding表現に変換

        Args:
            chord_batch: (B, T_frames, 3) [root, quality, inversion]
                        inversion は無視される
            target_length: リサンプリング後の時間長（Noneの場合はT_framesのまま）

        Returns:
            torch.Tensor: (B, embed_dim, target_length or T_frames)
        """
        B, T, _ = chord_batch.shape

        if chord_batch.numel() == 0:
            raise ValueError("Empty chord_batch provided")

        assert chord_batch.shape[-1] == 3, \
            f"Expected chord_batch shape (B, T, 3), got {chord_batch.shape}"

        root = chord_batch[..., 0].clone()
        qual = chord_batch[..., 1].clone()
        # inversion は使用しない

        # -1 を特殊インデックスにマップ
        root_idx = torch.where(root >= 0, root, torch.full_like(root, 12))
        qual_idx = torch.where(qual >= 0, qual, torch.full_like(qual, 9))

        # インデックス範囲チェック
        assert root_idx.max() <= 12 and root_idx.min() >= 0, \
            f"Invalid root indices: [{root_idx.min()}, {root_idx.max()}]"
        assert qual_idx.max() <= 9 and qual_idx.min() >= 0, \
            f"Invalid quality indices: [{qual_idx.min()}, {qual_idx.max()}]"

        # Embedding lookup
        with torch.cuda.amp.autocast(enabled=True):
            root_emb = self.root_embedding(root_idx.long())  # (B, T, embed_dim)
            qual_emb = self.quality_embedding(qual_idx.long())  # (B, T, embed_dim)

            # 結合して統合表現を生成
            combined = torch.cat([root_emb, qual_emb], dim=-1)  # (B, T, 2*embed_dim)
            chord_emb = self.chord_combiner(combined)  # (B, T, embed_dim)

        # (B, embed_dim, T) に転置
        out = chord_emb.transpose(1, 2)

        # target_lengthが指定されていればリサンプリング（linear補間）
        if target_length is not None and target_length != T:
            out = F.interpolate(out, size=target_length, mode='linear', align_corners=False)

        return out

    def get_config(self) -> Dict:
        return {
            "type": "embedding",
            "embed_dim": self.embed_dim,
            "num_roots": self.num_roots,
            "num_qualities": self.num_qualities,
            "output_channels": self.embed_dim,
        }

