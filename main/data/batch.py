"""統一バッチ構造の定義

データパイプライン全体で使用する統一されたバッチ形式を提供します。
ステムをそのまま保持し、Transform パイプラインで変換処理を分離することで、
データ拡張の追加や新しい条件付けの導入を容易にします。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import torch


class AudioBatchDict(TypedDict, total=False):
    """統一バッチ構造（TypedDict版）

    辞書互換のバッチ形式。Hydra/OmegaConfとの互換性が高い。
    """

    # === 必須フィールド ===
    stems: list[dict[str, torch.Tensor]]  # 各サンプルのステム辞書 {stem_name: (C, T)}
    sample_keys: list[str]  # サンプル識別子

    # === 時間情報 ===
    start_seconds: list[float]  # 開始時刻
    total_seconds: list[float]  # 総時間

    # === テキスト情報 ===
    prompts: list[str]  # テキストプロンプト

    # === 条件付け情報（オプショナル）===
    chord: torch.Tensor | None  # (B, T, 3) [root, quality, inversion]
    melody: torch.Tensor | None  # (B, bins, T)
    chroma: torch.Tensor | None  # (B, 12, T)

    # === ミックス済み音声（Transform適用後）===
    audio: torch.Tensor | None  # (B, C, T) ミックス済み出力
    audio_input: torch.Tensor | None  # (B, C, T) 入力音声（conditional用）


@dataclass
class AudioBatch:
    """統一バッチ構造（dataclass版）

    型安全性とvalidationが可能なバッチ形式。
    nn.Moduleでの処理に適している。
    """

    # === 必須フィールド ===
    stems: list[dict[str, torch.Tensor]]  # 各サンプルのステム辞書 {stem_name: (C, T)}
    sample_keys: list[str]  # サンプル識別子

    # === 時間情報 ===
    start_seconds: list[float] = field(default_factory=list)
    total_seconds: list[float] = field(default_factory=list)

    # === テキスト情報 ===
    prompts: list[str] = field(default_factory=list)

    # === 条件付け情報（オプショナル）===
    chord: torch.Tensor | None = None  # (B, T, 3)
    melody: torch.Tensor | None = None  # (B, bins, T)
    chroma: torch.Tensor | None = None  # (B, 12, T)

    # === ミックス済み音声（Transform適用後）===
    audio: torch.Tensor | None = None  # (B, C, T) ミックス済み出力
    audio_input: torch.Tensor | None = None  # (B, C, T) 入力音声

    @property
    def batch_size(self) -> int:
        """バッチサイズを取得"""
        return len(self.sample_keys)

    def to(self, device: torch.device | str) -> AudioBatch:
        """バッチ内のテンソルを指定デバイスに移動"""

        def move_stems(stems_list: list[dict[str, torch.Tensor]]):
            return [{k: v.to(device) for k, v in stems.items()} for stems in stems_list]

        def move_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(device) if t is not None else None

        return AudioBatch(
            stems=move_stems(self.stems),
            sample_keys=self.sample_keys,
            start_seconds=self.start_seconds,
            total_seconds=self.total_seconds,
            prompts=self.prompts,
            chord=move_tensor(self.chord),
            melody=move_tensor(self.melody),
            chroma=move_tensor(self.chroma),
            audio=move_tensor(self.audio),
            audio_input=move_tensor(self.audio_input),
        )

    def get_stem_names(self) -> set[str]:
        """バッチ内の全ステム名を取得"""
        names: set[str] = set()
        for stems in self.stems:
            names.update(stems.keys())
        return names

    def has_audio(self) -> bool:
        """ミックス済み音声が存在するか"""
        return self.audio is not None

    def validate(self) -> None:
        """バッチの整合性を検証"""
        batch_size = self.batch_size

        # stemsは空（ミックス済みで破棄された場合）か、batch_sizeと一致する必要がある
        if self.stems and len(self.stems) != batch_size:
            raise ValueError(
                f"stems length mismatch: {len(self.stems)} vs {batch_size}"
            )

        if self.start_seconds and len(self.start_seconds) != batch_size:
            raise ValueError(
                f"start_seconds length mismatch: {len(self.start_seconds)} vs {batch_size}"
            )

        if self.total_seconds and len(self.total_seconds) != batch_size:
            raise ValueError(
                f"total_seconds length mismatch: {len(self.total_seconds)} vs {batch_size}"
            )

        if self.prompts and len(self.prompts) != batch_size:
            raise ValueError(
                f"prompts length mismatch: {len(self.prompts)} vs {batch_size}"
            )

        if self.chord is not None and self.chord.shape[0] != batch_size:
            raise ValueError(
                f"chord batch size mismatch: {self.chord.shape[0]} vs {batch_size}"
            )

        if self.audio is not None and self.audio.shape[0] != batch_size:
            raise ValueError(
                f"audio batch size mismatch: {self.audio.shape[0]} vs {batch_size}"
            )
