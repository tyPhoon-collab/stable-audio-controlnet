"""バッチ変換パイプライン

AudioBatchに対する変換処理を提供します。
データ拡張、ステムミックス、プロンプト生成などをモジュール化し、
柔軟に組み合わせ可能な設計です。
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import pitch_shift

from .annotation import NUM_CHORD_ROOTS
from .batch import AudioBatch
from .common_mapping import DescriptionMapping, GenreMapping


class BatchTransform(ABC):
    """バッチ変換の基底クラス"""

    @abstractmethod
    def __call__(self, batch: AudioBatch) -> AudioBatch:
        """バッチを変換する"""
        raise NotImplementedError


class Compose(BatchTransform):
    """複数のTransformを順次適用"""

    def __init__(self, transforms: list[BatchTransform]):
        self.transforms = transforms

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        for transform in self.transforms:
            batch = transform(batch)
        return batch


class Identity(BatchTransform):
    """何もしない変換（テスト用）"""

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        return batch


# =============================================================================
# ステム操作
# =============================================================================


class VocalsDropTransform(BatchTransform):
    """ボーカルトラックを除去"""

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        new_stems = []
        for stems in batch.stems:
            new_stems.append({k: v for k, v in stems.items() if k != "vocals"})
        batch.stems = new_stems
        return batch


class StemMixTransform(BatchTransform):
    """ステムをミックスしてaudioフィールドに格納

    Args:
        strategy: ミックス戦略
            - "all": 全ステムをミックス
            - "random_subset": ランダムに選択したステムをミックス
        drop_vocals: ミックス前にボーカルを除去するか
        keep_stems: ミックス後にステムを保持するか（メモリ節約のためFalse推奨）
    """

    def __init__(
        self, strategy: str = "all", drop_vocals: bool = True, keep_stems: bool = True
    ):
        self.strategy = strategy
        self.drop_vocals = drop_vocals
        self.keep_stems = keep_stems

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        # ステムが空の場合（既にミックス済みでkeep_stems=Falseの場合など）
        if not batch.stems:
            if batch.audio is not None:
                return batch
            # ステムもAudioもない場合は、バッチサイズが0でない限りエラーの可能性が高いが、
            # ここでは空のAudioを作成して返すか、そのまま返す
            # バッチサイズ0の場合はそのまま返す
            if batch.batch_size == 0:
                return batch
            # バッチサイズがあるのにステムがない -> エラーだが、
            # torch.stackで落ちるよりは明確なエラーを出すか、あるいは何もしない
            # ここでは後続の処理に委ねる（ただしstackで落ちるのでチェックが必要）

        outputs = []

        for stems in batch.stems:
            if self.drop_vocals and "vocals" in stems:
                stems = {k: v for k, v in stems.items() if k != "vocals"}

            if self.strategy == "all":
                out_track = torch.stack(list(stems.values())).sum(dim=0)
            elif self.strategy == "random_subset":
                if len(stems) < 2:
                    out_track = torch.stack(list(stems.values())).sum(dim=0)
                else:
                    stem_keys = list(stems.keys())
                    k = random.randint(1, len(stem_keys))
                    selected = random.sample(stem_keys, k)
                    out_track = torch.stack([stems[s] for s in selected]).sum(dim=0)
            else:
                raise ValueError(f"Unknown mix strategy: {self.strategy}")

            outputs.append(out_track)

        if outputs:
            batch.audio = torch.stack(outputs)

        if not self.keep_stems:
            batch.stems = []
        return batch


class StemGainTransform(BatchTransform):
    """各ステムにランダムなゲインを適用

    Args:
        gain_min: 最小ゲイン（例: 0.7）
        gain_max: 最大ゲイン（例: 1.3）
        p: 適用確率
    """

    def __init__(self, gain_min: float = 0.7, gain_max: float = 1.3, p: float = 0.5):
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.p = p

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        if random.random() > self.p:
            return batch

        new_stems = []
        for stems in batch.stems:
            new_stem = {}
            for name, audio in stems.items():
                gain = random.uniform(self.gain_min, self.gain_max)
                new_stem[name] = audio * gain
            new_stems.append(new_stem)
        batch.stems = new_stems
        return batch


# =============================================================================
# プロンプト生成
# =============================================================================


class PromptTransform(BatchTransform):
    """固定プロンプトを設定"""

    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        batch.prompts = [self.prompt] * batch.batch_size
        return batch


class RandomPromptTransform(BatchTransform):
    """候補リストからランダムにプロンプトを選択"""

    def __init__(
        self,
        choices: list[str] | None = None,
    ):
        self.choices = choices or [
            "melodic music",
            "catchy song",
            "a song",
            "music tracks",
        ]

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        batch.prompts = [random.choice(self.choices) for _ in range(batch.batch_size)]
        return batch


class DescriptionPromptTransform(BatchTransform):
    """CSVファイルから説明文を取得してプロンプトに設定"""

    def __init__(self, csv_path: str, fallback: str = ""):
        self.mapper = DescriptionMapping()
        self.mapper.load_mapping(csv_path)
        self.fallback = fallback

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        prompts = []
        for sample_key in batch.sample_keys:
            desc = self.mapper.get_description(sample_key)
            prompts.append(desc if desc else self.fallback)
        batch.prompts = prompts
        return batch


class GenrePromptTransform(BatchTransform):
    """CSVファイルからジャンルを取得してプロンプトに設定"""

    def __init__(self, csv_path: str, fallback: str = "music"):
        self.mapper = GenreMapping()
        self.mapper.load_mapping(csv_path)
        self.fallback = fallback

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        prompts = []
        for sample_key in batch.sample_keys:
            genre = self.mapper.get_genre(sample_key)
            prompts.append(genre if genre else self.fallback)
        batch.prompts = prompts
        return batch


# =============================================================================
# データ拡張
# =============================================================================


class PitchShiftTransform(BatchTransform):
    """ピッチシフトによるデータ拡張

    音声と和音テンソルを同期してシフトする。

    Args:
        semitones_min: 最小シフト量（半音単位、例: -5）
        semitones_max: 最大シフト量（半音単位、例: 6）
        sample_rate: サンプリングレート
        p: 適用確率
    """

    def __init__(
        self,
        semitones_min: int = -5,
        semitones_max: int = 6,
        sample_rate: int = 44100,
        p: float = 0.5,
    ):
        self.semitones_min = semitones_min
        self.semitones_max = semitones_max
        self.sample_rate = sample_rate
        self.p = p

    def _shift_chord_tensor(
        self, chord_tensor: torch.Tensor, semitones: int
    ) -> torch.Tensor:
        """和音テンソルの根音をシフト

        Args:
            chord_tensor: (T, 3) [root, quality, inversion]
            semitones: シフト量

        Returns:
            シフト後の和音テンソル
        """
        shifted = chord_tensor.clone()
        root = shifted[:, 0]
        # -1（無音/N）はシフトしない
        valid_mask = root >= 0
        root[valid_mask] = (root[valid_mask] + semitones) % NUM_CHORD_ROOTS
        return shifted

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        if random.random() > self.p:
            return batch

        # バッチ内で同じシフト量を使用
        semitones = random.randint(self.semitones_min, self.semitones_max)
        if semitones == 0:
            return batch

        # Audioのピッチシフト
        if batch.audio is not None:
            batch.audio = pitch_shift(batch.audio, self.sample_rate, semitones)
        # ステムのピッチシフト（Audioがない場合のみ）
        elif batch.stems:
            new_stems = []
            for stems in batch.stems:
                new_stem = {}
                for name, audio in stems.items():
                    shifted = pitch_shift(audio, self.sample_rate, semitones)
                    new_stem[name] = shifted
                new_stems.append(new_stem)
            batch.stems = new_stems

        # 和音テンソルのシフト
        if batch.chord is not None:
            new_chords = []
            for i in range(batch.chord.shape[0]):
                shifted_chord = self._shift_chord_tensor(batch.chord[i], semitones)
                new_chords.append(shifted_chord)
            batch.chord = torch.stack(new_chords)

        return batch


class PerSamplePitchShiftTransform(BatchTransform):
    """サンプルごとに異なるピッチシフトを適用

    Args:
        semitones_min: 最小シフト量（半音単位）
        semitones_max: 最大シフト量（半音単位）
        sample_rate: サンプリングレート
        p: 各サンプルへの適用確率
    """

    def __init__(
        self,
        semitones_min: int = -5,
        semitones_max: int = 6,
        sample_rate: int = 44100,
        p: float = 0.5,
    ):
        self.semitones_min = semitones_min
        self.semitones_max = semitones_max
        self.sample_rate = sample_rate
        self.p = p

    def _shift_chord_tensor(
        self, chord_tensor: torch.Tensor, semitones: int
    ) -> torch.Tensor:
        shifted = chord_tensor.clone()
        root = shifted[:, 0]
        valid_mask = root >= 0
        root[valid_mask] = (root[valid_mask] + semitones) % NUM_CHORD_ROOTS
        return shifted

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        has_chord = batch.chord is not None

        # Audioのピッチシフト（優先）
        if batch.audio is not None:
            new_audios = []
            new_chords: list[torch.Tensor] = []

            for i in range(batch.batch_size):
                audio = batch.audio[i]

                # 確率判定
                if random.random() > self.p:
                    new_audios.append(audio)
                    if has_chord:
                        assert batch.chord is not None
                        new_chords.append(batch.chord[i])
                    continue

                semitones = random.randint(self.semitones_min, self.semitones_max)
                if semitones == 0:
                    new_audios.append(audio)
                    if has_chord:
                        assert batch.chord is not None
                        new_chords.append(batch.chord[i])
                    continue

                # ピッチシフトはGPUメモリを大量に消費するため、CPUで実行する
                # torchaudio.functional.pitch_shiftはリサンプリングを含み、
                # 長い音声（47秒など）の場合、GPU上で数GBの一時メモリを必要とすることがある
                original_device = audio.device
                audio_cpu = audio.cpu()
                shifted_cpu = pitch_shift(audio_cpu, self.sample_rate, semitones)
                shifted = shifted_cpu.to(original_device)

                new_audios.append(shifted)

                if has_chord:
                    assert batch.chord is not None
                    shifted_chord = self._shift_chord_tensor(batch.chord[i], semitones)
                    new_chords.append(shifted_chord)

            batch.audio = torch.stack(new_audios)
            if has_chord and new_chords:
                batch.chord = torch.stack(new_chords)

            return batch

        # ステムのピッチシフト（Audioがない場合）
        new_stems = []
        new_chords = []

        for i, stems in enumerate(batch.stems):
            if random.random() > self.p:
                new_stems.append(stems)
                if has_chord:
                    assert batch.chord is not None
                    new_chords.append(batch.chord[i])
                continue

            semitones = random.randint(self.semitones_min, self.semitones_max)
            if semitones == 0:
                new_stems.append(stems)
                if has_chord:
                    assert batch.chord is not None
                    new_chords.append(batch.chord[i])
                continue

            # ステムのピッチシフト
            new_stem = {}
            for name, audio in stems.items():
                shifted = pitch_shift(audio, self.sample_rate, semitones)
                new_stem[name] = shifted
            new_stems.append(new_stem)

            # 和音テンソルのシフト
            if has_chord:
                assert batch.chord is not None
                shifted_chord = self._shift_chord_tensor(batch.chord[i], semitones)
                new_chords.append(shifted_chord)

        batch.stems = new_stems
        if has_chord and new_chords:
            batch.chord = torch.stack(new_chords)

        return batch


class VarispeedPitchShiftTransform(BatchTransform):
    """リサンプリング（Varispeed）による軽量ピッチシフト

    ピッチと再生速度を同時に変更することで、計算コストを抑える。
    モデル入力長に合わせて、パディングまたはクロップを行う。
    和音テンソルも時間軸方向に伸縮させる。

    Args:
        semitones_min: 最小シフト量（半音単位）
        semitones_max: 最大シフト量（半音単位）
        sample_rate: サンプリングレート
        p: 適用確率
    """

    def __init__(
        self,
        semitones_min: int = -5,
        semitones_max: int = 6,
        sample_rate: int = 44100,
        p: float = 0.5,
    ):
        self.semitones_min = semitones_min
        self.semitones_max = semitones_max
        self.sample_rate = sample_rate
        self.p = p

    def _shift_chord_tensor(
        self, chord_tensor: torch.Tensor, semitones: int
    ) -> torch.Tensor:
        """和音テンソルの根音をシフト"""
        shifted = chord_tensor.clone()
        root = shifted[:, 0]
        valid_mask = root >= 0
        root[valid_mask] = (root[valid_mask] + semitones) % NUM_CHORD_ROOTS
        return shifted

    def _resample_chord_tensor(
        self, chord_tensor: torch.Tensor, ratio: float, target_frames: int
    ) -> torch.Tensor:
        """和音テンソルを時間軸方向にリサンプリング"""
        # (T, 3) -> (1, 3, T) for interpolate
        chord_t = chord_tensor.permute(1, 0).unsqueeze(0).float()

        # Nearest neighbor interpolation for categorical data
        # Calculate new length based on ratio
        # ratio > 1 (Pitch Up) -> Faster -> Shorter duration
        # ratio < 1 (Pitch Down) -> Slower -> Longer duration
        # We want to stretch/shrink the chord sequence to match the audio speed change.
        # If audio is played at speed `ratio`, the duration becomes `original_duration / ratio`.
        # So we need to resize the chord tensor to `original_frames / ratio`.

        original_frames = chord_tensor.shape[0]
        new_frames = int(original_frames / ratio)

        resampled = F.interpolate(chord_t, size=new_frames, mode="nearest")

        # (1, 3, new_T) -> (new_T, 3)
        resampled = resampled.squeeze(0).permute(1, 0).long()

        # Pad or Crop to match target_frames (original length)
        if new_frames < target_frames:
            # Pad with -1 (silence/unknown)
            pad_size = target_frames - new_frames
            pad_tensor = torch.full(
                (pad_size, 3), -1, dtype=resampled.dtype, device=resampled.device
            )
            resampled = torch.cat([resampled, pad_tensor], dim=0)
        elif new_frames > target_frames:
            # Crop
            resampled = resampled[:target_frames]

        return resampled

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        if batch.audio is None:
            # Audioがない場合はスキップ（実装簡略化のため）
            return batch

        new_audios = []
        new_chords: list[torch.Tensor] = []
        has_chord = batch.chord is not None

        for i in range(batch.batch_size):
            audio = batch.audio[i]
            original_length = audio.shape[-1]

            if random.random() > self.p:
                new_audios.append(audio)
                if has_chord:
                    assert batch.chord is not None
                    new_chords.append(batch.chord[i])
                continue

            semitones = random.randint(self.semitones_min, self.semitones_max)
            if semitones == 0:
                new_audios.append(audio)
                if has_chord:
                    assert batch.chord is not None
                    new_chords.append(batch.chord[i])
                continue

            # Calculate resampling ratio
            # ratio = 2^(n/12)
            # Pitch Up (n>0) -> ratio > 1 -> Faster speed -> Shorter duration
            ratio = 2 ** (semitones / 12.0)

            # Resample audio using F.interpolate (Linear Interpolation)
            # This is much faster and memory efficient than torchaudio.resample
            # audio: (C, T) -> (1, C, T) for interpolate
            audio_in = audio.unsqueeze(0)

            # Calculate new length
            # ratio > 1 (Pitch Up) -> Faster -> Shorter duration
            new_length = int(original_length / ratio)

            # Linear interpolation
            resampled_audio = F.interpolate(
                audio_in, size=new_length, mode="linear", align_corners=False
            ).squeeze(0)

            # Fix length
            current_length = resampled_audio.shape[-1]
            if current_length < original_length:
                # Pad
                pad_size = original_length - current_length
                resampled_audio = F.pad(resampled_audio, (0, pad_size))
            elif current_length > original_length:
                # Crop
                resampled_audio = resampled_audio[..., :original_length]

            new_audios.append(resampled_audio)

            if has_chord:
                assert batch.chord is not None
                # 1. Shift Root
                shifted_chord = self._shift_chord_tensor(batch.chord[i], semitones)
                # 2. Resample Time (Stretch/Shrink)
                target_frames = batch.chord.shape[1]
                resampled_chord = self._resample_chord_tensor(
                    shifted_chord, ratio, target_frames
                )
                new_chords.append(resampled_chord)

        batch.audio = torch.stack(new_audios)
        if has_chord and new_chords:
            batch.chord = torch.stack(new_chords)

        return batch


# =============================================================================
# nn.Moduleラッパー（GPU上での処理用）
# =============================================================================


class BatchTransformModule(nn.Module):
    """BatchTransformをnn.Moduleとしてラップ

    GPU上で効率的に処理するために使用。
    trainingモードに応じて拡張の適用を制御可能。
    """

    def __init__(
        self,
        train_transform: BatchTransform | None = None,
        eval_transform: BatchTransform | None = None,
    ):
        super().__init__()
        self.train_transform = train_transform or Identity()
        self.eval_transform = eval_transform or Identity()

    def forward(self, batch: AudioBatch) -> AudioBatch:
        if self.training:
            return self.train_transform(batch)
        else:
            return self.eval_transform(batch)


# =============================================================================
# ユーティリティ
# =============================================================================


def create_chord_training_transform(
    csv_path: str | None = None,
    drop_vocals: bool = True,
    pitch_shift_p: float = 0.0,
    gain_augment_p: float = 0.0,
    sample_rate: int = 44100,
) -> BatchTransform:
    """和音条件付け訓練用のTransformを作成

    Args:
        csv_path: 説明文CSVファイルのパス（Noneの場合はランダムプロンプト）
        drop_vocals: ボーカルを除去するか
        pitch_shift_p: ピッチシフトの適用確率
        gain_augment_p: ゲイン拡張の適用確率
        sample_rate: サンプリングレート

    Returns:
        構成済みのTransform
    """
    transforms: list[BatchTransform] = []

    # データ拡張（ステム操作）
    if gain_augment_p > 0:
        transforms.append(
            StemGainTransform(
                gain_min=0.7,
                gain_max=1.3,
                p=gain_augment_p,
            )
        )

    # ボーカル除去
    if drop_vocals:
        transforms.append(VocalsDropTransform())

    # ステムミックス（ここでAudioBatch.audioが生成される）
    # メモリ節約のため、ミックス後はステムを破棄する
    transforms.append(
        StemMixTransform(strategy="all", drop_vocals=False, keep_stems=False)
    )

    # データ拡張（ミックス後のAudioに対する操作）
    # ピッチシフトは重い処理なので、ミックス後に行うことで計算量を削減
    if pitch_shift_p > 0:
        transforms.append(
            VarispeedPitchShiftTransform(
                semitones_min=-5,
                semitones_max=6,
                sample_rate=sample_rate,
                p=pitch_shift_p,
            )
        )

    # プロンプト生成
    if csv_path:
        transforms.append(DescriptionPromptTransform(csv_path=csv_path))
    else:
        transforms.append(RandomPromptTransform())

    return Compose(transforms)


def create_chord_eval_transform(
    csv_path: str | None = None,
    drop_vocals: bool = True,
) -> BatchTransform:
    """和音条件付け評価用のTransformを作成（拡張なし）"""
    transforms: list[BatchTransform] = []

    if drop_vocals:
        transforms.append(VocalsDropTransform())

    transforms.append(
        StemMixTransform(strategy="all", drop_vocals=False, keep_stems=False)
    )

    if csv_path:
        transforms.append(DescriptionPromptTransform(csv_path=csv_path))
    else:
        transforms.append(RandomPromptTransform())

    return Compose(transforms)
