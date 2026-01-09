from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch

CHROMATIC_SCALE = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

_PREFERRED_ROOT_NAMES = {
    10: "Bb",
    25: "Bb",
}

QUALITY_NAMES = [
    "maj",
    "min",
    "dim",
    "aug",
    "min6",
    "maj6",
    "min7",
    "minmaj7",
    "maj7",
    "7",
    "dim7",
    "hdim7",
    "sus2",
    "sus4",
]

QUALITY_CHROMA_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "min6": (0, 3, 7, 9),
    "maj6": (0, 4, 7, 9),
    "min7": (0, 3, 7, 10),
    "minmaj7": (0, 3, 7, 11),
    "maj7": (0, 4, 7, 11),
    "7": (0, 4, 7, 10),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "N": (),
}

_FLAT_TO_SHARP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}

_QUALITY_ALIAS_MAP = {
    "major": "maj",
    "maj": "maj",
    "minor": "min",
    "m": "min",
    "dom": "7",
    "dom7": "7",
    "dominant": "7",
    "diminished": "dim",
    "dimin": "dim",
    "dim": "dim",
    "diminished7": "dim7",
    "dim7": "dim7",
    "hdim": "hdim7",
    "hdim7": "hdim7",
    "half-diminished": "hdim7",
    "half-diminished7": "hdim7",
    "half diminished": "hdim7",
    "m7b5": "hdim7",
    "augmented": "aug",
    "minor6": "min6",
    "m6": "min6",
    "major6": "maj6",
    "maj6": "maj6",
    "minor7": "min7",
    "m7": "min7",
    "maj7": "maj7",
    "major7": "maj7",
    "minmaj7": "minmaj7",
    "mmaj7": "minmaj7",
    "sus": "sus4",
    "suspended": "sus4",
    "sus4": "sus4",
    "sus2": "sus2",
    "n": "N",
}

NUM_CHORD_ROOTS = len(CHROMATIC_SCALE)
CHORD_ROOT_BINS = NUM_CHORD_ROOTS + 1  # +1 for 'N'
NUM_CHORD_QUALITIES = len(QUALITY_NAMES)
CHORD_QUALITY_BINS = NUM_CHORD_QUALITIES + 1  # +1 for 'N'
MAX_CHORD_INVERSION = 6
CHORD_INVERSION_BINS = MAX_CHORD_INVERSION + 2  # 0-6 + unknown slot
CHORD_ONEHOT_DIM = CHORD_ROOT_BINS + CHORD_QUALITY_BINS + CHORD_INVERSION_BINS

LabAnnotation = Tuple[float, float, str]


def parse_lab_content(content: str) -> List[LabAnnotation]:
    """Lab形式の文字列からアノテーションをパースする"""
    annotations: List[LabAnnotation] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        label = " ".join(parts[2:]).strip()
        if not label:
            label = "N"
        annotations.append((start, end, label))
    return annotations


def load_lab_annotations(path: Path | str) -> List[LabAnnotation]:
    """Labファイルから和音アノテーションを読み込む"""
    lab_path = Path(path)
    if not lab_path.exists():
        raise FileNotFoundError(f"Lab file not found: {lab_path}")

    try:
        content = lab_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        with lab_path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    return parse_lab_content(content)


def normalize_root(root: str) -> str:
    """フラット表記などを正規化して基音表記を統一"""
    if not root:
        return "N"
    stripped = root.strip()
    if not stripped:
        return "N"
    if stripped.upper() == "N":
        return "N"
    normalized = stripped[0].upper() + stripped[1:]
    for flat, sharp in _FLAT_TO_SHARP.items():
        normalized = normalized.replace(flat, sharp)
        normalized = normalized.replace(flat.lower(), sharp)
    return normalized


def normalize_quality(quality: str) -> str:
    """和音品質の表記ゆれを正規化"""
    if not quality:
        return "N"
    stripped = quality.strip()
    if not stripped:
        return "N"
    lowered = stripped.lower()
    if lowered in _QUALITY_ALIAS_MAP:
        return _QUALITY_ALIAS_MAP[lowered]
    for canonical in QUALITY_NAMES:
        if lowered == canonical.lower():
            return canonical
    if lowered == "n":
        return "N"
    return stripped


def parse_chord_label(chord_label: str) -> Tuple[str, str]:
    """'C:maj7' を (root, quality) に分解"""
    if not chord_label or chord_label == "N":
        return "N", "N"
    if ":" in chord_label:
        root, quality = chord_label.split(":", 1)
    else:
        root = chord_label
        quality = "maj"
    root = normalize_root(root)
    quality = normalize_quality(quality)
    return root, quality


def root_to_number(root: str) -> int:
    """基音を0-11の数値に変換"""
    if not root or root == "N":
        return -1
    normalized = normalize_root(root)
    try:
        return CHROMATIC_SCALE.index(normalized)
    except ValueError:
        return 0


def number_to_root(index: int) -> str:
    """インデックスを基音ラベルへ変換"""
    if index < 0:
        return "N"
    normalized = index % NUM_CHORD_ROOTS
    if normalized in _PREFERRED_ROOT_NAMES:
        return _PREFERRED_ROOT_NAMES[normalized]
    return CHROMATIC_SCALE[normalized]


def quality_to_number(quality: str) -> int:
    """和音品質をインデックスに変換"""
    normalized = normalize_quality(quality)
    if normalized == "N":
        return -1
    try:
        return QUALITY_NAMES.index(normalized)
    except ValueError:
        return 0


def number_to_quality(index: int) -> str:
    """品質インデックスをラベルへ変換"""
    if index < 0:
        return "N"
    if index < NUM_CHORD_QUALITIES:
        return QUALITY_NAMES[index]
    return QUALITY_NAMES[0]


def quality_chroma_offsets(quality: str) -> List[int]:
    """和音品質から相対クロマ（半音単位）のリストを取得"""
    normalized = normalize_quality(quality)
    return list(QUALITY_CHROMA_INTERVALS.get(normalized, ()))


ROOT_N_INDEX = NUM_CHORD_ROOTS
QUALITY_N_INDEX = NUM_CHORD_QUALITIES
INVERSION_UNKNOWN_INDEX = MAX_CHORD_INVERSION + 1
ROOT_OFFSET = 0
QUALITY_OFFSET = CHORD_ROOT_BINS
INVERSION_OFFSET = QUALITY_OFFSET + CHORD_QUALITY_BINS


class ChordAnnotation:
    """和音アノテーションを処理するクラス"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def parse_chord_symbol(self, chord_symbol: str) -> Tuple[int, int, int]:
        """
        和音記号を解析して (root, quality, inversion) のタプルを返す

        Args:
            chord_symbol: 'F:maj', 'Bb:maj7', 'F:maj/3' のような和音記号

        Returns:
            (root, quality, inversion) のタプル
        """
        if chord_symbol == "N":
            return (-1, -1, 0)

        # 転回形の処理
        inversion = 0
        if "/" in chord_symbol:
            chord_symbol, inversion_str = chord_symbol.split("/")
            if inversion_str.isdigit():
                inversion = int(inversion_str)

        root_str, quality_str = parse_chord_label(chord_symbol)
        root = root_to_number(root_str)
        quality = quality_to_number(quality_str)

        return (root, quality, inversion)

    def idx_to_chord_symbol(self, root: int, quality: int, inversion: int) -> str:
        """数値表現 (root, quality, inversion) を 'C:maj/0' 形式に変換"""
        if root < 0 or quality < 0:
            return "N"
        root_name = number_to_root(root)
        qual_name = number_to_quality(quality)
        symbol = f"{root_name}:{qual_name}"
        if inversion and inversion > 0:
            symbol += f"/{inversion}"
        return symbol

    def chord_timeline_text(
        self,
        chord_tensor: torch.Tensor,
        frame_rate: float,
    ) -> str:
        """
        和音テンソル (T_frames, 3) を可読なタイムライン文字列に整形。

        引数:
            chord_tensor: (T, 3) [root, quality, inversion]
            frame_rate: フレームレート

        返り値:
            譜面風のタイムライン文字列
        """
        if chord_tensor.ndim != 2 or chord_tensor.shape[1] != 3:
            return "(no chord data)"

        T = chord_tensor.shape[0]

        lines = []
        t = 0
        while t < T:
            r, q, v = chord_tensor[t].tolist()
            sym = self.idx_to_chord_symbol(int(r), int(q), int(v))
            seg_start = t
            t += 1
            while t < T:
                r2, q2, v2 = chord_tensor[t].tolist()
                if self.idx_to_chord_symbol(int(r2), int(q2), int(v2)) != sym:
                    break
                t += 1
            s_time = seg_start / frame_rate
            e_time = t / frame_rate
            lines.append(f"{s_time:8.3f} - {e_time:8.3f} : {sym}")

        header = [
            f"frame_rate: {frame_rate:.3f} Hz",
            f"frames: {T}",
        ]
        return "\n".join(header + ["chords:"] + lines)

    def chord_tensor_to_lab_format(
        self,
        chord_tensor: torch.Tensor,
        frame_rate: float,
    ) -> str:
        """
        和音テンソルを完全な.lab形式（TSV形式）に変換

        Args:
            chord_tensor: (T, 3) shape の tensor [root, quality, inversion]
            frame_rate: フレームレート（Hz）

        Returns:
            .lab形式の文字列（各行が「start_time\tend_time\tchord_symbol」）
        """
        lab_lines = []

        if chord_tensor.ndim != 2 or chord_tensor.shape[1] != 3:
            return ""

        T = chord_tensor.shape[0]

        t = 0
        while t < T:
            r, q, v = chord_tensor[t].tolist()
            sym = self.idx_to_chord_symbol(int(r), int(q), int(v))
            seg_start = t
            t += 1
            while t < T:
                r2, q2, v2 = chord_tensor[t].tolist()
                if self.idx_to_chord_symbol(int(r2), int(q2), int(v2)) != sym:
                    break
                t += 1
            s_time = seg_start / frame_rate
            e_time = t / frame_rate
            # .lab形式: start_time\tend_time\tchord_symbol
            lab_lines.append(f"{s_time:.3f}\t{e_time:.3f}\t{sym}")

        return "\n".join(lab_lines)

    def load_lab_file(self, lab_file_path: str) -> List[Tuple[float, float, str]]:
        """
        .labファイルを読み込んで時間区間と和音のリストを返す

        Args:
            lab_file_path: .labファイルのパス

        Returns:
            [(start_time, end_time, chord_symbol), ...] のリスト
        """
        return load_lab_annotations(lab_file_path)

    def load_lab_content(self, content: str) -> List[Tuple[float, float, str]]:
        """
        .lab形式の文字列を読み込んで時間区間と和音のリストを返す

        Args:
            content: .labファイルの内容

        Returns:
            [(start_time, end_time, chord_symbol), ...] のリスト
        """
        return parse_lab_content(content)

    def create_chord_tensor(
        self,
        annotations: List[Tuple[float, float, str]],
        audio_length: int,
        frame_rate: float,
    ) -> torch.Tensor:
        """
        和音アノテーションから時系列テンソルを作成

        Args:
            annotations: [(start_time, end_time, chord_symbol), ...] のリスト
            audio_length: 音響信号の長さ（サンプル数）
            frame_rate: 1秒あたりのフレーム数

        Returns:
            shape: (num_frames, 3) の tensor [root, quality, inversion]
        """
        duration_seconds = audio_length / self.sample_rate
        num_frames = int(duration_seconds * frame_rate)

        # 初期化（すべて無音として）
        chord_tensor = torch.full((num_frames, 3), -1, dtype=torch.long)

        for start_time, end_time, chord_symbol in annotations:
            start_frame = int(start_time * frame_rate)
            end_frame = int(end_time * frame_rate)

            # フレーム範囲のクリッピング
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)

            if start_frame < end_frame:
                root, quality, inversion = self.parse_chord_symbol(chord_symbol)
                chord_tensor[start_frame:end_frame] = torch.tensor(
                    [root, quality, inversion]
                )

        return chord_tensor
