from typing import List, Tuple

import torch

from main.eval.chord_metrics import (
    CHORD_QUALITY_BINS,
    CHORD_ROOT_BINS,
    MAX_CHORD_INVERSION,
    NUM_CHORD_QUALITIES,
    NUM_CHORD_ROOTS,
    load_lab_annotations,
    number_to_quality,
    number_to_root,
    parse_chord_label,
    quality_to_number,
    root_to_number,
)

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
