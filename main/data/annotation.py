from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio


class ChordAnnotation:
    """和音アノテーションを処理するクラス"""

    # 基本的なコード名から数値へのマッピング
    CHORD_ROOT_MAP = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
        "N": -1,  # N は無音/不明
    }

    CHORD_QUALITY_MAP = {
        "maj": 0,
        "min": 1,
        "maj7": 2,
        "min7": 3,
        "dom7": 4,
        "7": 4,
        "dim": 5,
        "aug": 6,
        "sus4": 7,
        "sus2": 8,
        "N": -1,
    }

    # 逆変換用の正規化名（出力表記のための代表名）
    ROOT_NAMES = [
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

    QUALITY_NAMES = [
        "maj",
        "min",
        "maj7",
        "min7",
        "7",
        "dim",
        "aug",
        "sus4",
        "sus2",
    ]

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

        # root:quality の形式を分割
        if ":" in chord_symbol:
            root_str, quality_str = chord_symbol.split(":")
        else:
            root_str = chord_symbol
            quality_str = "maj"  # デフォルトはメジャー

        # ルート音の取得
        root = self.CHORD_ROOT_MAP.get(root_str, -1)

        # 和音の質の取得
        quality = self.CHORD_QUALITY_MAP.get(quality_str, 0)

        return (root, quality, inversion)

    def idx_to_chord_symbol(self, root: int, quality: int, inversion: int) -> str:
        """数値表現 (root, quality, inversion) を 'C:maj/0' 形式に変換"""
        if root < 0 or quality < 0:
            return "N"
        root_name = self.ROOT_NAMES[root % 12]
        qual_name = (
            self.QUALITY_NAMES[quality]
            if 0 <= quality < len(self.QUALITY_NAMES)
            else "maj"
        )
        symbol = f"{root_name}:{qual_name}"
        if inversion and inversion > 0:
            symbol += f"/{inversion}"
        return symbol

    def chord_timeline_text(
        self,
        chord_tensor: torch.Tensor,
        start_s: float = 0.0,
        total_s: Optional[float] = None,
        frame_rate: Optional[float] = None,
    ) -> str:
        """
        和音テンソル (T_frames, 3) を可読なタイムライン文字列に整形。

        引数:
            chord_tensor: (T, 3) [root, quality, inversion]
            start_s: 区間の開始秒
            total_s: 区間の総秒（frame_rate 未指定時に推定に使用）
            frame_rate: 明示のフレームレート（優先して使用）

        返り値:
            譜面風のタイムライン文字列
        """
        if chord_tensor.ndim != 2 or chord_tensor.shape[1] != 3:
            return "(no chord data)"

        T = chord_tensor.shape[0]
        if frame_rate is None:
            if total_s is not None and total_s > 0:
                frame_rate_eff = T / float(total_s)
            else:
                frame_rate_eff = 1.0
        else:
            frame_rate_eff = float(frame_rate)

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
            s_time = start_s + seg_start / frame_rate_eff
            e_time = start_s + t / frame_rate_eff
            lines.append(f"{s_time:8.3f} - {e_time:8.3f} : {sym}")

        header = [
            f"chord_frame_rate_estimate: {frame_rate_eff:.3f} Hz",
            f"frames: {T}",
        ]
        return "\n".join(header + ["chords:"] + lines)

    def load_lab_file(self, lab_file_path: str) -> List[Tuple[float, float, str]]:
        """
        .labファイルを読み込んで時間区間と和音のリストを返す

        Args:
            lab_file_path: .labファイルのパス

        Returns:
            [(start_time, end_time, chord_symbol), ...] のリスト
        """
        annotations = []
        with open(lab_file_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    chord_symbol = parts[2]
                    annotations.append((start_time, end_time, chord_symbol))
        return annotations

    def create_chord_tensor(
        self,
        annotations: List[Tuple[float, float, str]],
        audio_length: int,
        frame_rate: float = 100.0,
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

    def create_one_hot_chord_tensor(
        self,
        annotations: List[Tuple[float, float, str]],
        audio_length: int,
        frame_rate: float = 100.0,
    ) -> torch.Tensor:
        """
        和音アノテーションからワンホットエンコードされた時系列テンソルを作成

        Args:
            annotations: [(start_time, end_time, chord_symbol), ...] のリスト
            audio_length: 音響信号の長さ（サンプル数）
            frame_rate: 1秒あたりのフレーム数

        Returns:
            shape: (num_frames, 12 + 9 + 8) の tensor [root_onehot, quality_onehot, inversion_onehot]
        """
        duration_seconds = audio_length / self.sample_rate
        num_frames = int(duration_seconds * frame_rate)

        # root: 12音 + N, quality: 9種類 + N, inversion: 最大7 + unknown
        chord_tensor = torch.zeros((num_frames, 12 + 9 + 8), dtype=torch.float32)

        for start_time, end_time, chord_symbol in annotations:
            start_frame = int(start_time * frame_rate)
            end_frame = int(end_time * frame_rate)

            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)

            if start_frame < end_frame:
                root, quality, inversion = self.parse_chord_symbol(chord_symbol)

                # ワンホットエンコーディング
                frame_vector = torch.zeros(12 + 9 + 8)

                # Root (12音 + N)
                if root >= 0:
                    frame_vector[root] = 1.0
                else:
                    frame_vector[12] = 1.0  # N (無音)

                # Quality
                if quality >= 0:
                    frame_vector[13 + quality] = 1.0
                else:
                    frame_vector[13 + 8] = 1.0  # N (無音)

                # Inversion
                if inversion <= 6:
                    frame_vector[22 + inversion] = 1.0
                else:
                    frame_vector[22 + 7] = 1.0  # unknown inversion

                chord_tensor[start_frame:end_frame] = frame_vector

        return chord_tensor


class AudioChordDataset:
    """音響信号と和音テンソルを同時に取得するデータセットクラス"""

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44100,
        frame_rate: float = 25.0,  # 制御信号用に最適化（100Hz → 25Hz）
        control_optimized: bool = True,  # 制御信号最適化フラグ
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.control_optimized = control_optimized
        self.chord_annotation = ChordAnnotation(sample_rate)

        if control_optimized:
            print(f"制御信号最適化モード: {frame_rate}Hz")

    def get_control_signal_info(self, audio_shape: torch.Size) -> dict:
        """制御信号の情報を返す"""
        audio_samples = audio_shape[-1]
        audio_duration = audio_samples / self.sample_rate
        control_frames = int(audio_duration * self.frame_rate)

        audio_memory = audio_samples * 2 * 4 / (1024 * 1024)  # ステレオ float32
        control_memory = control_frames * 3 * 8 / (1024 * 1024)  # 3要素 int64

        return {
            "audio_duration": audio_duration,
            "control_frames": control_frames,
            "audio_memory_mb": audio_memory,
            "control_memory_mb": control_memory,
            "memory_ratio": control_memory / audio_memory * 100,
        }

    def create_mix_from_stems(
        self, base_name: str, stem_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        ステム音源からミックス音響信号を作成

        Args:
            base_name: ベースファイル名（例: "A Classic Education - NightOwl"）
            stem_types: 使用するステム（例: ["bass", "drums", "vocals", "other"]）

        Returns:
            ミックスされた音響信号
        """
        if stem_types is None:
            stem_types = ["bass", "drums", "vocals", "other"]

        mixed_audio = None

        for stem_type in stem_types:
            stem_file = self.data_dir / f"{base_name}.{stem_type}.mp3"
            if stem_file.exists():
                try:
                    waveform, orig_sr = torchaudio.load(str(stem_file))

                    # サンプリングレートの調整
                    if orig_sr != self.sample_rate:
                        waveform = torchaudio.functional.resample(
                            waveform, orig_sr, self.sample_rate
                        )

                    if mixed_audio is None:
                        mixed_audio = waveform
                    else:
                        # 長さを合わせる（短い方に合わせる）
                        min_length = min(mixed_audio.shape[-1], waveform.shape[-1])
                        mixed_audio = (
                            mixed_audio[..., :min_length] + waveform[..., :min_length]
                        )

                except Exception as e:
                    print(f"Warning: Could not load {stem_file}: {e}")

        return mixed_audio

    def load_mix_and_chords(
        self, base_name: str, stem_types: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ステムからミックス音響信号を作成し、和音テンソルと組み合わせて返す

        Args:
            base_name: ベースファイル名
            stem_types: 使用するステム

        Returns:
            (mixed_audio_tensor, chord_tensor) のタプル
        """
        # ミックス音響信号の作成
        mixed_audio = self.create_mix_from_stems(base_name, stem_types)

        if mixed_audio is None:
            raise ValueError(f"No audio stems found for {base_name}")

        # 和音アノテーションの読み込み
        lab_file = self.data_dir / f"{base_name}.lab"
        if not lab_file.exists():
            raise FileNotFoundError(f"Lab file not found: {lab_file}")

        annotations = self.chord_annotation.load_lab_file(str(lab_file))

        # 和音テンソルの作成
        chord_tensor = self.chord_annotation.create_chord_tensor(
            annotations, mixed_audio.shape[-1], self.frame_rate
        )

        return mixed_audio, chord_tensor

    def load_audio_and_chords(
        self, audio_file: str, lab_file: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        音響ファイルと和音ファイルを読み込み、対応するテンソルを返す

        Args:
            audio_file: 音響ファイルのパス
            lab_file: .labファイルのパス

        Returns:
            (audio_tensor, chord_tensor) のタプル
        """
        # 音響信号の読み込み
        waveform, orig_sr = torchaudio.load(audio_file)

        # サンプリングレートの調整
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sr, self.sample_rate
            )

        # 和音アノテーションの読み込み
        annotations = self.chord_annotation.load_lab_file(lab_file)

        # 和音テンソルの作成
        chord_tensor = self.chord_annotation.create_chord_tensor(
            annotations, waveform.shape[-1], self.frame_rate
        )

        return waveform, chord_tensor

    def load_audio_and_chords_onehot(
        self, audio_file: str, lab_file: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        音響ファイルと和音ファイルを読み込み、ワンホットエンコードされた和音テンソルを返す
        """
        # 音響信号の読み込み
        waveform, orig_sr = torchaudio.load(audio_file)

        # サンプリングレートの調整
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sr, self.sample_rate
            )

        # 和音アノテーションの読み込み
        annotations = self.chord_annotation.load_lab_file(lab_file)

        # ワンホット和音テンソルの作成
        chord_tensor = self.chord_annotation.create_one_hot_chord_tensor(
            annotations, waveform.shape[-1], self.frame_rate
        )

        return waveform, chord_tensor

    def process_directory(
        self, use_onehot: bool = False
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        ディレクトリ内のすべての音響ファイルと対応する.labファイルを処理

        Args:
            use_onehot: ワンホットエンコードを使用するかどうか

        Returns:
            {filename: (audio_tensor, chord_tensor)} の辞書
        """
        results = {}

        # .labファイルを探す
        lab_files = list(self.data_dir.glob("*.lab"))

        for lab_file in lab_files:
            # 対応する音響ファイルを探す
            base_name = lab_file.stem

            # 可能な音響ファイル拡張子
            audio_extensions = [".mp3", ".wav", ".flac"]

            for ext in audio_extensions:
                audio_file = self.data_dir / f"{base_name}{ext}"
                if audio_file.exists():
                    try:
                        if use_onehot:
                            audio, chords = self.load_audio_and_chords_onehot(
                                str(audio_file), str(lab_file)
                            )
                        else:
                            audio, chords = self.load_audio_and_chords(
                                str(audio_file), str(lab_file)
                            )

                        results[base_name] = (audio, chords)
                        print(f"Processed: {base_name}")
                        print(f"  Audio shape: {audio.shape}")
                        print(f"  Chord tensor shape: {chords.shape}")
                        break

                    except Exception as e:
                        print(f"Error processing {base_name}: {e}")

        return results


def main():
    """使用例"""
    # データディレクトリのパス（コンテナ内の絶対パス）
    data_dir = "/app/data"

    # 制御信号最適化モードでデータセットの初期化
    dataset = AudioChordDataset(
        data_dir, sample_rate=44100, frame_rate=25.0, control_optimized=True
    )

    # 単一ファイルのテスト（ベース音声を使用）
    base_name = "A Classic Education - NightOwl"
    audio_file = f"{data_dir}/{base_name}.bass.mp3"
    lab_file = f"{data_dir}/{base_name}.lab"

    print(f"=== 制御信号最適化テスト: {base_name} ===")
    try:
        # 基本的な和音テンソル
        audio, chords = dataset.load_audio_and_chords(audio_file, lab_file)

        # 制御信号情報の表示
        info = dataset.get_control_signal_info(audio.shape)
        print(f"音響信号: {audio.shape} ({info['audio_memory_mb']:.1f}MB)")
        print(f"制御信号: {chords.shape} ({info['control_memory_mb']:.3f}MB)")
        print(f"メモリ効率: 制御信号は音響信号の {info['memory_ratio']:.3f}%")
        print(
            f"時間長: 音響{info['audio_duration']:.2f}秒, 制御{info['control_frames'] / dataset.frame_rate:.2f}秒"
        )

        # 最初の10フレームの和音情報を表示
        print("\n最初の10フレームの和音情報:")
        for i in range(min(10, chords.shape[0])):
            root, quality, inversion = chords[i].tolist()
            if root >= 0:
                # ルート音を音名に変換
                root_names = [
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
                root_name = root_names[root] if root < 12 else "N"
                quality_names = [
                    "maj",
                    "min",
                    "maj7",
                    "min7",
                    "dom7",
                    "dim",
                    "aug",
                    "sus4",
                    "sus2",
                ]
                quality_name = (
                    quality_names[quality]
                    if 0 <= quality < len(quality_names)
                    else "unknown"
                )
                print(f"  Frame {i}: {root_name}:{quality_name} (転回: {inversion})")
            else:
                print(f"  Frame {i}: N (無音)")

        # ワンホットエンコード版もテスト
        audio_onehot, chords_onehot = dataset.load_audio_and_chords_onehot(
            audio_file, lab_file
        )
        onehot_memory = (
            chords_onehot.numel() * chords_onehot.element_size() / (1024 * 1024)
        )
        print("\nワンホットエンコード版:")
        print(f"和音テンソルの形状: {chords_onehot.shape}")
        print(f"メモリ使用量: {onehot_memory:.3f}MB")
        print(f"データ型: {chords_onehot.dtype}")

        # 和音の変化点を探す
        print("\n和音の変化点 (最初の10個):")
        prev_chord = None
        change_count = 0
        for i in range(chords.shape[0]):
            current_chord = tuple(chords[i].tolist())
            if current_chord != prev_chord:
                if change_count < 10:
                    time_sec = i / dataset.frame_rate
                    root, quality, inversion = current_chord
                    if root >= 0:
                        root_names = [
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
                        root_name = root_names[root] if root < 12 else "N"
                        quality_names = [
                            "maj",
                            "min",
                            "maj7",
                            "min7",
                            "dom7",
                            "dim",
                            "aug",
                            "sus4",
                            "sus2",
                        ]
                        quality_name = (
                            quality_names[quality]
                            if 0 <= quality < len(quality_names)
                            else "unknown"
                        )
                        print(f"  {time_sec:.2f}秒: {root_name}:{quality_name}")
                    else:
                        print(f"  {time_sec:.2f}秒: N (無音)")
                    change_count += 1
                prev_chord = current_chord

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()

    # ミックス音響信号のテスト
    print("\n=== ControlNet用ミックス信号テスト ===")
    try:
        # 全ステムからミックスを作成
        mixed_audio, chords = dataset.load_mix_and_chords(base_name)
        info = dataset.get_control_signal_info(mixed_audio.shape)

        print(f"ミックス音響信号: {mixed_audio.shape}")
        print(f"制御和音信号: {chords.shape}")
        print("ControlNetでの使用:")
        print(f"  - 音響入力: {info['audio_memory_mb']:.1f}MB")
        print(
            f"  - 和音制御: {info['control_memory_mb']:.3f}MB ({info['memory_ratio']:.3f}%)"
        )

        # 特定のステムのみからミックスを作成
        selected_stems = ["bass", "drums"]
        mixed_audio_selected, chords_selected = dataset.load_mix_and_chords(
            base_name, selected_stems
        )
        print(f"\n選択ステム（{selected_stems}）:")
        print(f"  ミックス: {mixed_audio_selected.shape}")
        print(f"  制御: {chords_selected.shape}")

    except Exception as e:
        print(f"ミックス処理エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
