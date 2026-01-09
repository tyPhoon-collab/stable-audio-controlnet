import os
import random
import glob
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import IterableDataset

from .annotation import ChordAnnotation
# dataset_musdb_chord のロジックを再利用
from .dataset_musdb_chord import _get_slices


class SlakhDataset(IterableDataset):
    """Slakh2100 dataset iterator.

    ディレクトリ構造のSlakhデータセットを読み込み、WebDataset互換のストリームとして機能します。
    """
    def __init__(
        self,
        data_root: str,
        lab_dir: str,
        sample_rate: int,
        chunk_dur: float,
        chord_frame_rate: float,
        shuffle_buffer: int = 100,
        epoch_length: int = 1000,
        shardshuffle: bool = True,
    ):
        self.data_root = Path(data_root)
        self.lab_dir = Path(lab_dir)
        self.sample_rate = sample_rate
        self.chunk_dur = chunk_dur
        self.chord_frame_rate = chord_frame_rate
        self.shuffle_buffer = shuffle_buffer
        self.epoch_length = epoch_length
        self.shardshuffle = shardshuffle

        # Recursively find Track folders (as they are inside train/validation/test/omitted)
        self.tracks = sorted(list(self.data_root.glob("**/Track*")))
        if not self.tracks:
            print(f"Warning: No tracks found in {self.data_root}")

    def shuffle(self, size):
        """WebDataset互換のshuffleメソッド（内部バッファサイズを設定）"""
        self.shuffle_buffer = size
        return self

    def _load_track(self, track_path: Path):
        """1トラックのロードと前処理"""
        track_name = track_path.name
        stems_dir = track_path / "stems"

        # Audio loading
        stems = {}
        # Support both .wav and .flac
        stem_files = list(stems_dir.glob("*.wav")) + list(stems_dir.glob("*.flac"))
        stem_files = [f for f in stem_files if not f.name.startswith("._")]
        if not stem_files:
            return None

        # Load all stems
        max_len = 0
        loaded_stems = {}

        try:
            for stem_file in stem_files:
                # S00.wav -> S00
                stem_name = stem_file.stem
                wav, sr = torchaudio.load(stem_file)

                # Resample if needed
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

                # Mono to Stereo
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)

                loaded_stems[stem_name] = wav
                max_len = max(max_len, wav.shape[-1])
        except Exception as e:
            print(f"Error loading stems for {track_name}: {e}")
            return None

        # Pad strings
        for k, v in loaded_stems.items():
            if v.shape[-1] < max_len:
                loaded_stems[k] = F.pad(v, (0, max_len - v.shape[-1]))

        # Load Chord Annotations
        lab_file = self.lab_dir / f"{track_name}.lab"

        chord_annotator = ChordAnnotation(sample_rate=self.sample_rate)
        if lab_file.exists():
            annotations = chord_annotator.load_lab_file(str(lab_file))
        else:
            # ラベルがない場合はすべてN (No chord) とする
            # あるいはスキップする？ ここではNとする
            annotations = []

        chord_tensor = chord_annotator.create_chord_tensor(
            annotations, max_len, frame_rate=self.chord_frame_rate
        )

        return loaded_stems, self.sample_rate, chord_tensor, track_name

    def __iter__(self) -> Iterator:
        """データセットのイテレータ"""
        # ワーカー情報に基づくシャーディング
        worker_info = torch.utils.data.get_worker_info()
        tracks = list(self.tracks)

        if self.shardshuffle:
            random.shuffle(tracks)

        if worker_info is not None:
            # ワーカー間で分割
            per_worker = int(len(tracks) / worker_info.num_workers)
            iter_start = worker_info.id * per_worker
            iter_end = iter_start + per_worker
            tracks = tracks[iter_start:iter_end]

        # 無限ループまたはエポック長制限が必要かもしれないが、
        # ここではトラックを一巡したら終わるジェネレータを定義し、
        # 必要ならDataLoaderのpersistent_workersなどで制御される

        buffer = []

        for track_path in tracks:
            sample = self._load_track(track_path)
            if sample is None:
                continue

            # スライス生成
            # _get_slices expects iterable of samples
            # sample format: (stems, sr, chord_tensor, sample_key)
            slices_gen = _get_slices([sample], self.chunk_dur, self.chord_frame_rate)

            for slice_data in slices_gen:
                if self.shuffle_buffer > 0:
                    buffer.append(slice_data)
                    if len(buffer) >= self.shuffle_buffer:
                        idx = random.randint(0, len(buffer) - 1)
                        yield buffer.pop(idx)
                else:
                    yield slice_data

        # 残りのバッファを吐き出し
        if self.shuffle_buffer > 0:
            random.shuffle(buffer)
            for item in buffer:
                yield item


def create_slakh_dataset(
    data_root: str,
    lab_dir: str,
    sample_rate: int,
    chunk_dur: float,
    chord_frame_rate: float,
    shardshuffle: bool = False,
    **kwargs
):
    """SlakhDatasetのファクトリ関数"""
    return SlakhDataset(
        data_root=data_root,
        lab_dir=lab_dir,
        sample_rate=sample_rate,
        chunk_dur=chunk_dur,
        chord_frame_rate=chord_frame_rate,
        shardshuffle=shardshuffle,
    )
