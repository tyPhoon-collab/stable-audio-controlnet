"""
HDF5 キャッシュからメロディデータを読み込むデータセット実装

使用例:
  from dataset_musdb_melody_cached import create_musdb_dataset_melody_from_cache

  dataset = create_musdb_dataset_melody_from_cache(
      cache_file="/app/data/melody_cache/train_melody.h5",
      tar_path="/app/data/musdb18hq/train.tar",
      sample_rate=44100,
      chunk_dur=47.57,
  )
"""

import random
from functools import partial
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
import webdataset as wds
from torch import Tensor
from torch.utils.data import IterableDataset
from torchaudio.functional import resample
from webdataset.autodecode import torch_audio

from .common_mapping import DescriptionMapping, GenreMapping


def _fn_resample(sample, sample_rate: int):
    stems, sample_rate_orig, sample_key = sample
    return (
        {
            stem: resample(track, orig_freq=sample_rate_orig, new_freq=sample_rate)
            for stem, track in stems.items()
        },
        sample_rate,
        sample_key,
    )


def _weights_for_nonzero_refs(source_waveforms: Tensor):
    """Return shape (batch, source) weights for signals that are nonzero."""
    source_norms = torch.sqrt(torch.mean(source_waveforms**2, dim=-1))
    return torch.greater(source_norms, 1e-8)


def _fn_extract_stems_and_pad(sample):
    max_len = max(
        [
            v[0].shape[-1]
            for k, v in sample.items()
            if k.endswith(".mp3") or k.endswith(".wav")
        ]
    )
    default_sr = [
        v[1] for k, v in sample.items() if k.endswith(".mp3") or k.endswith(".wav")
    ][0]

    sample_key = sample.get("__key__", "unknown")

    stems = {
        k.split(".")[0]: F.pad(v[0], (0, max_len - v[0].shape[-1]))
        for k, v in sample.items()
        if (k.endswith(".mp3") or k.endswith(".wav"))
    }

    return stems, default_sr, sample_key


def _get_slices(
    src,
    chunk_dur: float,
):
    """Audio チャンクを生成（メロディ計算なし）"""
    for sample in src:
        stems, sr, sample_key = sample

        channels, length = list(stems.values())[0].shape
        chunk_size = int(sr * chunk_dur)

        if length < chunk_size:
            padding = torch.zeros(channels, chunk_size - length)
            stems = {
                stem: torch.cat([track, padding], dim=-1)
                for stem, track in stems.items()
            }
            length = chunk_size

        max_shift = length - (length // chunk_size) * chunk_size
        shift = torch.randint(0, max_shift + 1, (1,)).item()

        for i in range(length // chunk_size):
            start_idx = min(length - chunk_size, i * chunk_size + shift)
            end_idx = start_idx + chunk_size
            start_s = start_idx / sr

            chunks = {
                stem: track[:, start_idx:end_idx] for stem, track in stems.items()
            }

            chunks = {
                k: v
                for k, v in chunks.items()
                if _weights_for_nonzero_refs(v.sum(dim=0))
            }
            if len(chunks) < 2 or (
                len(chunks) == 2 and "vocals" in list(chunks.keys())
            ):
                continue

            yield chunks, start_s, length / sr, sample_key


class MelodyH5CachedDataset(IterableDataset):
    """
    HDF5 メロディキャッシュを使用するデータセット

    HDF5 ファイル構造（新形式）:
      /{sample_key}/melody -> (8, T_full) int64 (曲全体のメロディー)
      /{sample_key}/@duration_s -> float
      /{sample_key}/@sr -> int
    """

    def __init__(
        self,
        cache_file: str,
        tar_path: str,
        sample_rate: int,
        chunk_dur: float,
    ):
        self.cache_file = Path(cache_file)
        self.tar_path = tar_path
        self.sample_rate = sample_rate
        self.chunk_dur = chunk_dur
        self.shuffle_size = 0
        self.hop_length = -1  # Set in _load_melody_cache

        # HDF5 キャッシュをメモリに読み込み
        self._melody_cache = {}
        self._load_melody_cache()

    def shuffle(self, shuffle_size: int = 0):
        """WebDataset互換のshuffleメソッド"""
        self.shuffle_size = shuffle_size
        return self

    def _load_melody_cache(self):
        """HDF5 からメロディデータ（曲全体）をメモリに読み込む"""
        with h5py.File(self.cache_file, "r") as f:
            self.hop_length = int(f.attrs.get("hop_length", 512))
            for sample_key in f.keys():
                grp = f[sample_key]
                melody_data = grp["melody"]  # type: ignore
                if isinstance(melody_data, h5py.Dataset):
                    melody_np = melody_data[:]
                else:
                    melody_np = melody_data

                duration_s = float(grp.attrs["duration_s"])  # type: ignore
                sr = int(grp.attrs["sr"])  # type: ignore

                self._melody_cache[sample_key] = {
                    "melody": torch.from_numpy(melody_np).long(),  # (8, T_full)
                    "duration_s": duration_s,
                    "sr": sr,
                }
        print(f"✅ Loaded {len(self._melody_cache)} full melody tracks from cache")

    def __iter__(self):
        """WebDataset tar から audio チャンクを読み込み、キャッシュからメロディをスライス"""
        fill_missing_keys_and_pad = partial(_fn_extract_stems_and_pad)
        get_slices = partial(_get_slices, chunk_dur=self.chunk_dur)
        fn_resample = partial(_fn_resample, sample_rate=self.sample_rate)

        dataset = (
            wds.WebDataset(self.tar_path)
            .decode(torch_audio)
            .map(fill_missing_keys_and_pad)
            .map(fn_resample)
            .compose(get_slices)
        )

        for chunks, start_s, total_s, sample_key in dataset:
            if sample_key in self._melody_cache:
                cache_data = self._melody_cache[sample_key]
                melody_full = cache_data["melody"]  # (8, T_full)

                # フレーム位置を計算
                hop_length = self.hop_length
                chunk_size = int(self.sample_rate * self.chunk_dur)
                start_frame = int(start_s * self.sample_rate / hop_length)
                frames_per_chunk = int(chunk_size / hop_length)

                # メロディをスライス
                end_frame = min(start_frame + frames_per_chunk, melody_full.shape[1])
                melody_chunk = melody_full[:, start_frame:end_frame]

                # フレーム数が足りない場合はパディング
                if melody_chunk.shape[1] < frames_per_chunk:
                    padding = torch.full(
                        (8, frames_per_chunk - melody_chunk.shape[1]),
                        -1,
                        dtype=torch.long,
                    )
                    melody_chunk = torch.cat([melody_chunk, padding], dim=1)

                yield chunks, melody_chunk, start_s, total_s, sample_key
            else:
                print(f"⚠️  Melody cache miss for {sample_key}, skipping")


def create_musdb_dataset_melody_from_cache(
    cache_file: str,
    tar_path: str,
    sample_rate: int = 44100,
    chunk_dur: float = 47.57,
):
    """
    HDF5 キャッシュファイルとサンプル tar ファイルを使用してデータセットを作成

    Args:
        cache_file: HDF5 メロディキャッシュファイルパス
        tar_path: WebDataset tar ファイルパス
        sample_rate: サンプリングレート
        chunk_dur: チャンク長（秒）

    Returns:
        IterableDataset
    """
    return MelodyH5CachedDataset(
        cache_file=cache_file,
        tar_path=tar_path,
        sample_rate=sample_rate,
        chunk_dur=chunk_dur,
    )


def collate_fn_melody_conditional(
    samples,
    drop_vocals: bool = True,
    prompt_text: str | None = None,
):
    """Top-k CQT メロディ付きの conditional collate 関数"""
    start_seconds = [x for _, _, x, _, _ in samples]
    total_seconds = [x for _, _, _, x, _ in samples]
    melody_chunks = [x for _, x, _, _, _ in samples]
    samples = [x for x, _, _, _, _ in samples]

    if drop_vocals:
        for sample in samples:
            if "vocals" in sample:
                sample.pop("vocals")

    subsets_in = [
        random.sample(list(range(len(sample))), k=random.randint(1, len(sample) - 1))
        for sample in samples
    ]
    subsets_out = [
        random.sample(list(set(range(len(samples[i]))) - set(indices)), k=1)
        for i, indices in enumerate(subsets_in)
    ]

    outputs = []
    inputs = []
    prompts = []

    for i, sample in enumerate(samples):
        stem_keys = list(sample.keys())
        in_indices, out_indices = subsets_in[i], subsets_out[i]
        in_track = torch.stack([sample[stem_keys[i]] for i in in_indices]).sum(
            dim=0, keepdim=True
        )
        out_track = torch.stack([sample[stem_keys[i]] for i in out_indices]).sum(
            dim=0, keepdim=True
        )
        outputs.append(out_track)
        inputs.append(in_track)
        if prompt_text is None:
            in_stems_prompt = [stem_keys[j] for j in in_indices]
            out_stems_prompt = [stem_keys[j] for j in out_indices]
            prompts.append(
                f"in: {', '.join(in_stems_prompt)}; out: {', '.join(out_stems_prompt)}"
            )
        else:
            prompts.append(prompt_text)

    melody_batch = torch.stack(melody_chunks)
    return (
        torch.concat(outputs),
        torch.concat(inputs),
        prompts,
        start_seconds,
        total_seconds,
        melody_batch,
    )


def collate_fn_melody_mix(
    samples,
    drop_vocals: bool = True,
    prompt_text: str | None = None,
):
    """Top-k CQT メロディ付きの mix collate 関数"""
    start_seconds = [x for _, _, x, _, _ in samples]
    total_seconds = [x for _, _, _, x, _ in samples]
    melody_chunks = [x for _, x, _, _, _ in samples]
    samples = [x for x, _, _, _, _ in samples]

    if drop_vocals:
        for sample in samples:
            if "vocals" in sample:
                sample.pop("vocals")

    outputs = []
    prompts = []

    for i, sample in enumerate(samples):
        stem_keys = list(sample.keys())
        out_track = torch.stack(list(sample.values())).sum(dim=0, keepdim=True)
        outputs.append(out_track)
        if prompt_text is None:
            prompts.append(f"out: {', '.join(stem_keys)}")
        else:
            prompts.append(prompt_text)

    melody_batch = torch.stack(melody_chunks)
    return (
        torch.concat(outputs),
        prompts,
        start_seconds,
        total_seconds,
        melody_batch,
    )


def collate_fn_genre_melody(
    samples,
    csv_path: str,
    drop_vocals: bool = True,
    prompt_text: str | None = None,
):
    """ジャンル情報を使ったメロディ付き collate 関数

    Args:
        samples: バッチサンプル
        csv_path: ジャンル情報が含まれるCSVファイルのパス
        drop_vocals: ボーカルトラックを除去するか
        prompt_text: カスタムプロンプト（Noneの場合はジャンル情報を使用）
    """
    # ジャンルマッピングを読み込み
    genre_mapper = GenreMapping()
    genre_mapper.load_mapping(csv_path)

    # メロディ付きの形式
    start_seconds = [x for _, _, x, _, _ in samples]
    total_seconds = [x for _, _, _, x, _ in samples]
    melody_chunks = [x for _, x, _, _, _ in samples]
    sample_keys = [x for _, _, _, _, x in samples]
    samples_data = [x for x, _, _, _, _ in samples]

    if drop_vocals:
        for sample in samples_data:
            if "vocals" in sample:
                sample.pop("vocals")

    outputs = []
    prompts = []

    for i, sample in enumerate(samples_data):
        out_track = torch.stack(list(sample.values())).sum(dim=0, keepdim=True)
        outputs.append(out_track)

        if prompt_text is None:
            # sample_keyからジャンルを取得
            sample_key = sample_keys[i]
            genre = genre_mapper.get_genre(sample_key)
            prompts.append(f"{genre}")
        else:
            prompts.append(prompt_text)

    melody_batch = torch.stack(melody_chunks)
    return (
        torch.concat(outputs),
        prompts,
        start_seconds,
        total_seconds,
        melody_batch,
    )


def collate_fn_description_melody(
    samples,
    csv_path: str,
    drop_vocals: bool = True,
    prompt_text: str | None = None,
):
    """説明文をテキストプロンプトとして利用するメロディ付き collate 関数"""

    description_mapper = DescriptionMapping()
    description_mapper.load_mapping(csv_path)

    start_seconds = [x for _, _, x, _, _ in samples]
    total_seconds = [x for _, _, _, x, _ in samples]
    melody_chunks = [x for _, x, _, _, _ in samples]
    sample_keys = [x for _, _, _, _, x in samples]
    samples_data = [x for x, _, _, _, _ in samples]

    if drop_vocals:
        for sample in samples_data:
            if "vocals" in sample:
                sample.pop("vocals")

    outputs = []
    prompts = []

    for i, sample in enumerate(samples_data):
        out_track = torch.stack(list(sample.values())).sum(dim=0, keepdim=True)
        outputs.append(out_track)

        if prompt_text is None:
            sample_key = sample_keys[i]
            description = description_mapper.get_description(sample_key)
            prompts.append(description if description else "")
        else:
            prompts.append(prompt_text)

    melody_batch = torch.stack(melody_chunks)
    return (
        torch.concat(outputs),
        prompts,
        start_seconds,
        total_seconds,
        melody_batch,
    )
