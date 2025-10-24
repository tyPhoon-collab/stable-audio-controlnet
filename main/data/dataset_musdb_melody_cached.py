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
from typing import Dict

import h5py
import torch
import torch.nn.functional as F
import webdataset as wds
from torch import Tensor
from torch.utils.data import IterableDataset
from torchaudio.functional import resample
from webdataset.autodecode import torch_audio


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


def _mix_stems(stems: Dict[str, Tensor]) -> Tensor:
    """Sum all stems to create a stereo mixture. Ensures 2 channels."""
    tracks = list(stems.values())
    mix = torch.stack(tracks).sum(dim=0)
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    if mix.size(0) == 1:
        mix = mix.repeat(2, 1)
    return mix


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

    HDF5 ファイル構造:
      /{chunk_key}/melody -> (8, T_fk) int64
      /{chunk_key}/@start_s -> float
      /{chunk_key}/@total_s -> float
      /{chunk_key}/@sample_key -> str
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

        # HDF5 キャッシュをメモリに読み込み（小～中規模な場合）
        self._melody_cache = {}
        self._load_melody_cache()

    def shuffle(self, shuffle_size: int = 0):
        """WebDataset互換のshuffleメソッド（shuffle_size は無視）"""
        self.shuffle_size = shuffle_size
        return self

    def _load_melody_cache(self):
        """HDF5 からメロディデータをメモリに読み込む"""
        with h5py.File(self.cache_file, "r") as f:
            for chunk_key in f.keys():
                grp = f[chunk_key]
                melody_data = grp["melody"]  # type: ignore
                if isinstance(melody_data, h5py.Dataset):
                    melody_np = melody_data[:]
                else:
                    melody_np = melody_data

                start_s_val = grp.attrs["start_s"]  # type: ignore
                total_s_val = grp.attrs["total_s"]  # type: ignore
                sample_key_val = grp.attrs["sample_key"]  # type: ignore

                self._melody_cache[chunk_key] = {
                    "melody": torch.from_numpy(melody_np).long(),
                    "start_s": float(start_s_val),  # type: ignore
                    "total_s": float(total_s_val),  # type: ignore
                    "sample_key": str(sample_key_val),
                }
        print(f"✅ Loaded {len(self._melody_cache)} melody samples from cache")

    def __iter__(self):
        """WebDataset tar から audio チャンクを読み込み、キャッシュからメロディを取得"""
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

        chunk_counters = {}  # sample_key ごとのチャンク番号を追跡
        for chunks, start_s, total_s, sample_key in dataset:
            # sample_key ごとに独立したチャンク番号をカウント
            if sample_key not in chunk_counters:
                chunk_counters[sample_key] = 0

            chunk_idx = chunk_counters[sample_key]
            chunk_key = f"{sample_key}_chunk_{chunk_idx}"
            chunk_counters[sample_key] += 1

            if chunk_key in self._melody_cache:
                cache_data = self._melody_cache[chunk_key]
                melody_chunk = cache_data["melody"]
                yield chunks, melody_chunk, start_s, total_s, sample_key
            else:
                print(f"⚠️  Melody cache miss for {chunk_key}, skipping")


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


if __name__ == "__main__":
    print("=== キャッシュ付きメロディデータセット テスト ===")
    # 注意: 先に precompute_melody_cache.py を実行してキャッシュを作成してください
    # dataset = create_musdb_dataset_melody_from_cache(
    #     cache_file="/app/data/melody_cache/train_melody.h5",
    #     tar_path="/app/data/musdb18hq/train.tar",
    #     sample_rate=44100,
    #     chunk_dur=47.57,
    # )
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     pin_memory=True,
    #     collate_fn=collate_fn_melody_mix,
    #     num_workers=0,
    # )
    # for i, batch in enumerate(dataloader):
    #     print(f"Batch {i}: {len(batch)} elements")
    #     outputs, prompts, start_seconds, total_seconds, melody_batch = batch
    #     print(f"  Audio: {outputs.shape}, Melody: {melody_batch.shape}")
    #     if i >= 1:
    #         break
    print("キャッシュ作成後、上記をコメントアウト解除して実行してください")
