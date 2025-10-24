"""
メロディデータの事前計算とHDF5キャッシング

使用例:
  python precompute_melody_cache.py --path /app/data/musdb18hq/train.tar \
                                     --cache-dir /app/data/melody_cache \
                                     --sample-rate 44100 \
                                     --chunk-dur 47.57 \
                                     --workers 4
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Dict

import h5py
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import webdataset as wds
from torch import Tensor
from torchaudio.functional import highpass_biquad, resample
from webdataset.autodecode import torch_audio


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
    source_norms = torch.sqrt(torch.mean(source_waveforms**2, dim=-1))
    return torch.greater(source_norms, 1e-8)


def _mix_stems(stems: Dict[str, Tensor]) -> Tensor:
    tracks = list(stems.values())
    mix = torch.stack(tracks).sum(dim=0)
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    if mix.size(0) == 1:
        mix = mix.repeat(2, 1)
    return mix


def _compute_topk_cqt_melody(
    waveform: Tensor,
    sample_rate: int,
    n_bins: int = 128,
    bins_per_octave: int = 12,
    topk: int = 4,
    hpf_cutoff_hz: float = 261.2,
    magnitude_threshold: float = 0.1,
    hop_length: int | None = None,
) -> Tensor:
    """Compute top-k CQT-based melody indices for stereo waveform."""
    x = highpass_biquad(waveform, sample_rate, cutoff_freq=hpf_cutoff_hz)

    if x.size(0) == 1:
        x = x.repeat(2, 1)

    fmin = librosa.midi_to_hz(0)
    if hop_length is None:
        hop_length = max(1, int(sample_rate / 100))

    mel_list = []
    for ch in [0, 1]:
        x_np = x[ch].detach().cpu().numpy().astype(np.float64)
        C = librosa.cqt(
            x_np,
            sr=sample_rate,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
        )
        mag = np.abs(C).astype(np.float32)
        mag_t = torch.from_numpy(mag)

        max_per_frame = torch.clamp(mag_t.max(dim=0).values, min=1e-8)
        norm_mag = mag_t / max_per_frame.unsqueeze(0)

        vals, idx = torch.topk(norm_mag, k=topk, dim=0, largest=True, sorted=True)

        idx_int = idx.to(torch.int64)
        rest_mask = vals < magnitude_threshold
        out = idx_int.clone()
        out[rest_mask] = -1
        mel_list.append(out)

        del x_np, C, mag, mag_t, max_per_frame, norm_mag, vals, idx, idx_int, rest_mask

    L, R = mel_list
    interleaved = torch.stack([L[0], R[0], L[1], R[1], L[2], R[2], L[3], R[3]], dim=0)
    return interleaved


def _get_slices_with_melody(
    src,
    chunk_dur: float,
    n_bins: int = 128,
    bins_per_octave: int = 12,
    topk: int = 4,
    hpf_cutoff_hz: float = 261.2,
    magnitude_threshold: float = 0.1,
    hop_length: int | None = None,
):
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

            mix = _mix_stems(chunks)
            melody_chunk = _compute_topk_cqt_melody(
                mix,
                sample_rate=sr,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                topk=topk,
                hpf_cutoff_hz=hpf_cutoff_hz,
                magnitude_threshold=magnitude_threshold,
                hop_length=hop_length,
            )

            yield chunks, melody_chunk, start_s, length / sr, sample_key


def precompute_melody_cache(
    tar_path: str,
    cache_dir: str,
    sample_rate: int = 44100,
    chunk_dur: float = 47.57,
    cqt_bins: int = 128,
    bins_per_octave: int = 12,
    topk: int = 4,
    hpf_cutoff_hz: float = 261.2,
    magnitude_threshold: float = 0.1,
):
    """
    WebDataset tar ファイルから全メロディデータを事前計算して HDF5 に保存

    HDF5 構造:
      /{sample_key}/melody -> (8, T_fk) int64 テンソル
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # tar ファイル名から HDF5 キャッシュファイル名を生成
    tar_name = Path(tar_path).stem
    cache_file = Path(cache_dir) / f"{tar_name}_melody.h5"

    print(f"📊 Creating melody cache: {cache_file}")
    print(f"   Source tar: {tar_path}")

    # Dataset の作成
    fill_missing_keys_and_pad = partial(_fn_extract_stems_and_pad)
    get_slices = partial(
        _get_slices_with_melody,
        chunk_dur=chunk_dur,
        n_bins=cqt_bins,
        bins_per_octave=bins_per_octave,
        topk=topk,
        hpf_cutoff_hz=hpf_cutoff_hz,
        magnitude_threshold=magnitude_threshold,
        hop_length=None,
    )
    fn_resample = partial(_fn_resample, sample_rate=sample_rate)

    dataset = (
        wds.WebDataset(tar_path)
        .decode(torch_audio)
        .map(fill_missing_keys_and_pad)
        .map(fn_resample)
        .compose(get_slices)
    )

    # HDF5 ファイルに保存
    with h5py.File(cache_file, "w") as f:
        f.attrs["tar_path"] = str(tar_path)
        f.attrs["sample_rate"] = sample_rate
        f.attrs["chunk_dur"] = chunk_dur
        f.attrs["cqt_bins"] = cqt_bins
        f.attrs["topk"] = topk

        chunk_counters = {}  # sample_key ごとのチャンク番号を追跡
        total_count = 0

        for chunks, melody_chunk, start_s, total_s, sample_key in tqdm.tqdm(
            dataset, desc="Computing melodies"
        ):
            # sample_key ごとに独立したチャンク番号をカウント
            if sample_key not in chunk_counters:
                chunk_counters[sample_key] = 0

            chunk_idx = chunk_counters[sample_key]
            chunk_key = f"{sample_key}_chunk_{chunk_idx}"
            chunk_counters[sample_key] += 1

            # NumPy にコンバート
            melody_np = melody_chunk.numpy().astype(np.int64)

            grp = f.create_group(chunk_key)
            grp.create_dataset("melody", data=melody_np)
            grp.attrs["start_s"] = float(start_s)
            grp.attrs["total_s"] = float(total_s)
            grp.attrs["sample_key"] = sample_key

            total_count += 1

    print(f"✅ Saved {total_count} melody samples to {cache_file}")
    return cache_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute melody cache from WebDataset tar"
    )
    parser.add_argument("--path", type=str, required=True, help="Path to tar file")
    parser.add_argument("--cache-dir", type=str, default="/app/data/melody_cache")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--chunk-dur", type=float, default=47.57)
    parser.add_argument("--cqt-bins", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--hpf-cutoff-hz", type=float, default=261.2)
    parser.add_argument("--magnitude-threshold", type=float, default=0.1)

    args = parser.parse_args()

    precompute_melody_cache(
        tar_path=args.path,
        cache_dir=args.cache_dir,
        sample_rate=args.sample_rate,
        chunk_dur=args.chunk_dur,
        cqt_bins=args.cqt_bins,
        topk=args.topk,
        hpf_cutoff_hz=args.hpf_cutoff_hz,
        magnitude_threshold=args.magnitude_threshold,
    )
