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
    chunk_duration_s: float | None = 60.0,
) -> Tensor:
    """Compute top-k CQT-based melody indices for stereo waveform."""
    x = highpass_biquad(waveform, sample_rate, cutoff_freq=hpf_cutoff_hz)

    if x.size(0) == 1:
        x = x.repeat(2, 1)

    fmin = librosa.midi_to_hz(0)
    if hop_length is None:
        hop_length = max(1, int(sample_rate / 100))

    if chunk_duration_s is None:
        chunk_samples = x.shape[-1]
    else:
        chunk_samples = max(hop_length, int(chunk_duration_s * sample_rate))

    mel_list = []
    for ch in [0, 1]:
        channel = x[ch]
        channel_chunks = []

        for start in range(0, channel.shape[-1], chunk_samples):
            end = min(start + chunk_samples, channel.shape[-1])
            chunk = channel[start:end]
            if chunk.numel() == 0:
                continue

            x_np = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
            try:
                C = librosa.cqt(
                    x_np,
                    sr=sample_rate,
                    fmin=fmin,
                    n_bins=n_bins,
                    bins_per_octave=bins_per_octave,
                    hop_length=hop_length,
                    dtype=np.complex64,
                )
            except TypeError:
                C = librosa.cqt(
                    x_np,
                    sr=sample_rate,
                    fmin=fmin,
                    n_bins=n_bins,
                    bins_per_octave=bins_per_octave,
                    hop_length=hop_length,
                ).astype(np.complex64, copy=False)

            mag = np.abs(C).astype(np.float32, copy=False)
            if mag.size == 0:
                continue

            mag_t = torch.from_numpy(mag)
            max_per_frame = torch.clamp(mag_t.max(dim=0).values, min=1e-8)
            norm_mag = mag_t / max_per_frame.unsqueeze(0)
            vals, idx = torch.topk(norm_mag, k=topk, dim=0, largest=True, sorted=True)
            idx_int = idx.to(torch.int64)
            rest_mask = vals < magnitude_threshold
            out = idx_int.clone()
            out[rest_mask] = -1
            channel_chunks.append(out)

            del (
                chunk,
                x_np,
                C,
                mag,
                mag_t,
                max_per_frame,
                norm_mag,
                vals,
                idx,
                idx_int,
                rest_mask,
            )

        if channel_chunks:
            mel_list.append(torch.cat(channel_chunks, dim=1))
        else:
            mel_list.append(torch.empty(topk, 0, dtype=torch.int64))

    L, R = mel_list
    interleaved_channels = []
    for i in range(topk):
        interleaved_channels.extend([L[i], R[i]])
    interleaved = torch.stack(interleaved_channels, dim=0)
    return interleaved


def precompute_melody_cache(
    tar_path: str,
    cache_dir: str,
    sample_rate: int = 44100,
    cqt_bins: int = 128,
    bins_per_octave: int = 12,
    topk: int = 4,
    hpf_cutoff_hz: float = 261.2,
    magnitude_threshold: float = 0.1,
    chunk_duration_s: float | None = 60.0,
):
    """
    WebDataset tar ファイルから曲全体のメロディーを事前計算して HDF5 に保存

    HDF5 構造:
      /{sample_key}/melody -> (8, T_full) int64 テンソル (曲全体)
      /{sample_key}/@duration_s -> float (曲の長さ)
      /{sample_key}/@sr -> int (サンプリングレート)
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # tar ファイル名から HDF5 キャッシュファイル名を生成
    tar_name = Path(tar_path).stem
    cache_file = Path(cache_dir) / f"{tar_name}_melody.h5"

    print(f"📊 Creating melody cache: {cache_file}")
    print(f"   Source tar: {tar_path}")

    # Dataset の作成
    fill_missing_keys_and_pad = partial(_fn_extract_stems_and_pad)
    fn_resample = partial(_fn_resample, sample_rate=sample_rate)

    dataset = (
        wds.WebDataset(tar_path)
        .decode(torch_audio)
        .map(fill_missing_keys_and_pad)
        .map(fn_resample)
    )

    # HDF5 ファイルに保存
    with h5py.File(cache_file, "w") as f:
        f.attrs["tar_path"] = str(tar_path)
        f.attrs["sample_rate"] = sample_rate
        f.attrs["cqt_bins"] = cqt_bins
        f.attrs["topk"] = topk

        total_count = 0

        for stems, sr, sample_key in tqdm.tqdm(dataset, desc="Computing full melodies"):
            # 曲全体をミックス
            mix = _mix_stems(stems)

            # 曲全体のメロディーを計算
            melody_full = _compute_topk_cqt_melody(
                mix,
                sample_rate=sr,
                n_bins=cqt_bins,
                bins_per_octave=bins_per_octave,
                topk=topk,
                hpf_cutoff_hz=hpf_cutoff_hz,
                magnitude_threshold=magnitude_threshold,
                hop_length=None,
                chunk_duration_s=chunk_duration_s,
            )

            melody_np = melody_full.numpy().astype(np.int64)
            duration_s = mix.shape[-1] / sr

            # sample_key でグループ作成
            grp = f.create_group(sample_key)
            grp.create_dataset(
                "melody", data=melody_np, compression="gzip", compression_opts=4
            )
            grp.attrs["duration_s"] = float(duration_s)
            grp.attrs["sr"] = int(sr)

            total_count += 1

    print(f"✅ Saved {total_count} full melodies to {cache_file}")
    return cache_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute melody cache from WebDataset tar"
    )
    parser.add_argument("--path", type=str, required=True, help="Path to tar file")
    parser.add_argument("--cache-dir", type=str, default="/app/data/melody_cache")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--cqt-bins", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--hpf-cutoff-hz", type=float, default=261.2)
    parser.add_argument("--magnitude-threshold", type=float, default=0.1)
    parser.add_argument("--chunk-duration", type=float, default=60.0)

    args = parser.parse_args()

    precompute_melody_cache(
        tar_path=args.path,
        cache_dir=args.cache_dir,
        sample_rate=args.sample_rate,
        cqt_bins=args.cqt_bins,
        topk=args.topk,
        hpf_cutoff_hz=args.hpf_cutoff_hz,
        magnitude_threshold=args.magnitude_threshold,
        chunk_duration_s=args.chunk_duration,
    )
