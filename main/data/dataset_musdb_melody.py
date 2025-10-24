import random
from functools import partial
from typing import Dict, List

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from torch import Tensor
from torch.utils.data import DataLoader
from torchaudio.functional import highpass_biquad, resample
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

    # サンプル名を取得（__key__が利用可能な場合）
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
    mix = torch.stack(tracks).sum(dim=0)  # (C, T)
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    if mix.size(0) == 1:  # mono -> duplicate to stereo
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
    """
    Compute top-k CQT-based melody indices for stereo waveform.

    Returns tensor of shape (8, T_frames) with values in [0..127] per index and -1 for rest.
    Interleaving order: [L_k1, R_k1, L_k2, R_k2, L_k3, R_k3, L_k4, R_k4].
    """
    # High-pass biquad filter
    x = highpass_biquad(waveform, sample_rate, cutoff_freq=hpf_cutoff_hz)

    # Ensure stereo
    if x.size(0) == 1:
        x = x.repeat(2, 1)

    # Librosa expects numpy float64; compute per channel CQT magnitude
    fmin = librosa.midi_to_hz(0)  # MIDI 0
    if hop_length is None:
        # Choose hop to get a reasonable frame rate (~100 fps)
        hop_length = max(1, int(sample_rate / 100))

    mel_list: List[Tensor] = []  # will collect [L_k1, R_k1, ..., L_k4, R_k4]
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
        mag = np.abs(C).astype(np.float32)  # (n_bins, T)
        mag_t = torch.from_numpy(mag)  # (n_bins, T)

        # Normalize per frame to compute thresholding
        max_per_frame = torch.clamp(mag_t.max(dim=0).values, min=1e-8)  # (T)
        norm_mag = mag_t / max_per_frame.unsqueeze(0)

        # Top-k indices and their normalized magnitudes
        vals, idx = torch.topk(
            norm_mag, k=topk, dim=0, largest=True, sorted=True
        )  # (k, T)

        # Map to 0..127 indices; apply threshold -> -1 rest
        idx_int = idx.to(torch.int64)  # already 0..127
        # Create rest mask for values below threshold
        rest_mask = vals < magnitude_threshold
        # Prepare output array (k, T) with -1 where below threshold
        out = idx_int.clone()
        out[rest_mask] = -1
        mel_list.append(out)

        # 中間テンソルの明示的削除でメモリを解放
        del x_np, C, mag, mag_t, max_per_frame, norm_mag, vals, idx, idx_int, rest_mask

    # Interleave L/R: [L0, R0, L1, R1, L2, R2, L3, R3]
    L, R = mel_list  # each (k, T)
    interleaved = torch.stack([L[0], R[0], L[1], R[1], L[2], R[2], L[3], R[3]], dim=0)
    return interleaved  # (8, T)


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

        # Pad signals to chunk_size if they are shorter
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

            # Drop silent stems combinations (same heuristic as chord ds)
            chunks = {
                k: v
                for k, v in chunks.items()
                if _weights_for_nonzero_refs(v.sum(dim=0))
            }
            if len(chunks) < 2 or (
                len(chunks) == 2 and "vocals" in list(chunks.keys())
            ):
                continue

            # Compute melody from mixture of all stems in this chunk
            mix = _mix_stems(chunks)  # (2, T)
            melody_chunk = _compute_topk_cqt_melody(
                mix,
                sample_rate=sr,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                topk=topk,
                hpf_cutoff_hz=hpf_cutoff_hz,
                magnitude_threshold=magnitude_threshold,
                hop_length=hop_length,
            )  # (8, T_fk)

            yield chunks, melody_chunk, start_s, length / sr, sample_key


def create_musdb_dataset_with_melody(
    path: str,
    sample_rate: int,
    chunk_dur: float,
    shardshuffle: bool = False,
    cqt_bins: int = 128,
    bins_per_octave: int = 12,
    topk: int = 4,
    hpf_cutoff_hz: float = 261.2,
    magnitude_threshold: float = 0.1,
    hop_length: int | None = None,
):
    """
    メロディプロンプト(Top-k CQT, 8xT)付き MUSDB データセットを作成

    Returns WebDataset where each sample yields:
      (chunks: Dict[str, Tensor(C,T)], melody_chunk: Tensor(8,T_fk), start_s: float, total_s: float, sample_key: str)
    """
    fill_missing_keys_and_pad = partial(_fn_extract_stems_and_pad)
    get_slices = partial(
        _get_slices_with_melody,
        chunk_dur=chunk_dur,
        n_bins=cqt_bins,
        bins_per_octave=bins_per_octave,
        topk=topk,
        hpf_cutoff_hz=hpf_cutoff_hz,
        magnitude_threshold=magnitude_threshold,
        hop_length=hop_length,
    )
    fn_resample = partial(_fn_resample, sample_rate=sample_rate)

    dataset = (
        wds.WebDataset(path, shardshuffle=shardshuffle)
        .decode(torch_audio)
        .map(fill_missing_keys_and_pad)
        .map(fn_resample)
        .compose(get_slices)
    )

    return dataset


def collate_fn_melody_conditional(
    samples,
    drop_vocals: bool = True,
    prompt_text: str | None = None,
):
    """Top-k CQTメロディ付きのconditional collate関数

    Returns tuple of length 6:
      (outputs, inputs, prompts, start_seconds, total_seconds, melody_batch)
    where melody_batch has shape (B, 8, T_fk)
    """
    start_seconds = [x for _, _, x, _, _ in samples]
    total_seconds = [x for _, _, _, x, _ in samples]
    melody_chunks = [x for _, x, _, _, _ in samples]
    # sample_keys = [x for _, _, _, _, x in samples]
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

    melody_batch = torch.stack(melody_chunks)  # (B, 8, T_fk)
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
    """Top-k CQTメロディ付きのmix collate関数

    Returns tuple of length 5:
      (outputs, prompts, start_seconds, total_seconds, melody_batch)
    where melody_batch has shape (B, 8, T_fk)
    """
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

    melody_batch = torch.stack(melody_chunks)  # (B, 8, T_fk)
    return (
        torch.concat(outputs),
        prompts,
        start_seconds,
        total_seconds,
        melody_batch,
    )


if __name__ == "__main__":
    # 簡易自己テスト
    print("=== メロディ付きのテスト（Top-k CQT使用） ===")
    dataset_with_melody = create_musdb_dataset_with_melody(
        path="/app/data/musdb18hq/train.tar",
        sample_rate=44100,
        chunk_dur=47.57,
        cqt_bins=128,
        topk=4,
        hpf_cutoff_hz=261.2,
        magnitude_threshold=0.1,
    )
    dataloader_with_melody = DataLoader(
        dataset_with_melody,
        batch_size=2,
        pin_memory=True,
        collate_fn=collate_fn_melody_mix,
        num_workers=0,
    )
    for i, batch in enumerate(dataloader_with_melody):
        print(f"Batch {i}: {len(batch)} elements")
        outputs, prompts, start_seconds, total_seconds, melody_batch = batch
        print(f"  Audio outputs: {outputs.shape}")
        print(
            f"  Melody batch: {melody_batch.shape} (values min={melody_batch.min().item()}, max={melody_batch.max().item()})"
        )
        if i >= 1:  # 2バッチだけテスト
            break
