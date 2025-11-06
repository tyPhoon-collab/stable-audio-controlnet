import math
from typing import Dict, Iterable, Tuple

import pytest
import torch

from main.data.dataset_musdb_chord import (
    _get_slices,
    collate_fn_conditional,
    collate_fn_mix,
)


def make_mock_sample(
    length_samples: int,
    sr: int,
    chord_frame_rate: float,
    n_stems: int = 4,
    channels: int = 2,
    sample_key: str = "mock-track",
) -> Tuple[Dict[str, torch.Tensor], int, torch.Tensor, str]:
    torch.manual_seed(0)
    stem_names = ["drums", "bass", "other", "vocals"][:n_stems]
    stems = {}
    t = torch.arange(length_samples, dtype=torch.float32) / sr
    base_signal = 0.1 * torch.sin(2 * math.pi * 220.0 * t)
    noise = 1e-2 * torch.randn_like(base_signal)
    waveform = (base_signal + noise).unsqueeze(0).repeat(channels, 1)
    for name in stem_names:
        stems[name] = waveform.clone()

    duration = length_samples / sr
    n_frames = max(1, int(duration * chord_frame_rate))
    chord_tensor = torch.randint(low=0, high=12, size=(n_frames, 3), dtype=torch.long)
    return stems, sr, chord_tensor, sample_key


def collect_slices(
    samples: Iterable[Tuple[Dict[str, torch.Tensor], int, torch.Tensor, str]],
    chunk_dur: float,
    chord_frame_rate: float,
):
    return list(_get_slices(samples, chunk_dur=chunk_dur, chord_frame_rate=chord_frame_rate))


@pytest.mark.parametrize(
    ("length_sec", "chunk_dur", "chord_rate"),
    [
        (1.0, 2.5, 24.0),
        (2.5, 2.5, 24.0),
        (6.3, 2.5, 24.0),
        (5.0, 2.0, 25.0),
    ],
)
def test_get_slices_shapes(length_sec, chunk_dur, chord_rate):
    sr = 44100
    length_samples = int(length_sec * sr)
    sample = make_mock_sample(length_samples, sr, chord_rate)

    results = collect_slices([sample], chunk_dur=chunk_dur, chord_frame_rate=chord_rate)
    assert results, "expected at least one slice"

    expected_chunk_size = int(sr * chunk_dur)
    expected_frames = int(chunk_dur * chord_rate)

    for chunks, chord_chunk, start_s, total_s, _ in results:
        assert 0.0 <= start_s
        upper_bound = max(0.0, total_s - chunk_dur)
        assert start_s <= upper_bound + 1e-3
        for tensor in chunks.values():
            assert tensor.ndim == 2
            channels, timesteps = tensor.shape
            assert timesteps == expected_chunk_size
            assert channels in (1, 2)
        assert chord_chunk.shape == (expected_frames, 3)


def test_collate_functions_shapes():
    sr = 44100
    chunk_dur = 2.5
    chord_rate = 24.0
    expected_chunk_size = int(sr * chunk_dur)
    expected_frames = int(chunk_dur * chord_rate)

    sample_a = make_mock_sample(int(3.1 * sr), sr, chord_rate, sample_key="track-A")
    sample_b = make_mock_sample(int(5.0 * sr), sr, chord_rate, sample_key="track-B")

    slice_a = collect_slices([sample_a], chunk_dur, chord_rate)[0]
    slice_b = collect_slices([sample_b], chunk_dur, chord_rate)[0]

    batch_samples = [slice_a, slice_b]

    outputs_mix, prompts_mix, starts_mix, totals_mix, chord_mix = collate_fn_mix(batch_samples)
    assert outputs_mix.shape[0] == 2
    assert outputs_mix.shape[1] in (1, 2)
    assert outputs_mix.shape[2] == expected_chunk_size
    assert chord_mix.shape == (2, expected_frames, 3)
    assert len(prompts_mix) == len(starts_mix) == len(totals_mix) == 2

    (
        outputs_cond,
        inputs_cond,
        prompts_cond,
        starts_cond,
        totals_cond,
        chord_cond,
    ) = collate_fn_conditional(batch_samples)

    assert outputs_cond.shape[0] == 2
    assert outputs_cond.shape[1] in (1, 2)
    assert outputs_cond.shape[2] == expected_chunk_size
    assert inputs_cond.shape == outputs_cond.shape
    assert chord_cond.shape == (2, expected_frames, 3)
    assert len(prompts_cond) == len(starts_cond) == len(totals_cond) == 2
