import pytest
import torch
import torch.nn.functional as F

from main.data.annotation import ChordAnnotation


@pytest.mark.parametrize("frame_rate", [10, 25, 50, 100])
def test_chord_tensor_matches_audio_duration(frame_rate):
    chord_annotation = ChordAnnotation(sample_rate=44100)
    audio_length = 44100 * 2
    annotations = [
        (0.0, 1.0, "C:maj"),
        (1.0, 2.0, "G:maj"),
    ]

    chord_tensor = chord_annotation.create_chord_tensor(
        annotations=annotations,
        audio_length=audio_length,
        frame_rate=frame_rate,
    )

    duration = chord_tensor.shape[0] / frame_rate
    assert pytest.approx(duration, rel=1e-3) == audio_length / chord_annotation.sample_rate


def test_control_signal_interpolation_matches_original_symbols():
    chord_annotation = ChordAnnotation(sample_rate=44100)
    annotations = [
        (0.0, 0.5, "C:maj"),
        (0.5, 1.0, "G:maj"),
    ]

    low_rate = 10
    high_rate = 100
    audio_length = 44100

    low_tensor = chord_annotation.create_chord_tensor(
        annotations=annotations,
        audio_length=audio_length,
        frame_rate=low_rate,
    )
    high_tensor = chord_annotation.create_chord_tensor(
        annotations=annotations,
        audio_length=audio_length,
        frame_rate=high_rate,
    )

    upsampled = F.interpolate(
        low_tensor.float().unsqueeze(0).transpose(1, 2),
        scale_factor=high_rate / low_rate,
        mode="nearest",
    ).transpose(1, 2).squeeze(0).round().long()

    assert upsampled.shape == high_tensor.shape
    assert torch.equal(upsampled, high_tensor)
