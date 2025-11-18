import torch
import torch.nn.functional as F

from main.chord_conditioner import ChromaChordConditioner
from main.data.annotation import QUALITY_CHROMA_INTERVALS, QUALITY_NAMES


def _quality_index(name: str) -> int:
    return QUALITY_NAMES.index(name)


def test_chroma_conditioner_rotates_chroma_by_root():
    conditioner = ChromaChordConditioner(output_dim=16, embed_dim=8, conv_channels=4)
    chord_data = torch.tensor(
        [
            [0, _quality_index("maj"), 0],
            [4, _quality_index("maj"), 0],
            [7, _quality_index("min"), 0],
        ],
        dtype=torch.long,
    )

    encoded = conditioner.encode_chords(chord_data, torch.device("cpu"))

    assert encoded.shape == (3, 25)

    expected_roots = F.one_hot(torch.tensor([0, 4, 7]), num_classes=12).float()
    assert torch.equal(encoded[:, 0:12], expected_roots)

    expected_chroma = torch.zeros(3, 12)
    maj_intervals = QUALITY_CHROMA_INTERVALS["maj"]
    min_intervals = QUALITY_CHROMA_INTERVALS["min"]
    expected_chroma[0, list(maj_intervals)] = 1
    expected_chroma[1, [((4 + iv) % 12) for iv in maj_intervals]] = 1
    expected_chroma[2, [((7 + iv) % 12) for iv in min_intervals]] = 1

    assert torch.equal(encoded[:, 12:24], expected_chroma)
    assert torch.all(encoded[:, 24] == 0)


def test_chroma_conditioner_sets_no_chord_flag():
    conditioner = ChromaChordConditioner(output_dim=16, embed_dim=8, conv_channels=4)
    chord_data = torch.tensor(
        [
            [-1, -1, 0],
            [0, _quality_index("maj"), 0],
        ],
        dtype=torch.long,
    )

    encoded = conditioner.encode_chords(chord_data, torch.device("cpu"))

    assert encoded[0, 24] == 1
    assert torch.sum(encoded[0, :24]) == 0
    assert encoded[1, 24] == 0


def test_chroma_conditioner_forward_matches_target_size():
    conditioner = ChromaChordConditioner(output_dim=32, embed_dim=16, conv_channels=8)
    chord_data = torch.tensor(
        [
            [0, _quality_index("maj"), 0],
            [4, _quality_index("min"), 0],
            [7, _quality_index("maj7"), 0],
            [-1, -1, 0],
        ],
        dtype=torch.long,
    )
    target_size = 96

    output, mask = conditioner(
        [{"data": chord_data, "target_size": target_size}], device="cpu"
    )

    assert output.shape == (1, 32, target_size)
    assert mask.shape == (1, target_size)
    assert torch.all(mask == 1)
    assert torch.all(torch.isfinite(output))
