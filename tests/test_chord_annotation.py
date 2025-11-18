import pytest
import torch

from main.data.annotation import ChordAnnotation, quality_chroma_offsets


@pytest.fixture(scope="module")
def chord_annotation():
    return ChordAnnotation(sample_rate=44100)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("C:maj", (0, 0, 0)),
        ("C#:min", (1, 1, 0)),
        ("D:maj7", (2, 8, 0)),
        ("Bb:7", (10, 9, 0)),
        ("F:maj/3", (5, 0, 3)),
        ("N", (-1, -1, 0)),
        ("G:dim", (7, 2, 0)),
        ("A:sus4", (9, 13, 0)),
    ],
)
def test_parse_chord_symbol(chord_annotation, symbol, expected):
    assert chord_annotation.parse_chord_symbol(symbol) == expected


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        ((0, 0, 0), "C:maj"),
        ((1, 1, 0), "C#:min"),
        ((2, 8, 0), "D:maj7"),
        ((10, 9, 0), "Bb:7"),
        ((5, 0, 3), "F:maj/3"),
        ((-1, -1, 0), "N"),
        ((7, 2, 0), "G:dim"),
        ((9, 13, 0), "A:sus4"),
    ],
)
def test_idx_to_chord_symbol(chord_annotation, encoded, expected):
    root, quality, inversion = encoded
    assert chord_annotation.idx_to_chord_symbol(root, quality, inversion) == expected


def test_chord_timeline_text(chord_annotation):
    chord_tensor = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [7, 0, 0],
            [7, 0, 0],
        ],
        dtype=torch.long,
    )

    result = chord_annotation.chord_timeline_text(chord_tensor, frame_rate=4.0)
    lines = result.splitlines()
    assert lines[0] == "frame_rate: 4.000 Hz"
    assert lines[1] == "frames: 4"
    assert lines[2] == "chords:"
    assert lines[3].endswith("C:maj")
    assert lines[4].endswith("G:maj")


def test_chord_tensor_to_lab_format(chord_annotation):
    chord_tensor = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [7, 0, 0],
            [7, 0, 0],
        ],
        dtype=torch.long,
    )

    lab_text = chord_annotation.chord_tensor_to_lab_format(chord_tensor, frame_rate=4.0)
    lines = lab_text.splitlines()
    assert len(lines) == 2

    first_start, first_end, first_symbol = lines[0].split("\t")
    assert pytest.approx(float(first_start)) == 0.0
    assert pytest.approx(float(first_end)) == 0.5
    assert first_symbol == "C:maj"

    second_start, second_end, second_symbol = lines[1].split("\t")
    assert pytest.approx(float(second_start)) == 0.5
    assert pytest.approx(float(second_end)) == 1.0
    assert second_symbol == "G:maj"


def test_create_chord_tensor(chord_annotation):
    annotations = [
        (0.0, 0.5, "C:maj"),
        (0.5, 1.0, "G:maj"),
    ]
    chord_tensor = chord_annotation.create_chord_tensor(
        annotations=annotations,
        audio_length=44100,
        frame_rate=4.0,
    )

    assert chord_tensor.shape == (4, 3)
    assert tuple(chord_tensor[0].tolist()) == (0, 0, 0)
    assert tuple(chord_tensor[2].tolist()) == (7, 0, 0)


def test_load_and_save_lab_file_roundtrip(tmp_path, chord_annotation):
    lab_file = tmp_path / "test.lab"
    lab_file.write_text(
        "0.000\t0.500\tC:maj\n0.500\t1.000\tG:maj\n1.000\t1.500\tAm:min\n"
    )

    annotations = chord_annotation.load_lab_file(str(lab_file))
    assert annotations == [
        (0.0, 0.5, "C:maj"),
        (0.5, 1.0, "G:maj"),
        (1.0, 1.5, "Am:min"),
    ]

    chord_tensor = chord_annotation.create_chord_tensor(
        annotations=annotations,
        audio_length=44100 * 2,
        frame_rate=2.0,
    )
    generated = chord_annotation.chord_tensor_to_lab_format(
        chord_tensor, frame_rate=2.0
    )
    saved = tmp_path / "resaved.lab"
    saved.write_text(generated)

    reloaded = chord_annotation.load_lab_file(str(saved))
    assert reloaded[0][2] == "C:maj"
    assert reloaded[1][2] == "G:maj"


def test_edge_cases(chord_annotation):
    empty = torch.empty((0, 3), dtype=torch.long)
    assert chord_annotation.chord_tensor_to_lab_format(empty, frame_rate=4.0) == ""

    wrong_shape = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    assert (
        chord_annotation.chord_tensor_to_lab_format(wrong_shape, frame_rate=4.0) == ""
    )

    silent = torch.tensor([[-1, -1, 0], [-1, -1, 0]], dtype=torch.long)
    lab = chord_annotation.chord_tensor_to_lab_format(silent, frame_rate=4.0)
    assert "N" in lab

    high_rate = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)
    lab_high = chord_annotation.chord_tensor_to_lab_format(high_rate, frame_rate=100.0)
    first_line = lab_high.splitlines()[0]
    start, end, _ = first_line.split("\t")
    assert pytest.approx(float(end)) == 0.02


def test_quality_chroma_offsets():
    assert quality_chroma_offsets("maj") == [0, 4, 7]
    assert quality_chroma_offsets("m7") == [0, 3, 7, 10]
    assert quality_chroma_offsets("sus4") == [0, 5, 7]
    assert quality_chroma_offsets("N") == []
