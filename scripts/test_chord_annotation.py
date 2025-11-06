"""
ChordAnnotationクラスの動作検証テスト
"""

import tempfile
from pathlib import Path

import torch

from main.data.annotation import ChordAnnotation


def test_parse_chord_symbol():
    """和音記号のパースをテスト"""
    print("=" * 80)
    print("TEST: parse_chord_symbol")
    print("=" * 80)

    chord_annotation = ChordAnnotation()

    test_cases = [
        ("C:maj", (0, 0, 0)),
        ("C#:min", (1, 1, 0)),
        ("D:maj7", (2, 2, 0)),
        ("Bb:7", (10, 4, 0)),
        ("F:maj/3", (5, 0, 3)),
        ("N", (-1, -1, 0)),
        ("G:dim", (7, 5, 0)),
        ("A:sus4", (9, 7, 0)),
    ]

    for chord_symbol, expected in test_cases:
        result = chord_annotation.parse_chord_symbol(chord_symbol)
        status = "✓" if result == expected else "✗"
        print(f"{status} {chord_symbol:12} -> {result} (expected: {expected})")

    print()


def test_idx_to_chord_symbol():
    """数値表現を和音記号に変換するテスト"""
    print("=" * 80)
    print("TEST: idx_to_chord_symbol")
    print("=" * 80)

    chord_annotation = ChordAnnotation()

    test_cases = [
        ((0, 0, 0), "C:maj"),
        ((1, 1, 0), "C#:min"),
        ((2, 2, 0), "D:maj7"),
        ((10, 4, 0), "Bb:7"),
        ((5, 0, 3), "F:maj/3"),
        ((-1, -1, 0), "N"),
        ((7, 5, 0), "G:dim"),
        ((9, 7, 0), "A:sus4"),
    ]

    for (root, quality, inversion), expected in test_cases:
        result = chord_annotation.idx_to_chord_symbol(root, quality, inversion)
        status = "✓" if result == expected else "✗"
        print(
            f"{status} ({root}, {quality}, {inversion}) -> {result:12} (expected: {expected})"
        )

    print()


def test_chord_timeline_text():
    """可読なタイムライン文字列生成をテスト"""
    print("=" * 80)
    print("TEST: chord_timeline_text")
    print("=" * 80)

    chord_annotation = ChordAnnotation(sample_rate=44100)

    # テスト用の和音テンソルを作成
    # 4つのフレーム：C:maj -> C:maj -> G:maj -> G:maj（フレームレート: 4.0Hz）
    chord_tensor = torch.tensor(
        [
            [0, 0, 0],  # C:maj
            [0, 0, 0],  # C:maj
            [7, 0, 0],  # G:maj
            [7, 0, 0],  # G:maj
        ],
        dtype=torch.long,
    )

    result = chord_annotation.chord_timeline_text(chord_tensor, frame_rate=4.0)
    print("Output:")
    print(result)
    print()

    # 検証
    lines = result.split("\n")
    assert "frame_rate: 4.000 Hz" in lines[0], "Frame rate header not found"
    assert "frames: 4" in lines[1], "Frames header not found"
    assert "chords:" in lines[2], "Chords header not found"
    print("✓ Timeline text format is correct")
    print()


def test_chord_tensor_to_lab_format():
    """完全な.lab形式（TSV形式）生成をテスト"""
    print("=" * 80)
    print("TEST: chord_tensor_to_lab_format")
    print("=" * 80)

    chord_annotation = ChordAnnotation(sample_rate=44100)

    # テスト用の和音テンソルを作成
    chord_tensor = torch.tensor(
        [
            [0, 0, 0],  # C:maj
            [0, 0, 0],  # C:maj
            [7, 0, 0],  # G:maj
            [7, 0, 0],  # G:maj
        ],
        dtype=torch.long,
    )

    result = chord_annotation.chord_tensor_to_lab_format(chord_tensor, frame_rate=4.0)
    print("Output (.lab format):")
    print(result)
    print()

    # 検証
    lines = result.split("\n")
    assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

    # 最初の行：C:maj (0.000-0.500)
    parts = lines[0].split("\t")
    assert len(parts) == 3, f"Expected 3 columns, got {len(parts)}"
    assert float(parts[0]) == 0.0, "Start time should be 0.0"
    assert float(parts[1]) == 0.5, "End time should be 0.5"
    assert parts[2] == "C:maj", "Chord should be C:maj"

    # 2番目の行：G:maj (0.500-1.000)
    parts = lines[1].split("\t")
    assert float(parts[0]) == 0.5, "Start time should be 0.5"
    assert float(parts[1]) == 1.0, "End time should be 1.0"
    assert parts[2] == "G:maj", "Chord should be G:maj"

    print(
        "✓ .lab format is correct (TSV format with start_time, end_time, chord_symbol)"
    )
    print()


def test_create_chord_tensor():
    """和音テンソル作成をテスト"""
    print("=" * 80)
    print("TEST: create_chord_tensor")
    print("=" * 80)

    chord_annotation = ChordAnnotation(sample_rate=44100)

    # アノテーションを作成（時間ベース）
    annotations = [
        (0.0, 0.5, "C:maj"),  # 0.0秒～0.5秒: C:maj
        (0.5, 1.0, "G:maj"),  # 0.5秒～1.0秒: G:maj
    ]

    # 44100サンプル（1秒）、フレームレート4.0Hz => 4フレーム
    audio_length = 44100
    frame_rate = 4.0

    chord_tensor = chord_annotation.create_chord_tensor(
        annotations, audio_length, frame_rate
    )

    print(f"Chord tensor shape: {chord_tensor.shape}")
    print("Expected shape: (4, 3) [4 frames, 3 values per frame]")
    assert chord_tensor.shape == (4, 3), f"Shape mismatch: {chord_tensor.shape}"

    # フレーム内容を確認
    print("\nFrame-by-frame content:")
    for i in range(chord_tensor.shape[0]):
        root, quality, inversion = chord_tensor[i].tolist()
        sym = chord_annotation.idx_to_chord_symbol(
            int(root), int(quality), int(inversion)
        )
        time_sec = i / frame_rate
        print(f"  Frame {i} ({time_sec:.1f}s): {sym}")

    # 検証
    assert chord_tensor[0, 0] == 0 and chord_tensor[0, 1] == 0, (
        "Frame 0 should be C:maj"
    )
    assert chord_tensor[2, 0] == 7 and chord_tensor[2, 1] == 0, (
        "Frame 2 should be G:maj"
    )

    print("\n✓ Chord tensor creation is correct")
    print()


def test_load_and_save_lab_file():
    """ファイルI/Oをテスト（.labファイルの読み込みと保存）"""
    print("=" * 80)
    print("TEST: load_lab_file and save via chord_tensor_to_lab_format")
    print("=" * 80)

    chord_annotation = ChordAnnotation(sample_rate=44100)

    # テンポラリファイルを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        lab_file = tmpdir / "test.lab"

        # テスト用の.labファイルを作成
        lab_content = "0.000\t0.500\tC:maj\n0.500\t1.000\tG:maj\n1.000\t1.500\tAm:min\n"
        lab_file.write_text(lab_content)

        print(f"Created test .lab file: {lab_file}")
        print("Content:")
        print(lab_content)

        # ファイルを読み込み
        annotations = chord_annotation.load_lab_file(str(lab_file))
        print(f"Loaded {len(annotations)} annotations")
        for start, end, chord in annotations:
            print(f"  {start:.3f} - {end:.3f} : {chord}")

        # 検証
        assert len(annotations) == 3, f"Expected 3 annotations, got {len(annotations)}"
        assert annotations[0] == (0.0, 0.5, "C:maj"), "First annotation mismatch"
        assert annotations[1] == (0.5, 1.0, "G:maj"), "Second annotation mismatch"
        assert annotations[2] == (1.0, 1.5, "Am:min"), "Third annotation mismatch"

        print("\n✓ .lab file loading is correct")

        # 和音テンソルを作成して再度保存
        chord_tensor = chord_annotation.create_chord_tensor(
            annotations, 44100 * 2, frame_rate=2.0
        )
        lab_text = chord_annotation.chord_tensor_to_lab_format(
            chord_tensor, frame_rate=2.0
        )

        print("\nGenerated .lab format text:")
        print(lab_text)

        # 再度読み込んで検証
        saved_lab_file = tmpdir / "resaved.lab"
        saved_lab_file.write_text(lab_text)
        reloaded_annotations = chord_annotation.load_lab_file(str(saved_lab_file))

        print(f"\nReloaded {len(reloaded_annotations)} annotations")
        for start, end, chord in reloaded_annotations:
            print(f"  {start:.3f} - {end:.3f} : {chord}")

        print("\n✓ Save and reload cycle is correct")

    print()


def test_edge_cases():
    """エッジケースをテスト"""
    print("=" * 80)
    print("TEST: Edge cases")
    print("=" * 80)

    chord_annotation = ChordAnnotation(sample_rate=44100)

    # 1. 空のテンソル
    print("1. Empty tensor:")
    empty_tensor = torch.tensor([], dtype=torch.long).reshape(0, 3)
    result = chord_annotation.chord_tensor_to_lab_format(empty_tensor, frame_rate=4.0)
    print(f"   Result: '{result}'")
    assert result == "", "Empty tensor should return empty string"
    print("   ✓ Empty tensor handled correctly")

    # 2. 間違ったシェイプのテンソル
    print("\n2. Wrong shape tensor:")
    wrong_shape = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    result = chord_annotation.chord_tensor_to_lab_format(wrong_shape, frame_rate=4.0)
    print(f"   Result: '{result}'")
    assert result == "", "Wrong shape tensor should return empty string"
    print("   ✓ Wrong shape handled correctly")

    # 3. すべて無音（N）の場合
    print("\n3. All N (silence) chords:")
    silent_tensor = torch.tensor([[-1, -1, 0], [-1, -1, 0]], dtype=torch.long)
    result = chord_annotation.chord_tensor_to_lab_format(silent_tensor, frame_rate=4.0)
    print(f"   Result:\n{result}")
    assert "N" in result, "Silent chords should contain 'N'"
    print("   ✓ Silent chords handled correctly")

    # 4. 高いフレームレート
    print("\n4. High frame rate:")
    chord_tensor = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)
    result = chord_annotation.chord_tensor_to_lab_format(chord_tensor, frame_rate=100.0)
    print(f"   Result:\n{result}")
    lines = result.split("\n")
    first_line = lines[0].split("\t")
    assert float(first_line[1]) == 0.02, (
        "Second frame should end at 0.02 with 100Hz frame rate"
    )
    print("   ✓ High frame rate handled correctly")

    print()


def run_all_tests():
    """すべてのテストを実行"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ChordAnnotation Test Suite" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        test_parse_chord_symbol()
        test_idx_to_chord_symbol()
        test_chord_timeline_text()
        test_chord_tensor_to_lab_format()
        test_create_chord_tensor()
        test_load_and_save_lab_file()
        test_edge_cases()

        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 25 + "✓ All tests passed!" + " " * 33 + "║")
        print("╚" + "=" * 78 + "╝")
        print()

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
