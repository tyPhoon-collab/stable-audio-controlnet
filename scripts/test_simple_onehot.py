"""
SimpleOneHotChordRepresentation のテストスクリプト

25チャンネル (12音×2質+1無和音) のOneHot表現が正しく動作するか確認
"""

import torch
from main.chord_representation import SimpleOneHotChordRepresentation


def test_simple_onehot():
    """基本的な動作テスト"""
    print("=== SimpleOneHotChordRepresentation Test ===\n")

    repr_model = SimpleOneHotChordRepresentation()

    # 出力チャンネル数の確認
    assert repr_model.get_output_channels() == 25
    print(f"✓ Output channels: {repr_model.get_output_channels()}")

    # 設定情報の確認
    config = repr_model.get_config()
    print(f"✓ Config: {config}\n")

    # テストケース1: C major (root=0, quality=0)
    print("Test 1: C major")
    chord_batch = torch.tensor([[[0, 0, 0]]])  # (B=1, T=1, 3)
    output = repr_model(chord_batch)
    print(f"  Input: root=0, quality=0")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (1, 25, 1)
    assert output[0, 0, 0] == 1.0  # C major at index 0
    assert output.sum() == 1.0
    print(f"  ✓ C major encoded at index 0\n")

    # テストケース2: C minor (root=0, quality=1)
    print("Test 2: C minor")
    chord_batch = torch.tensor([[[0, 1, 0]]])
    output = repr_model(chord_batch)
    print(f"  Input: root=0, quality=1")
    assert output[0, 12, 0] == 1.0  # C minor at index 12
    assert output.sum() == 1.0
    print(f"  ✓ C minor encoded at index 12\n")

    # テストケース3: B major (root=11, quality=0)
    print("Test 3: B major")
    chord_batch = torch.tensor([[[11, 0, 0]]])
    output = repr_model(chord_batch)
    print(f"  Input: root=11, quality=0")
    assert output[0, 11, 0] == 1.0  # B major at index 11
    assert output.sum() == 1.0
    print(f"  ✓ B major encoded at index 11\n")

    # テストケース4: B minor (root=11, quality=1)
    print("Test 4: B minor")
    chord_batch = torch.tensor([[[11, 1, 0]]])
    output = repr_model(chord_batch)
    print(f"  Input: root=11, quality=1")
    assert output[0, 23, 0] == 1.0  # B minor at index 23
    assert output.sum() == 1.0
    print(f"  ✓ B minor encoded at index 23\n")

    # テストケース5: No chord (root=-1, quality=-1)
    print("Test 5: No chord")
    chord_batch = torch.tensor([[[-1, -1, 0]]])
    output = repr_model(chord_batch)
    print(f"  Input: root=-1, quality=-1")
    assert output[0, 24, 0] == 1.0  # No chord at index 24
    assert output.sum() == 1.0
    print(f"  ✓ No chord encoded at index 24\n")

    # テストケース6: 複数タイムステップ
    print("Test 6: Multiple timesteps")
    chord_batch = torch.tensor([
        [[0, 0, 0], [4, 1, 0], [-1, -1, 0]]  # C major, E minor, N
    ])  # (B=1, T=3, 3)
    output = repr_model(chord_batch)
    print(f"  Input: C major, E minor, N")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (1, 25, 3)
    assert output[0, 0, 0] == 1.0   # C major at t=0
    assert output[0, 16, 1] == 1.0  # E minor (12+4) at t=1
    assert output[0, 24, 2] == 1.0  # N at t=2
    print(f"  ✓ All timesteps correctly encoded\n")

    # テストケース7: バッチ処理
    print("Test 7: Batch processing")
    chord_batch = torch.tensor([
        [[0, 0, 0]],   # Batch 1: C major
        [[7, 1, 0]],   # Batch 2: G minor
    ])  # (B=2, T=1, 3)
    output = repr_model(chord_batch)
    print(f"  Input: Batch of C major, G minor")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 25, 1)
    assert output[0, 0, 0] == 1.0   # C major
    assert output[1, 19, 0] == 1.0  # G minor (12+7)
    print(f"  ✓ Batch processing works correctly\n")

    # テストケース8: 部分的な無和音
    print("Test 8: Partial no-chord (root=-1, quality=0)")
    chord_batch = torch.tensor([[[-1, 0, 0]]])
    output = repr_model(chord_batch)
    print(f"  Input: root=-1, quality=0")
    assert output[0, 24, 0] == 1.0  # Should be treated as no chord
    assert output.sum() == 1.0
    print(f"  ✓ Partial no-chord handled correctly\n")

    print("Test 9: Partial no-chord (root=0, quality=-1)")
    chord_batch = torch.tensor([[[0, -1, 0]]])
    output = repr_model(chord_batch)
    print(f"  Input: root=0, quality=-1")
    assert output[0, 24, 0] == 1.0  # Should be treated as no chord
    assert output.sum() == 1.0
    print(f"  ✓ Partial no-chord handled correctly\n")

    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

    # インデックスマッピングの表示
    print("\n=== Index Mapping Reference ===")
    print("Major chords (0-11):")
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i, note in enumerate(notes):
        print(f"  {i:2d}: {note} major")

    print("\nMinor chords (12-23):")
    for i, note in enumerate(notes):
        print(f"  {i+12:2d}: {note} minor")

    print("\nNo chord:")
    print(f"  24: N (no chord)")


if __name__ == "__main__":
    test_simple_onehot()
