#!/usr/bin/env python3
"""
Chord Representation Module のテストスクリプト

異なる和音表現方式の動作確認を行います。
"""

import torch


def create_chord_representation(
    representation_type: str = "onehot",
    embed_dim: int = 16,
    simplified_use_embedding: bool = False,
) -> ChordRepresentation:
    """
    Factory関数: 和音表現オブジェクトを生成

    Args:
        representation_type: "onehot", "embedding", "simplified"
        embed_dim: Embedding次元数（embedding/simplified用）
        simplified_use_embedding: simplified使用時にEmbeddingを使うか

    Returns:
        ChordRepresentation インスタンス
    """
    if representation_type == "onehot":
        return OneHotChordRepresentation()
    elif representation_type == "embedding":
        return EmbeddingChordRepresentation(embed_dim=embed_dim)
    elif representation_type == "simplified":
        return SimplifiedChordRepresentation(
            use_embedding=simplified_use_embedding,
            embed_dim=embed_dim
        )
    else:
        raise ValueError(
            f"Unknown representation_type: {representation_type}. "
            f"Supported: 'onehot', 'embedding', 'simplified'"
        )


def test_representation(repr_type: str, **kwargs):
    """和音表現のテスト"""
    print(f"\n{'='*60}")
    print(f"Testing: {repr_type}")
    print(f"Config: {kwargs}")
    print(f"{'='*60}")

    # 和音表現オブジェクトを作成
    chord_repr = create_chord_representation(repr_type, **kwargs)

    # 設定情報を表示
    config = chord_repr.get_config()
    print(f"\n[Configuration]")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # テストデータを作成
    batch_size = 2
    time_frames = 10

    # 和音データ: (B, T, 3) [root, quality, inversion]
    # root: 0-11 (C-B) or -1 (N)
    # quality: 0-8 or -1 (N)
    # inversion: 0 (使用されない)
    chord_batch = torch.tensor([
        # Sample 1
        [
            [0, 0, 0],   # C:maj
            [7, 1, 0],   # G:min
            [2, 4, 0],   # D:7
            [5, 0, 0],   # F:maj
            [9, 1, 0],   # A:min
            [-1, -1, 0], # N (no chord)
            [4, 0, 0],   # E:maj
            [11, 5, 0],  # B:dim
            [0, 2, 0],   # C:maj7
            [7, 0, 0],   # G:maj
        ],
        # Sample 2
        [
            [2, 0, 0],   # D:maj
            [9, 0, 0],   # A:maj
            [4, 1, 0],   # E:min
            [11, 1, 0],  # B:min
            [7, 4, 0],   # G:7
            [0, 3, 0],   # C:min7
            [-1, -1, 0], # N
            [5, 0, 0],   # F:maj
            [10, 0, 0],  # Bb:maj
            [2, 1, 0],   # D:min
        ],
    ])  # (2, 10, 3)

    print(f"\n[Input]")
    print(f"  Shape: {chord_batch.shape}")
    print(f"  Sample chords (batch 0):")
    chord_names = ["C:maj", "G:min", "D:7", "F:maj", "A:min", "N", "E:maj", "B:dim", "C:maj7", "G:maj"]
    for i, name in enumerate(chord_names):
        print(f"    Frame {i}: {name} → {chord_batch[0, i].tolist()}")

    # 推論
    try:
        output = chord_repr(chord_batch)
        print(f"\n[Output]")
        print(f"  Shape: {output.shape}")
        print(f"  Expected: (batch={batch_size}, channels={chord_repr.get_output_channels()}, time={time_frames})")

        # 統計情報
        print(f"\n[Statistics]")
        print(f"  Min: {output.min().item():.4f}")
        print(f"  Max: {output.max().item():.4f}")
        print(f"  Mean: {output.mean().item():.4f}")
        print(f"  Std: {output.std().item():.4f}")

        # メモリ使用量
        memory_mb = output.element_size() * output.numel() / (1024 ** 2)
        print(f"  Memory: {memory_mb:.2f} MB")

        print(f"\n✅ Test passed for {repr_type}")
        return True

    except Exception as e:
        print(f"\n❌ Test failed for {repr_type}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """すべての和音表現をテスト"""
    print("=" * 60)
    print("Chord Representation Module - Test Suite")
    print("=" * 60)

    results = {}

    # 1. OneHot表現
    results['onehot'] = test_representation("onehot")

    # 2. Embedding表現（16次元）
    results['embedding_16'] = test_representation("embedding", embed_dim=16)

    # 3. Embedding表現（32次元）
    results['embedding_32'] = test_representation("embedding", embed_dim=32)

    # 4. Simplified OneHot表現
    results['simplified_onehot'] = test_representation(
        "simplified",
        simplified_use_embedding=False
    )

    # 5. Simplified Embedding表現（8次元）
    results['simplified_emb_8'] = test_representation(
        "simplified",
        embed_dim=8,
        simplified_use_embedding=True
    )

    # 6. Simplified Embedding表現（16次元）
    results['simplified_emb_16'] = test_representation(
        "simplified",
        embed_dim=16,
        simplified_use_embedding=True
    )

    # 結果サマリー
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:25s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit(main())
