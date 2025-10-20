"""
SimpleOneHotChordRepresentation の統合テスト

Modelクラスとの統合を確認
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf


def test_model_integration():
    """Modelクラスとの統合テスト"""
    print("=== Model Integration Test ===\n")

    # Hydraを使わずに直接インスタンス化
    from main.chord_representation import SimpleOneHotChordRepresentation

    repr_model = SimpleOneHotChordRepresentation()

    print(f"Output channels: {repr_model.get_output_channels()}")
    print(f"Config: {repr_model.get_config()}\n")

    # ダミーデータで動作確認
    B, T = 2, 100
    chord_batch = torch.randint(-1, 12, (B, T, 3))
    chord_batch[:, :, 1] = torch.randint(-1, 2, (B, T))  # quality: 0(maj), 1(min), -1(N)

    print(f"Input shape: {chord_batch.shape}")
    output = repr_model(chord_batch)
    print(f"Output shape: {output.shape}")

    assert output.shape == (B, 25, T)
    assert output.min() >= 0.0
    assert output.max() <= 1.0

    # 各タイムステップでちょうど1つのチャンネルが1.0であることを確認
    sums = output.sum(dim=1)  # (B, T)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    print("✓ All integration tests passed!\n")

    # 統計情報
    major_count = (chord_batch[:, :, 1] == 0).sum()
    minor_count = (chord_batch[:, :, 1] == 1).sum()
    no_chord_count = ((chord_batch[:, :, 0] == -1) | (chord_batch[:, :, 1] == -1)).sum()

    print(f"Statistics:")
    print(f"  Major chords: {major_count}")
    print(f"  Minor chords: {minor_count}")
    print(f"  No chord: {no_chord_count}")
    print(f"  Total: {B * T}")


if __name__ == "__main__":
    test_model_integration()
