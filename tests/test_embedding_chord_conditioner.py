import torch

from main.chord_conditioner import EmbeddingChordConditioner


def test_embedding_conditioner_rescales_to_target_size():
    conditioner = EmbeddingChordConditioner(output_dim=32, embed_dim=8, conv_channels=4)
    chord_data = torch.tensor(
        [
            [0, 0, 0],
            [4, 1, 0],
            [7, 0, 0],
            [-1, -1, 0],
            [11, 1, 0],
        ],
        dtype=torch.long,
    )
    target_size = 160

    output, mask = conditioner(
        [{"data": chord_data, "target_size": target_size}],
        device="cpu",
    )

    assert output.shape == (1, 32, target_size)
    assert mask.shape == (1, target_size)
    assert torch.all(mask == 1)
    assert torch.all(torch.isfinite(output))


def test_embedding_conditioner_handles_no_chord_tokens():
    conditioner = EmbeddingChordConditioner(output_dim=16, embed_dim=4, conv_channels=2)
    chord_data = torch.tensor(
        [
            [-1, -1, 0],
            [0, -1, 0],
            [-1, 0, 0],
            [3, 1, 0],
        ],
        dtype=torch.long,
    )

    output, mask = conditioner(
        [{"data": chord_data, "target_size": 32}],
        device="cpu",
    )

    assert output.shape == (1, 16, 32)
    assert torch.all(torch.isfinite(output))
    assert torch.all(mask == 1)
