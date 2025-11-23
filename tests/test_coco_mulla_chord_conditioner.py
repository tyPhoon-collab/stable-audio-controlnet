"""Test suite for FlexibleChordConditioner."""

import pytest
import torch

from main.chord_conditioner import CocoMullaChordConditioner


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCocoMullaChordConditioner:
    """CocoMullaChordConditionerのテストスイート。"""

    def test_init_with_bass_true(self):
        """with_bass=Trueで37次元を使用。"""
        conditioner = CocoMullaChordConditioner(
            output_dim=64,
            embed_dim=128,
            with_bass=True,
            rotate_chroma=False,
        )
        assert conditioner.input_dim == 37
        assert not conditioner.rotate_chroma

    def test_init_with_bass_false(self):
        """with_bass=Falseで25次元を使用。"""
        conditioner = CocoMullaChordConditioner(
            output_dim=64,
            embed_dim=128,
            with_bass=False,
            rotate_chroma=False,
        )
        assert conditioner.input_dim == 25

    def test_encode_chords_no_chord(self, device):
        """'no-chord' フレームの処理。"""
        conditioner = CocoMullaChordConditioner(with_bass=True)

        # no-chord フレーム
        chords_data = torch.tensor([[-1, -1, -1]], dtype=torch.long, device=device)

        c = conditioner.encode_chords(chords_data, device)

        assert c.shape == (1, 37)
        assert c[0, 36].item() == 1.0  # NoChord flag
        assert torch.sum(c[0, :36]) == 0.0

    def test_encode_chords_basic_chord_with_bass(self, device):
        """基本的な和音 + bass noteのエンコード（37次元）。"""
        conditioner = CocoMullaChordConditioner(with_bass=True, rotate_chroma=False)

        # C:maj with C in bass -> (0, 0, 0)
        chords_data = torch.tensor([[0, 0, 0]], dtype=torch.long, device=device)

        c = conditioner.encode_chords(chords_data, device)

        assert c.shape == (1, 37)
        # Root: one-hot at position 0
        assert c[0, 0].item() == 1.0
        assert torch.sum(c[0, :12]) == 1.0
        # Bass: one-hot at position 12
        assert c[0, 12].item() == 1.0
        assert torch.sum(c[0, 12:24]) == 1.0
        # Chroma: maj = (0, 4, 7)
        chroma = c[0, 24:36]
        assert chroma[0].item() == 1.0
        assert chroma[4].item() == 1.0
        assert chroma[7].item() == 1.0
        assert torch.sum(chroma) == 3.0

    def test_encode_chords_with_bass_note(self, device):
        """Bass noteが異なる場合。"""
        conditioner = CocoMullaChordConditioner(with_bass=True)

        # C:maj with Bb (10) in bass -> (0, 0, 10)
        chords_data = torch.tensor([[0, 0, 10]], dtype=torch.long, device=device)

        c = conditioner.encode_chords(chords_data, device)

        # Root: position 0
        assert c[0, 0].item() == 1.0
        # Bass: position 22 (10 + 12)
        assert c[0, 22].item() == 1.0

    def test_encode_chords_without_bass(self, device):
        """with_bass=Falseで25次元。"""
        conditioner = CocoMullaChordConditioner(with_bass=False, rotate_chroma=False)

        chords_data = torch.tensor([[0, 0, 0]], dtype=torch.long, device=device)
        c = conditioner.encode_chords(chords_data, device)

        assert c.shape == (1, 25)
        # Root: one-hot at 0
        assert c[0, 0].item() == 1.0
        # Chroma: maj = (0, 4, 7)
        chroma = c[0, 12:24]
        assert torch.sum(chroma) == 3.0

    def test_encode_chords_rotated_chroma(self, device):
        """Chroma回転テスト。"""
        conditioner = CocoMullaChordConditioner(with_bass=True, rotate_chroma=True)

        # D:maj (root=2) with D in bass -> (2, 0, 2)
        # maj = (0, 4, 7) 相対
        # rotate=Trueのとき、絶対pitch classが得られる: (2, 6, 9)
        chords_data = torch.tensor([[2, 0, 2]], dtype=torch.long, device=device)

        c = conditioner.encode_chords(chords_data, device)

        # Root: position 2
        assert c[0, 2].item() == 1.0
        # Chroma (回転): 絶対pitch class (2, 6, 9)
        chroma = c[0, 24:36]
        assert chroma[2].item() == 1.0
        assert chroma[6].item() == 1.0
        assert chroma[9].item() == 1.0
        assert torch.sum(chroma) == 3.0

    def test_forward_basic(self, device):
        """Forward passの基本テスト。"""
        conditioner = CocoMullaChordConditioner(
            output_dim=64,
            embed_dim=128,
            conv_channels=32,
            with_bass=True,
        ).to(device)

        # 4フレームの和音
        chords_data = torch.tensor(
            [
                [0, 0, 0],  # C:maj
                [7, 0, 7],  # G:maj
                [0, 0, 0],  # C:maj
                [-1, -1, -1],  # N
            ],
            dtype=torch.long,
            device=device,
        )

        chords_dict = [{"data": chords_data, "target_size": 16}]

        x, attention_mask = conditioner(chords_dict, device)

        # 出力形状を確認
        assert x.shape == (1, 64, 16)  # (batch, output_dim, target_size)
        assert attention_mask.shape == (1, 16)

    def test_forward_different_target_sizes(self, device):
        """異なるtarget_sizeでのテスト。"""
        conditioner = CocoMullaChordConditioner(output_dim=32, with_bass=False).to(
            device
        )

        chords_data = torch.tensor(
            [[0, 0, 0], [7, 0, 7]],
            dtype=torch.long,
            device=device,
        )

        for target_size in [8, 16, 32]:
            chords_dict = [{"data": chords_data, "target_size": target_size}]
            x, _ = conditioner(chords_dict, device)
            assert x.shape[2] == target_size

    def test_forward_batch_processing(self, device):
        """バッチ処理のテスト。"""
        conditioner = CocoMullaChordConditioner(output_dim=64).to(device)

        # 複数フレーム
        chords_data = torch.tensor(
            [
                [0, 0, 0],
                [7, 0, 7],
                [0, 1, 0],
                [5, 1, 5],
                [-1, -1, -1],
            ],
            dtype=torch.long,
            device=device,
        )

        chords_dict = [{"data": chords_data, "target_size": 20}]
        x, attention_mask = conditioner(chords_dict, device)

        assert x.shape == (1, 64, 20)
        assert attention_mask.shape == (1, 20)

    def test_parameters_initialization(self):
        """パラメータの初期化をテスト。"""
        conditioner = CocoMullaChordConditioner(
            output_dim=128,
            embed_dim=256,
            with_bass=True,
        )

        # Parametersが存在することを確認
        assert sum(1 for _ in conditioner.parameters()) > 0

        # Buffersが登録されていることを確認
        assert hasattr(conditioner, "chord_quality_map")


class TestCocoMullaChordConditionerConfiguration:
    """設定パターンのテスト。"""

    @pytest.mark.parametrize(
        "with_bass,rotate_chroma,expected_dim",
        [
            (True, False, 37),
            (True, True, 37),
            (False, False, 25),
            (False, True, 25),
        ],
    )
    def test_dimension_configurations(
        self, with_bass: bool, rotate_chroma: bool, expected_dim: int
    ):
        """異なる設定での次元数を確認。"""
        conditioner = CocoMullaChordConditioner(
            with_bass=with_bass,
            rotate_chroma=rotate_chroma,
        )
        assert conditioner.input_dim == expected_dim

    def test_device_compatibility(self):
        """GPU/CPU互換性のテスト。"""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            device_obj = torch.device(device)
            conditioner = CocoMullaChordConditioner().to(device_obj)
            chords_data = torch.tensor([[0, 0, 0]], dtype=torch.long, device=device_obj)
            c = conditioner.encode_chords(chords_data, device_obj)
            # デバイス型の比較 (cuda と cuda:0 は同じ)
            assert c.device.type == device_obj.type
