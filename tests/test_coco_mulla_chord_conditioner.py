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


class TestCocoMullaChordConditionerBatchOptimization:
    """バッチ処理最適化のテストスイート。"""

    def test_encode_chords_single_sample(self, device):
        """単一サンプル（バッチなし）の処理。"""
        conditioner = CocoMullaChordConditioner(with_bass=True, rotate_chroma=False)

        # (T_frames, 3) の入力
        chords_data = torch.tensor(
            [[0, 0, 0], [7, 0, 7], [-1, -1, -1]], dtype=torch.long, device=device
        )

        c = conditioner.encode_chords(chords_data, device)

        # 出力は (T_frames, 37) でバッチ次元なし
        assert c.shape == (3, 37)
        assert c[0, 0].item() == 1.0  # C:maj root
        assert c[1, 7].item() == 1.0  # G:maj root
        assert c[2, 36].item() == 1.0  # NoChord flag

    def test_encode_chords_batch(self, device):
        """バッチ処理のテスト。"""
        conditioner = CocoMullaChordConditioner(with_bass=True, rotate_chroma=False)

        # (batch_size=2, T_frames=3, 3) の入力
        chords_data = torch.tensor(
            [
                [[0, 0, 0], [7, 0, 7], [-1, -1, -1]],  # バッチ1
                [[2, 0, 2], [9, 0, 9], [4, 1, 4]],  # バッチ2
            ],
            dtype=torch.long,
            device=device,
        )

        c = conditioner.encode_chords(chords_data, device)

        # 出力は (batch_size=2, T_frames=3, 37)
        assert c.shape == (2, 3, 37)

        # バッチ1の検証
        assert c[0, 0, 0].item() == 1.0  # C:maj root
        assert c[0, 1, 7].item() == 1.0  # G:maj root
        assert c[0, 2, 36].item() == 1.0  # NoChord

        # バッチ2の検証
        assert c[1, 0, 2].item() == 1.0  # D:maj root
        assert c[1, 1, 9].item() == 1.0  # A:maj root
        assert c[1, 2, 4].item() == 1.0  # E:min root

    def test_encode_chords_batch_with_bass(self, device):
        """バッチ処理でbass noteが正しく処理されるかテスト。"""
        conditioner = CocoMullaChordConditioner(with_bass=True, rotate_chroma=False)

        # 異なるbass noteを持つバッチ
        chords_data = torch.tensor(
            [
                [[0, 0, 0], [0, 0, 7]],  # C:maj with C, C:maj with G
                [[7, 0, 7], [7, 0, 2]],  # G:maj with G, G:maj with D
            ],
            dtype=torch.long,
            device=device,
        )

        c = conditioner.encode_chords(chords_data, device)

        assert c.shape == (2, 2, 37)

        # バッチ1: C:maj with C bass (root=0, bass=0)
        assert c[0, 0, 0].item() == 1.0  # root
        assert c[0, 0, 12].item() == 1.0  # bass at position 12 (0+12)

        # バッチ1: C:maj with G bass (root=0, bass=7)
        assert c[0, 1, 0].item() == 1.0  # root
        assert c[0, 1, 19].item() == 1.0  # bass at position 19 (7+12)

    def test_encode_chords_batch_rotated_chroma(self, device):
        """バッチ処理でchroma回転が正しく動作するかテスト。"""
        conditioner = CocoMullaChordConditioner(with_bass=True, rotate_chroma=True)

        # 異なるrootを持つバッチ
        chords_data = torch.tensor(
            [
                [[0, 0, 0]],  # C:maj, chroma should be (0, 4, 7)
                [[2, 0, 2]],  # D:maj, chroma should be (2, 6, 9)
            ],
            dtype=torch.long,
            device=device,
        )

        c = conditioner.encode_chords(chords_data, device)

        # バッチ1: C:maj (回転後も0, 4, 7)
        chroma1 = c[0, 0, 24:36]
        assert chroma1[0].item() == 1.0
        assert chroma1[4].item() == 1.0
        assert chroma1[7].item() == 1.0
        assert torch.sum(chroma1) == 3.0

        # バッチ2: D:maj (回転後は2, 6, 9)
        chroma2 = c[1, 0, 24:36]
        assert chroma2[2].item() == 1.0
        assert chroma2[6].item() == 1.0
        assert chroma2[9].item() == 1.0
        assert torch.sum(chroma2) == 3.0

    def test_forward_with_batch(self, device):
        """forward()がバッチ入力を正しく処理するかテスト。"""
        conditioner = CocoMullaChordConditioner(
            output_dim=64, embed_dim=128, with_bass=True
        ).to(device)

        # (batch_size=2, T_frames=4, 3)
        chords_data = torch.tensor(
            [
                [[0, 0, 0], [7, 0, 7], [0, 0, 0], [-1, -1, -1]],
                [[2, 0, 2], [9, 0, 9], [4, 1, 4], [11, 1, 11]],
            ],
            dtype=torch.long,
            device=device,
        )

        chords_dict = [{"data": chords_data, "target_size": 16}]

        x, attention_mask = conditioner(chords_dict, device)

        # 出力形状: (batch_size=2, output_dim=64, target_size=16)
        assert x.shape == (2, 64, 16)
        assert attention_mask.shape == (2, 16)
        assert torch.all(attention_mask == 1.0)

    def test_forward_batch_sizes(self, device):
        """異なるバッチサイズでのforward処理。"""
        conditioner = CocoMullaChordConditioner(output_dim=32).to(device)

        for batch_size in [1, 2, 4, 8]:
            chords_data = torch.randint(
                -1, 12, (batch_size, 5, 3), dtype=torch.long, device=device
            )
            # 一部を有効なコードに設定
            chords_data[:, :, 0] = torch.clamp(chords_data[:, :, 0], 0, 11)
            chords_data[:, :, 1] = torch.clamp(chords_data[:, :, 1], 0, 13)
            chords_data[:, :, 2] = torch.clamp(chords_data[:, :, 2], 0, 11)

            chords_dict = [{"data": chords_data, "target_size": 20}]
            x, attention_mask = conditioner(chords_dict, device)

            assert x.shape == (batch_size, 32, 20)
            assert attention_mask.shape == (batch_size, 20)

    def test_batch_consistency_with_single(self, device):
        """バッチ処理と単一処理の結果が一致するかテスト。"""
        conditioner = CocoMullaChordConditioner(with_bass=True, rotate_chroma=False)

        # 同じデータを単一とバッチで処理
        single_data = torch.tensor(
            [[0, 0, 0], [7, 0, 7]], dtype=torch.long, device=device
        )

        batch_data = single_data.unsqueeze(0)  # (1, 2, 3)

        # 単一処理
        c_single = conditioner.encode_chords(single_data, device)

        # バッチ処理
        c_batch = conditioner.encode_chords(batch_data, device)

        # バッチの最初の要素と単一処理の結果が一致するか
        assert torch.allclose(c_single, c_batch[0], atol=1e-6)

    def test_large_batch_performance(self, device):
        """大きなバッチサイズでのパフォーマンステスト。"""
        conditioner = CocoMullaChordConditioner(output_dim=64).to(device)

        # 大きなバッチ
        batch_size = 16
        T_frames = 50
        chords_data = torch.randint(
            0, 12, (batch_size, T_frames, 3), dtype=torch.long, device=device
        )

        # エンコード処理が正常に完了するか
        c = conditioner.encode_chords(chords_data, device)
        assert c.shape == (batch_size, T_frames, 37)

        # Forward処理も確認
        chords_dict = [{"data": chords_data, "target_size": 100}]
        x, attention_mask = conditioner(chords_dict, device)
        assert x.shape == (batch_size, 64, 100)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_no_chord_in_batch(self, device, batch_size):
        """バッチ内にno-chordが混在する場合のテスト。"""
        conditioner = CocoMullaChordConditioner(with_bass=True)

        # 各バッチの最後のフレームをno-chordに
        chords_data = torch.zeros((batch_size, 3, 3), dtype=torch.long, device=device)
        chords_data[:, -1, :] = -1  # 最後のフレームをno-chord

        c = conditioner.encode_chords(chords_data, device)

        # 各バッチの最後のフレームがno-chord flagを持つか確認
        for i in range(batch_size):
            assert c[i, -1, 36].item() == 1.0  # NoChord flag
            assert torch.sum(c[i, -1, :36]) == 0.0  # 他の次元は0
