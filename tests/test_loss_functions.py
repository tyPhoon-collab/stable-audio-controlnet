"""損失関数のテスト"""

import pytest
import torch

from main.loss_functions import LossConfig, create_loss_function


class TestCreateLossFunction:
    """create_loss_function のテスト"""

    def test_mse_loss(self):
        """MSELossの作成と動作確認"""
        loss_fn = create_loss_function("mse")
        output = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(output, target)
        assert loss.item() > 0
        assert loss.dim() == 0  # スカラー

    def test_smooth_l1_loss_default(self):
        """SmoothL1Lossのデフォルト設定での作成"""
        loss_fn = create_loss_function("smooth_l1")
        output = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(output, target)
        assert loss.item() > 0

    def test_smooth_l1_loss_with_beta(self):
        """SmoothL1Lossのbetaパラメータ指定"""
        loss_fn = create_loss_function("smooth_l1", beta=0.5)
        output = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(output, target)
        assert loss.item() > 0

    def test_l1_loss(self):
        """L1Lossの作成と動作確認"""
        loss_fn = create_loss_function("l1")
        output = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(output, target)
        assert abs(loss.item() - 0.5) < 1e-5

    def test_huber_loss_default(self):
        """HuberLossのデフォルト設定での作成"""
        loss_fn = create_loss_function("huber")
        output = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(output, target)
        assert loss.item() > 0

    def test_huber_loss_with_delta(self):
        """HuberLossのdeltaパラメータ指定"""
        loss_fn = create_loss_function("huber", delta=2.0)
        output = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(output, target)
        assert loss.item() > 0

    def test_reduction_none(self):
        """reduction='none'の動作確認"""
        loss_fn = create_loss_function("mse", reduction="none")
        output = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(output, target)
        assert loss.shape == output.shape

    def test_reduction_sum(self):
        """reduction='sum'の動作確認"""
        loss_fn = create_loss_function("mse", reduction="sum")
        output = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(output, target)
        assert loss.dim() == 0

    def test_invalid_loss_type(self):
        """無効な損失関数タイプでエラー"""
        with pytest.raises(ValueError, match="Unknown loss type"):
            create_loss_function("invalid_loss")


class TestLossConfig:
    """LossConfig のテスト"""

    def test_mse_config(self):
        """MSE設定の作成"""
        config = LossConfig("mse")
        loss_fn = config.create()
        assert loss_fn is not None

    def test_smooth_l1_config_with_params(self):
        """SmoothL1設定とパラメータ"""
        config = LossConfig("smooth_l1", beta=0.5)
        loss_fn = config.create()
        output = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.5, 2.5])
        loss = loss_fn(output, target)
        assert loss.item() > 0

    def test_config_repr(self):
        """LossConfigの文字列表現"""
        config = LossConfig("smooth_l1", reduction="mean", beta=0.5)
        repr_str = repr(config)
        assert "smooth_l1" in repr_str
        assert "mean" in repr_str
        assert "beta=0.5" in repr_str


class TestLossFunctionComparison:
    """損失関数の挙動比較テスト"""

    def test_mse_vs_l1_for_small_errors(self):
        """小さな誤差でのMSEとL1の比較"""
        output = torch.tensor([1.0, 1.1, 1.2])
        target = torch.tensor([1.0, 1.0, 1.0])

        mse_loss = create_loss_function("mse")
        l1_loss = create_loss_function("l1")

        mse_val = mse_loss(output, target).item()
        l1_val = l1_loss(output, target).item()

        # 小さな誤差ではMSEの方が小さくなる傾向
        assert mse_val < l1_val

    def test_mse_vs_l1_for_large_errors(self):
        """大きな誤差でのMSEとL1の比較"""
        output = torch.tensor([1.0, 5.0, 10.0])
        target = torch.tensor([1.0, 1.0, 1.0])

        mse_loss = create_loss_function("mse")
        l1_loss = create_loss_function("l1")

        mse_val = mse_loss(output, target).item()
        l1_val = l1_loss(output, target).item()

        # 大きな誤差ではMSEの方が大きくなる（外れ値に敏感）
        assert mse_val > l1_val

    def test_smooth_l1_behavior(self):
        """SmoothL1Lossの振る舞い確認"""
        output = torch.tensor([0.0, 2.0, 10.0])
        target = torch.tensor([0.0, 0.0, 0.0])

        # beta=1.0の場合、誤差が1以下では二乗、1以上では線形
        loss_fn = create_loss_function("smooth_l1", reduction="none", beta=1.0)
        loss = loss_fn(output, target)

        # 誤差0の場合は損失0
        assert loss[0].item() == 0.0
        # 誤差2の場合は線形領域
        # 誤差10の場合も線形領域
        assert loss[2].item() > loss[1].item()
