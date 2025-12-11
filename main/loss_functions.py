"""損失関数のファクトリーと関連ユーティリティ"""

from typing import Literal

import torch.nn as nn

LossType = Literal["mse", "smooth_l1", "l1", "huber"]


def create_loss_function(
    loss_type: LossType = "mse",
    reduction: str = "mean",
    **kwargs,
) -> nn.Module:
    """損失関数を作成

    Args:
        loss_type: 損失関数の種類
        reduction: 損失の集約方法 ('none', 'mean', 'sum')
        **kwargs: 各損失関数固有のパラメータ
            - smooth_l1: beta (float, default=1.0)
            - huber: delta (float, default=1.0)

    Returns:
        損失関数モジュール

    Examples:
        >>> loss_fn = create_loss_function("mse")
        >>> loss_fn = create_loss_function("smooth_l1", beta=0.5)
        >>> loss_fn = create_loss_function("huber", delta=1.0)
    """
    if loss_type == "mse":
        return nn.MSELoss(reduction=reduction)

    elif loss_type == "smooth_l1":
        beta = kwargs.get("beta", 1.0)
        return nn.SmoothL1Loss(reduction=reduction, beta=beta)

    elif loss_type == "l1":
        return nn.L1Loss(reduction=reduction)

    elif loss_type == "huber":
        delta = kwargs.get("delta", 1.0)
        return nn.HuberLoss(reduction=reduction, delta=delta)

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Supported types: mse, smooth_l1, l1, huber"
        )


class LossConfig:
    """損失関数の設定を保持するデータクラス"""

    def __init__(
        self,
        loss_type: LossType = "mse",
        reduction: str = "mean",
        **kwargs,
    ):
        self.loss_type = loss_type
        self.reduction = reduction
        self.params = kwargs

    def create(self) -> nn.Module:
        """設定から損失関数を作成"""
        return create_loss_function(
            loss_type=self.loss_type,
            reduction=self.reduction,
            **self.params,
        )

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        if params_str:
            return f"LossConfig({self.loss_type}, {self.reduction}, {params_str})"
        return f"LossConfig({self.loss_type}, {self.reduction})"
