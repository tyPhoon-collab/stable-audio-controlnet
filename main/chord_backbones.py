from abc import ABC, abstractmethod

import torch
from torch import nn


class ChordBackbone(nn.Module, ABC):
    """
    コード表現処理用のバックボーンの基底クラス。

    すべてのBackboneは以下を実装する必要があります：
    - get_input_dim(): 入力次元を返す
    - get_output_dim(): 出力次元を返す
    - forward(): (batch, T, input_dim) -> (batch, T, output_dim)
    """

    @abstractmethod
    def get_input_dim(self) -> int:
        """Backboneの入力次元を返す"""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Backboneの出力次元を返す"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, input_dim)

        Returns:
            (batch, T, output_dim)
        """
        pass


class ChordConvBackbone(ChordBackbone):
    """シンプルな1層Conv1dバックボーン。デフォルトは入力次元128、出力次元32"""

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 32,
        kernel_size: int = 7,
        padding: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 入力射影層は持たない（Conditioner側で処理）
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding),
            nn.SiLU(),
        )

    def get_input_dim(self) -> int:
        """Backboneの入力次元を返す"""
        return self.input_dim

    def get_output_dim(self) -> int:
        """Backboneの出力次元を返す"""
        return self.output_dim

    def forward(self, x):
        # x: (batch, T, input_dim) -> (batch, T, output_dim)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class ChordMlpBackbone(ChordBackbone):
    """軽量なMLP（2層Linear + SiLU）バックボーン。デフォルトは入力次元128、出力次元32"""

    def __init__(self, input_dim: int = 128, output_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

    def get_input_dim(self) -> int:
        """Backboneの入力次元を返す"""
        return self.input_dim

    def get_output_dim(self) -> int:
        """Backboneの出力次元を返す"""
        return self.output_dim

    def forward(self, x):
        # x: (batch, T, input_dim) -> (batch, T, output_dim)
        return self.mlp(x)


class ChordResConvBackbone(ChordBackbone):
    """
    残差結合(Residual Connection)を用いた畳み込みバックボーン。
    より深い層を学習しやすくし、勾配消失を防ぐ。
    """

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 32,
        hidden_dim: int = 128,
        kernel_size: int = 7,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 入力を隠れ層の次元へ射影
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        # Residual Blocks
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2
                    ),
                    nn.SiLU(),
                    nn.Conv1d(
                        hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2
                    ),
                    nn.SiLU(),
                )
                for _ in range(num_layers)
            ]
        )

        # 出力次元へ射影
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, 1)

    def get_input_dim(self) -> int:
        """Backboneの入力次元を返す"""
        return self.input_dim

    def get_output_dim(self) -> int:
        """Backboneの出力次元を返す"""
        return self.output_dim

    def forward(self, x):
        # x: (batch, T, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, T)
        x = self.input_proj(x)

        for layer in self.layers:
            x = x + layer(x)  # Residual Connection

        x = self.output_proj(x)
        x = x.transpose(1, 2)  # (batch, T, output_dim)
        return x


class ChordDilatedConvBackbone(ChordBackbone):
    """
    Dilated Convolutionを用いたバックボーン。
    パラメータ数を増やさずに受容野（Receptive Field）を広げ、より長い文脈を考慮する。
    """

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 32,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.SiLU(),
                )
            )

        self.output_proj = nn.Conv1d(hidden_dim, output_dim, 1)

    def get_input_dim(self) -> int:
        """Backboneの入力次元を返す"""
        return self.input_dim

    def get_output_dim(self) -> int:
        """Backboneの出力次元を返す"""
        return self.output_dim

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)

        for layer in self.layers:
            # Skip connection + Dilated Conv
            x = x + layer(x)

        x = self.output_proj(x)
        x = x.transpose(1, 2)
        return x


class ChordGRUBackbone(ChordBackbone):
    """
    双方向GRUを用いたバックボーン。
    時系列データの前後関係を強力にモデル化する。
    """

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def get_input_dim(self) -> int:
        """Backboneの入力次元を返す"""
        return self.input_dim

    def get_output_dim(self) -> int:
        """Backboneの出力次元を返す"""
        return self.output_dim

    def forward(self, x):
        # x: (batch, T, input_dim)
        x, _ = self.gru(x)
        x = self.proj(x)
        return x
