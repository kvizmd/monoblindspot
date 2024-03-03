import torch
from torch import nn


class XavierConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            bias: bool = True,
            prior_prob: float = 0.01):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias)

        nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class XavierLinear(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            prior_prob: float = 0.01):
        super().__init__()

        self.linear = nn.Linear(
            in_channels, out_channels, bias=bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
