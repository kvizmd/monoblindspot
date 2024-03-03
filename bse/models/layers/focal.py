import torch
from torch import nn

from .util import set_prior_prob


class FocalConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            groups: int = 1,
            prior_prob: float = 0.01):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups)

        nn.init.normal_(self.conv.weight, std=0.01)
        set_prior_prob(self.conv, prior_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FocalLinear(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            prior_prob: float = 0.01):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)

        nn.init.normal_(self.linear.weight, std=0.01)
        set_prior_prob(self.linear, prior_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
