import torch
from torch import nn


class LateralBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            group_num: int = 16):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.GroupNorm(16, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = nn.Conv2d(
            inplanes, planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(16, planes)

        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = self.relu(out)
        return out
