import torch
import torch.nn as nn
from torch.nn import functional as F


class PoolingModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            output_size: tuple = (1, 1)):
        super().__init__()
        self.prior = nn.AdaptiveAvgPool2d(output_size=output_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(self.prior(x)))


class PyramidPoolingModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            channels: int = 512):
        super().__init__()
        self.stage_1 = PoolingModule(in_channels, channels, (1, 1))
        self.stage_2 = PoolingModule(in_channels, channels, (2, 2))
        self.stage_3 = PoolingModule(in_channels, channels, (3, 3))
        self.stage_4 = PoolingModule(in_channels, channels, (6, 6))

        self.bottleneck = nn.Conv2d(
            in_channels + 4 * channels, in_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.bottleneck.weight, nonlinearity='relu')
        nn.init.constant_(self.bottleneck.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        f1 = self.stage_1(x)
        f2 = self.stage_2(x)
        f3 = self.stage_3(x)
        f4 = self.stage_4(x)

        features = [x]
        for f in [f4, f3, f2, f1]:
            f = F.interpolate(
                f, size=(H, W), mode='bilinear', align_corners=False)
            features.append(f)

        out = self.act(self.bottleneck(torch.cat(features, 1)))
        return out
