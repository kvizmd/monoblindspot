import torch
import torch.nn as nn
from torch.nn import functional as F

from .layer import PyramidPoolingModule


class ContextExtractor(nn.Module):
    def __init__(
            self,
            num_ch_enc: list,
            channels: int = 256):
        super().__init__()
        C3, C4, C5 = num_ch_enc[-3:]
        self.out_channels = channels

        self.proj_1 = nn.Conv2d(C3, channels, kernel_size=1)
        self.proj_2 = nn.Conv2d(C4, channels, kernel_size=1)
        self.proj_3 = nn.Conv2d(C5, channels, kernel_size=1)

        self.merge_1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1)
        self.merge_2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1)
        self.merge_3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1)

        self.ppm = PyramidPoolingModule(channels, channels=channels // 4)

        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.proj_1, self.proj_2, self.proj_3]:
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)

        for m in [self.merge_1, self.merge_2, self.merge_3]:
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)

        nn.init.kaiming_uniform_(self.fusion.weight, nonlinearity='relu')
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, features: list) -> torch.Tensor:
        f1, f2, f3, f4, f5 = features

        h5 = self.ppm(self.proj_3(f5))
        # h5 = self.proj_3(f5)

        h = F.interpolate(h5, size=f4.shape[-2:], mode='nearest')
        h4 = self.proj_2(f4) + h

        h = F.interpolate(h4, size=f3.shape[-2:], mode='nearest')
        h3 = self.proj_1(f3) + h

        features = [
            self.merge_1(h3),
            F.interpolate(
                self.merge_2(h4), h3.shape[-2:],
                mode='bilinear', align_corners=False),
            F.interpolate(
                self.merge_3(h5), h3.shape[-2:],
                mode='bilinear', align_corners=False),
        ]

        out = self.fusion(torch.cat(features, dim=1))
        return out
