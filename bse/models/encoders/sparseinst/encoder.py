import torch
import torch.nn as nn

from ..resnet import ResNetEncoder
from .ctx import ContextExtractor


class SparseInstEncoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            pretrained: bool = True,
            ppm_channels: int = 256,
            down_ratio: int = 8,
            **kwargs):
        super().__init__()

        assert down_ratio in [8]

        self.base = ResNetEncoder(num_layers, pretrained=pretrained)
        self.extractor = ContextExtractor(self.base.num_ch_enc, ppm_channels)
        self.out_channels = ppm_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.extractor(self.base(x))
        return y
