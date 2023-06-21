import numpy as np
import torch
from torch import nn

from .backbone import dla34, dla60, dla102, dla169
from .up import DLAUp, IDAUp
from .layer import GroupNorm2d, DCNBlock, ConvBlock


class DLAEncoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            pretrained: bool = True,
            down_ratio: int = 4,
            last_level: int = 5,
            out_channel: int = 0,
            group_norm: bool = False,
            deform_conv: bool = False,
            **kwargs):
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]

        if group_norm:
            norm_func = GroupNorm2d
        else:
            norm_func = nn.BatchNorm2d

        if deform_conv:
            conv_func = DCNBlock
        else:
            conv_func = ConvBlock

        self.base = {
            34: dla34,
            60: dla60,
            102: dla102,
            169: dla169
        }[num_layers](pretrained=pretrained, norm_func=norm_func)

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
                self.first_level,
                channels[self.first_level:],
                scales,
                conv_func=conv_func)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
                out_channel,
                channels[self.first_level:self.last_level],
                [2 ** i for i in range(self.last_level - self.first_level)],
                conv_func=conv_func)

        self.out_channels = out_channel

    def forward(self, x: torch.Tensor) -> list:
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return y
