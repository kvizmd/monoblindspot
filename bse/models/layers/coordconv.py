import torch
from torch import nn

from .util import compute_coordinates


class CoordConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels + 2, out_channels, *args, **kwargs)

    def forward(self, x):
        coord = compute_coordinates(x)
        return super().forward(torch.cat((x, coord), dim=1))
