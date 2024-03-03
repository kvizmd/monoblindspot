from torch import nn
from torch.nn import functional as F


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

        linear_modes = {
            'linear',
            'bilinear',
            'bicubic',
            'trilinear'
        }
        if self.mode in linear_modes:
            self.align_corners = False
        else:
            self.align_corners = None

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=self.align_corners)
