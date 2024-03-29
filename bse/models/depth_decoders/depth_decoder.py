import torch
import torch.nn as nn

from bse.models.layers import Upsample, RefrectConv


class ELUBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = RefrectConv(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthDecoder(nn.Module):
    def __init__(
            self,
            num_ch_enc: list,
            scales: list = list(range(4))):
        super().__init__()
        self.scales = scales
        e1, e2, e3, e4, e5 = num_ch_enc[-5:]
        d1, d2, d3, d4, d5 = 16, 32, 64, 128, 256

        self.conv1 = ELUBlock(e5, d5)
        self.node1 = ELUBlock(d5 + e4, d5)

        self.conv2 = ELUBlock(d5, d4)
        self.node2 = ELUBlock(d4 + e3, d4)

        self.conv3 = ELUBlock(d4, d3)
        self.node3 = ELUBlock(d3 + e2, d3)

        self.conv4 = ELUBlock(d3, d2)
        self.node4 = ELUBlock(d2 + e1, d2)

        self.conv5 = ELUBlock(d2, d1)
        self.node5 = ELUBlock(d1, d1)

        self.disp1 = nn.Sequential(RefrectConv(d4, 1), nn.Sigmoid())
        self.disp2 = nn.Sequential(RefrectConv(d3, 1), nn.Sigmoid())
        self.disp3 = nn.Sequential(RefrectConv(d2, 1), nn.Sigmoid())
        self.disp4 = nn.Sequential(RefrectConv(d1, 1), nn.Sigmoid())

        self.upsample = Upsample(scale_factor=2, mode='nearest')

    def forward(self, input_features: list) -> dict:
        outputs = {}
        e1, e2, e3, e4, e5 = input_features[-5:]

        x = self.conv1(e5)
        x = self.upsample(x)
        x = self.node1(torch.cat((x, e4), dim=1))

        x = self.conv2(x)
        x = self.upsample(x)
        x = self.node2(torch.cat((x, e3), dim=1))
        outputs['disp', 3] = self.disp1(x)

        x = self.conv3(x)
        x = self.upsample(x)
        x = self.node3(torch.cat((x, e2), dim=1))
        outputs['disp', 2] = self.disp2(x)

        x = self.conv4(x)
        x = self.upsample(x)
        x = self.node4(torch.cat((x, e1), dim=1))
        outputs['disp', 1] = self.disp3(x)

        x = self.conv5(x)
        x = self.upsample(x)
        x = self.node5(x)
        outputs['disp', 0] = self.disp4(x)

        return outputs
