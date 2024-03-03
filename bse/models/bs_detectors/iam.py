import torch
import torch.nn as nn

from bse.models.layers import \
    ReLUBlock, \
    CoordConv, \
    IAM, \
    FocalLinear, \
    XavierLinear


class IAMDetector(nn.Module):
    """
    Single head version of SparseInst.
    """

    def __init__(
            self,
            num_ch_enc: list,
            head_channels: int = 256,
            inst_num: int = 32,
            prior_prob: float = 0.01,
            group_num: int = 4,
            **kwargs):
        super().__init__()

        in_channels = num_ch_enc[-1]

        self.inst_head = nn.Sequential(
            ReLUBlock(in_channels, head_channels, conv_layer=CoordConv),
            ReLUBlock(head_channels, head_channels),
            ReLUBlock(head_channels, head_channels),
            ReLUBlock(head_channels, head_channels))

        self.iam = IAM(
            head_channels,
            inst_num=inst_num,
            prior_prob=prior_prob,
            group_num=group_num)
        iam_channels = self.iam.out_channels

        self.cls = nn.Sequential(
            FocalLinear(iam_channels, 1, prior_prob=prior_prob),
            nn.Sigmoid())

        self.loc = nn.Sequential(
            XavierLinear(iam_channels, 2),
            nn.Sigmoid())

        self.inst_num = inst_num

    def forward(self, features: list) -> dict:
        outputs = {}

        if not isinstance(features, torch.Tensor):
            ft = features[-1]
        else:
            ft = features

        inst_ft = self.inst_head(ft)
        bs_ft = self.iam(inst_ft)

        B = bs_ft.shape[0]

        outputs['bs_point', 0] = self.loc(bs_ft).view(B, -1, 2)
        outputs['bs_confidence', 0] = self.cls(bs_ft).view(B, -1)

        return outputs
