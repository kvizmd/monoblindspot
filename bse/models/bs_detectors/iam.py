import torch
import torch.nn as nn

from .util import set_prior_prob, _sigmoid
from .layer import ReLUBlock, CoordReLUBlock, IAM


class IAMDetector(nn.Module):
    """
    Single head version of SparseInst.
    """

    def __init__(
            self,
            in_channels: int,
            head_channels: int = 256,
            inst_num: int = 32,
            prior_prob: float = 0.01,
            group_num: int = 4,
            **kwargs):
        super().__init__()

        self.inst_head = nn.Sequential(
            CoordReLUBlock(in_channels, head_channels),
            ReLUBlock(head_channels, head_channels),
            ReLUBlock(head_channels, head_channels),
            ReLUBlock(head_channels, head_channels))

        self.iam = IAM(
            head_channels,
            inst_num=inst_num,
            prior_prob=prior_prob,
            group_num=group_num)
        iam_channels = self.iam.out_channels

        self.cls = nn.Linear(iam_channels, 1)
        self.loc = nn.Linear(iam_channels, 2)

        self.inst_num = inst_num
        self.prior_prob = prior_prob

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls]:
            nn.init.normal_(m.weight, std=0.01)
            set_prior_prob(m, self.prior_prob)

        for m in [self.loc]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features: list) -> dict:
        outputs = {}

        if not isinstance(features, torch.Tensor):
            ft = features[-1]
        else:
            ft = features

        inst_ft = self.inst_head(ft)
        bs_ft, iam_map = self.iam(inst_ft)

        B = bs_ft.shape[0]

        # Normalize coordinates with height and width of the input image.
        loc = self.loc(bs_ft).view(B, -1, 2)
        outputs['bs_point', 0] = _sigmoid(loc)

        cls_logit = self.cls(bs_ft).view(B, -1)
        outputs['bs_confidence', 0] = cls_logit.sigmoid()

        return outputs
