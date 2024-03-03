import torch
import torch.nn as nn

from .focal import FocalConv


def normalize_iam(x: torch.Tensor) -> torch.Tensor:
    normalizer = x.sum(-1, keepdims=True).clamp(min=1e-6)
    return x / normalizer


class IAM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            inst_num: int = 32,
            prior_prob: float = 0.01,
            group_num: int = 4):
        super().__init__()

        self.iam_conv = FocalConv(
            in_channels, group_num * inst_num,
            kernel_size=3, groups=group_num)

        self.inst_num = inst_num
        self.group_num = group_num
        self.prior_prob = prior_prob
        self.out_channels = group_num * in_channels

    def forward(self, x):
        iam = self.iam_conv(x).sigmoid()

        B, N, H, W = iam.shape
        G = self.group_num
        N = N // G
        # iam_map = iam.view(B, G, N, H, W).transpose(1, 2)

        iam = iam.view(B, G * N, -1)
        iam = normalize_iam(iam)

        C = x.shape[1]
        inst_ft = x.view(B, C, -1).permute(0, 2, 1)
        bs_ft = torch.bmm(iam, inst_ft)

        bs_ft = bs_ft.view(B, G, N, C).permute(0, 2, 1, 3).reshape(B, N, -1)

        return bs_ft
