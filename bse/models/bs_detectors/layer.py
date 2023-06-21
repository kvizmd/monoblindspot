import torch
import torch.nn as nn

from .util import set_prior_prob, normalize_iam


class LinearBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class ReLUBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pad_reflection: bool = True):
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels, pad_reflection)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ELUBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pad_reflection: bool = True):
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels, pad_reflection)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Conv3x3(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pad_reflection: bool = True):
        super().__init__(in_channels, out_channels, kernel_size=3)

        if pad_reflection:
            self.pad1 = nn.ReflectionPad2d(1)
        else:
            self.pad1 = nn.ZeroPad2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.pad1(x))


class ProjectBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: bool = True,
            bias: bool = True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias)
        if activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Indentity()

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


@torch.no_grad()
def compute_coordinates(x: torch.Tensor) -> torch.Tensor:
    H, W = x.shape[-2:]
    y_loc = -1.0 + 2.0 * torch.arange(H) / (H - 1)
    x_loc = -1.0 + 2.0 * torch.arange(W) / (W - 1)
    x_loc, y_loc = torch.meshgrid(x_loc, y_loc, indexing='xy')
    y_loc = y_loc.expand(x.shape[0], 1, -1, -1)
    x_loc = x_loc.expand(x.shape[0], 1, -1, -1)
    locations = torch.cat([y_loc, x_loc], 1)
    return locations.to(x)


class CoordConv(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *args,
            **kwargs):
        super().__init__(in_channels + 2, out_channels, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coord = compute_coordinates(x)
        return super().forward(torch.cat((x, coord), dim=1))


class CoordConv3x3(CoordConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pad_reflection: bool = True):
        super().__init__(in_channels, out_channels, kernel_size=3)

        if pad_reflection:
            self.pad1 = nn.ReflectionPad2d(1)
        else:
            self.pad1 = nn.ZeroPad2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.pad1(x))


class CoordReLUBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pad_reflection: bool = True):
        super().__init__()

        self.conv = CoordConv3x3(in_channels, out_channels, pad_reflection)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class IAM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            inst_num: int = 32,
            prior_prob: float = 0.01,
            group_num: int = 4):
        super().__init__()

        self.iam_conv = nn.Conv2d(
            in_channels, group_num * inst_num,
            kernel_size=3, padding=1, groups=group_num)

        self.inst_num = inst_num
        self.group_num = group_num
        self.prior_prob = prior_prob
        self.out_channels = group_num * in_channels

        self._init_weights()

    def _init_weights(self):
        for m in [self.iam_conv]:
            nn.init.normal_(m.weight, std=0.01)
            set_prior_prob(m, self.prior_prob)

    def forward(self, x: torch.Tensor) -> tuple:
        iam = self.iam_conv(x).sigmoid()

        B, N, H, W = iam.shape
        G = self.group_num
        N = N // G
        iam_map = iam.view(B, G, N, H, W).transpose(1, 2)

        iam = iam.view(B, G * N, -1)
        iam = normalize_iam(iam)

        C = x.shape[1]
        inst_ft = x.view(B, C, -1).permute(0, 2, 1)
        bs_ft = torch.bmm(iam, inst_ft)

        bs_ft = bs_ft.view(B, G, N, C).permute(0, 2, 1, 3).reshape(B, N, -1)

        return bs_ft, iam_map
