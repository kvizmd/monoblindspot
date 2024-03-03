import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.act(self.linear(x))


class ReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pad_reflection=False):
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels, pad_reflection)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class ELUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pad_reflection=True):
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels, pad_reflection)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class Conv3x3(nn.Conv2d):
    def __init__(self, in_channels, out_channels, pad_reflection=True):
        super().__init__(in_channels, out_channels, kernel_size=3)

        if pad_reflection:
            self.pad1 = nn.ReflectionPad2d(1)
        else:
            self.pad1 = nn.ZeroPad2d(1)

    def forward(self, x):
        return super().forward(self.pad1(x))


class ProjectBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, bias=True):
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

    def forward(self, x):
        return self.act(self.conv(x))


@torch.no_grad()
def compute_coordinates(x):
    H, W = x.shape[-2:]
    y_loc = -1.0 + 2.0 * torch.arange(H) / (H - 1)
    x_loc = -1.0 + 2.0 * torch.arange(W) / (W - 1)
    x_loc, y_loc = torch.meshgrid(x_loc, y_loc, indexing='xy')
    y_loc = y_loc.expand(x.shape[0], 1, -1, -1)
    x_loc = x_loc.expand(x.shape[0], 1, -1, -1)
    locations = torch.cat([y_loc, x_loc], 1)
    return locations.to(x)


class CoordConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels + 2, out_channels, *args, **kwargs)

    def forward(self, x):
        coord = compute_coordinates(x)
        return super().forward(torch.cat((x, coord), dim=1))


class CoordConv3x3(CoordConv):
    def __init__(self, in_channels, out_channels, pad_reflection=True):
        super().__init__(in_channels, out_channels, kernel_size=3)

        if pad_reflection:
            self.pad1 = nn.ReflectionPad2d(1)
        else:
            self.pad1 = nn.ZeroPad2d(1)

    def forward(self, x):
        return super().forward(self.pad1(x))


class CoordReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pad_reflection=True):
        super().__init__()

        self.conv = CoordConv3x3(in_channels, out_channels, pad_reflection)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))
