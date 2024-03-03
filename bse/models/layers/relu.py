import torch.nn as nn


class ReLUBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: int = None,
            stride: int = 1,
            bias: bool = True,
            norm_layer: nn.Module = nn.Identity,
            conv_layer: nn.Module = nn.Conv2d):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = conv_layer(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ReLULinearBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True):
        super().__init__()

        self.linear = nn.Linear(
            in_channels, out_channels, bias=bias)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.act(self.linear(x))
