import torch.nn as nn


class RefrectConv(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            bias: bool = True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=bias)

        padding_num = kernel_size // 2
        if padding_num == 0:
            self.ref_pad = nn.Identity()
        else:
            self.ref_pad = nn.ReflectionPad2d(padding_num)

    def forward(self, x):
        return super().forward(self.ref_pad(x))
