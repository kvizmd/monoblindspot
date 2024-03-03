import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channel_list: list):
        super().__init__()
        num_hidden_layers = len(mid_channel_list)
        assert num_hidden_layers > 0

        self.encode = nn.Conv2d(
            in_channels, mid_channel_list[0], kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self._init_conv_weights(self.encode)

        if num_hidden_layers < 2:
            self.convs = nn.Identity()
        else:
            self.convs = []
            for i in range(1, num_hidden_layers):
                conv = nn.Conv2d(
                    mid_channel_list[i - 1],
                    mid_channel_list[i], kernel_size=1)
                self._init_conv_weights(conv)
                self.convs += [conv, nn.ReLU(inplace=True)]
            self.convs = nn.Sequential(*self.convs)

        self.out = nn.Conv2d(
            mid_channel_list[-1], out_channels, kernel_size=1)
        self.out_dim = out_channels

    def _init_conv_weights(self, conv: nn.Module):
        nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
        nn.init.constant_(conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.encode(x))
        x = self.convs(x)
        x = self.out(x)
        return x
