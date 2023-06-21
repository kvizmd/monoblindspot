import numpy as np

from torch import nn

from .util import fill_up_weights


class IDAUp(nn.Module):
    def __init__(
            self,
            o: int,
            channels: list,
            up_f: list,
            conv_func=nn.Conv2d):
        super().__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = conv_func(c, o)
            node = conv_func(o, o)

            up = nn.ConvTranspose2d(
                o, o,
                f * 2, stride=f, padding=f // 2,
                output_padding=0, groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers: list, startp: int, endp: int):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(
            self,
            startp: int,
            channels: list,
            scales: list,
            in_channels=None,
            conv_func=nn.Conv2d):
        super().__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            ida_up = IDAUp(
                channels[j], in_channels[j:],
                scales[j:] // scales[j], conv_func=conv_func)
            setattr(self, 'ida_{}'.format(i), ida_up)
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers: list) -> list:
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out
