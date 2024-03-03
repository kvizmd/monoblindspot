import torch
from torch import nn

from bse.models.layers import DeformableConv2d


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.actf(x)
        return x


class DCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = DeformableConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.actf(x)
        return x


class GroupNorm2d(nn.GroupNorm):
    def __init__(self, out_channels: int, num_groups: int = 32):
        if out_channels % 32 != 0:
            num_groups = num_groups // 2
        super().__init__(num_groups, out_channels)


class BasicBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            dilation: int = 1,
            norm_func=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes,
            kernel_size=3, stride=stride, padding=dilation,
            bias=False, dilation=dilation)
        self.bn1 = norm_func(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1, padding=dilation,
            bias=False, dilation=dilation)
        self.bn2 = norm_func(planes)
        self.stride = stride

    def forward(
            self,
            x: torch.Tensor,
            residual=None) -> torch.Tensor:
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            dilation: int = 1,
            norm_func=nn.BatchNorm2d):
        super().__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = norm_func(bottle_planes)

        self.conv2 = nn.Conv2d(
            bottle_planes, bottle_planes,
            kernel_size=3, stride=stride, padding=dilation,
            bias=False, dilation=dilation)
        self.bn2 = norm_func(bottle_planes)

        self.conv3 = nn.Conv2d(
            bottle_planes, planes,
            kernel_size=1, bias=False)
        self.bn3 = norm_func(planes)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(
            self,
            x: torch.Tensor,
            residual=None) -> torch.Tensor:
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            dilation: int = 1,
            norm_func=nn.BatchNorm2d):
        super().__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = norm_func(bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes, bottle_planes,
            kernel_size=3, stride=stride, padding=dilation,
            bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = norm_func(bottle_planes)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = norm_func(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(
            self,
            x: torch.Tensor,
            residual=None) -> torch.Tensor:
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            residual: bool,
            norm_func=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride=1,
            bias=False, padding=(kernel_size - 1) // 2)
        self.bn = norm_func(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x) -> torch.Tensor:
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
            self,
            levels: int,
            block: int,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            level_root: bool = False,
            root_dim: int = 0,
            root_kernel_size: int = 1,
            dilation: int = 1,
            root_residual: bool = False,
            norm_func=nn.BatchNorm2d):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels, out_channels,
                stride, dilation=dilation, norm_func=norm_func)
            self.tree2 = block(
                out_channels, out_channels,
                1, dilation=dilation, norm_func=norm_func)
        else:
            self.tree1 = Tree(
                levels - 1, block, in_channels, out_channels,
                stride, root_dim=0, root_kernel_size=root_kernel_size,
                dilation=dilation, root_residual=root_residual,
                norm_func=norm_func)
            self.tree2 = Tree(
                levels - 1, block, out_channels, out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation, root_residual=root_residual,
                norm_func=norm_func)

        if levels == 1:
            self.root = Root(
                root_dim, out_channels,
                root_kernel_size, root_residual, norm_func=norm_func)

        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=1, bias=False),
                norm_func(out_channels)
            )

    def forward(
            self,
            x: torch.Tensor,
            residual=None,
            children=None) -> torch.Tensor:
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x
