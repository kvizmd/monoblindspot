import torch
from torch import nn
import torch.nn.functional as F


GAUSSIAN_3x3_FILTER = [
    [1 / 16, 2 / 16, 1 / 16],
    [2 / 16, 4 / 16, 2 / 16],
    [1 / 16, 2 / 16, 1 / 16]
]


def gaussian_3x3(x):
    kernel = torch.tensor(
        GAUSSIAN_3x3_FILTER,
        dtype=x.dtype, device=x.device,
        requires_grad=False).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, stride=1, padding=1)


class GaussianConv2d(nn.Module):
    def __init__(self, sigma=1, trunc=2, kernel_weight=None):
        """
        from https://github.com/tom-roddick/oft
        """
        super().__init__()
        width = round(trunc * sigma)
        x = torch.arange(-width, width + 1, dtype=torch.float32) / sigma
        kernel1d = torch.exp(-0.5 * x ** 2)
        kernel2d = kernel1d.view(1, -1) * kernel1d.view(-1, 1)

        if kernel_weight is not None:
            kernel2d *= kernel_weight

        kernel = kernel2d / kernel2d.sum()

        kernel_size = kernel.shape[-1]
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.padding = int((kernel_size - 1) / 2)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.padding)
