import math


import torch
from torch import nn


def set_prior_prob(
        module: nn.Module,
        prior_prob: float = 0.01):
    module.bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))


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
