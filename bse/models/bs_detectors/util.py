import math

import torch


def set_prior_prob(
        module: torch.nn.Module,
        prior_prob: float = 0.01):
    module.bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))


def _sigmoid(
        x: torch.Tensor,
        eps: float = 1e-4) -> torch.Tensor:
    return x.sigmoid().clamp(min=eps, max=1 - eps)


def logit(
        p: torch.Tensor,
        epx: float = 1e-10) -> torch.Tensor:
    return torch.log(p / (1 - p).clamp(min=1e-10))


def normalize_iam(x: torch.Tensor) -> torch.Tensor:
    normalizer = x.sum(-1, keepdims=True).clamp(min=1e-6)
    return x / normalizer
