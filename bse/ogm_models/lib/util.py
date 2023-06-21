import math

import numpy as np

import torch


def mathlib(x):
    if isinstance(x, float):
        return math
    elif isinstance(x, np.ndarray):
        return np
    elif isinstance(x, torch.Tensor):
        return torch
    raise NotImplementedError()


def prob2logodds(p):
    odds = p / (1 - p + 1e-10)
    lib = mathlib(odds)
    return lib.log(odds)


def logodds2prob(logodds):
    lib = mathlib(logodds)
    neg_odds = lib.exp(-logodds)
    return 1 / (1 + neg_odds)


def convert_to_sampler(
        img_points: torch.Tensor,
        height: int,
        width: int) -> tuple:
    """
    Convert the B x 3 x N image points for the sampling indices
    The points order is (x, y, ....).
    """
    img_range_mask = \
        (img_points[:, 0] >= 0) \
        & (img_points[:, 1] >= 0) \
        & (img_points[:, 0] <= width - 1) \
        & (img_points[:, 1] <= height - 1)

    sampling_y = img_points[:, 1].clamp(0, height - 1).long()
    sampling_x = img_points[:, 0].clamp(0, width - 1).long()
    sampling_indices = sampling_y * width + sampling_x

    return sampling_indices, img_range_mask


def batch_masked_sampling(
        src_tensor: torch.Tensor,
        indices: torch.Tensor,
        mask_of_indices: torch.Tensor,
        empty_val: float = 0.0) -> torch.Tensor:
    sampled = torch.full_like(
        indices, empty_val,
        dtype=src_tensor.dtype, device=src_tensor.device)

    for b in range(indices.shape[0]):
        sampled[b][mask_of_indices[b]] = torch.gather(
            src_tensor[b], 0, indices[b][mask_of_indices[b]])

    return sampled


def normalize_with_mean(disp: torch.Tensor) -> tuple:
    mean_scale = 1 / disp.mean((1, 2, 3)).view(-1, 1, 1, 1)
    norm_disp = disp * mean_scale
    return norm_disp, mean_scale
