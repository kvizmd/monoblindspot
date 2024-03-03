import math

import numpy as np

import torch
from torch.nn import functional as F

from bse.utils.depth import disp_to_depth


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


class DepthScaleEstimator(object):
    def __init__(
            self,
            min_rel_depth: float = 1e-3,
            max_rel_depth: float = 100,
            min_metric_depth: float = 1e-3,
            max_metric_depth: float = 80,
            scaling_ransac_iters: int = 1000,
            scaling_ransac_thr: float = 0.01):

        self.min_rel_depth = min_rel_depth
        self.max_rel_depth = max_rel_depth
        self.min_metric_depth = min_metric_depth
        self.max_metric_depth = max_metric_depth
        self.scaling_ransac_iters = scaling_ransac_iters
        self.scaling_ransac_thr = scaling_ransac_thr  # Abs-Rel

    def __call__(
            self,
            disp: torch.Tensor,
            gt_depth: torch.Tensor) -> torch.Tensor:
        gt_height, gt_width = gt_depth.shape[-2:]

        # Convert inversed depth to depth
        scaled_disp, _ = disp_to_depth(
            disp, self.min_rel_depth, self.max_rel_depth)

        resized_scaled_disp = F.interpolate(
            scaled_disp, (gt_height, gt_width),
            mode='bilinear', align_corners=False)
        resized_depth = 1 / resized_scaled_disp  # GT resolution for mask

        # Masking
        range_mask = torch.logical_and(
            gt_depth > self.min_metric_depth,
            gt_depth < self.max_metric_depth)

        # Apply garg/eigen crop
        crop_mask = torch.zeros_like(range_mask)
        cy1 = int(0.40810811 * gt_height)
        cy2 = int(0.99189189 * gt_height)
        cx1 = int(0.03594771 * gt_width)
        cx2 = int(0.96405229 * gt_width)
        crop_mask[:, :, cy1:cy2, cx1:cx2] = True

        mask = torch.logical_and(range_mask, crop_mask)

        ratios = []
        for m, p, g in zip(mask, resized_depth, gt_depth):
            # ratio = g[m].median() / p[m].median()  # Median scaling
            # ratio = (g[m] * p[m]).sum() / (p[m] ** 2).sum()

            # RANSAC
            x = p[m].view(-1)
            y = g[m].view(-1)
            size = x.shape[0]

            if size > 0:
                indices = torch.randint(
                    size, size=(self.scaling_ransac_iters, 1), device=x.device)
                model = y[indices] / x[indices]

                error = (y - model * x).abs() / y  # abs rel
                score = (error < self.scaling_ransac_thr).sum(dim=1)

                best_idx = score.argmax()
                ratio = model[best_idx]

                ratios.append(ratio)
            else:
                ratios.append(torch.ones((1,), dtype=x.dtype, device=x.device))

        return torch.stack(ratios)
