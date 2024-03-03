import numpy as np

import torch
from torch.nn import functional as F


def overlay_at_center(
        base: torch.Tensor,
        kernel: torch.Tensor,
        cy: int,
        cx: int) -> torch.Tensor:
    if kernel.dim() == 1:
        kh = 1
        kw = kernel.shape[0]
    else:
        kh, kw = kernel.shape[-2:]
    ry, rx = kh // 2, kw // 2

    cx, cy = int(cx), int(cy)

    top, bottom = int(cy - ry), int(cy + ry)
    left, right = int(cx - rx), int(cx + rx)

    if base.dim() == 1:
        h = 1
        w = base.shape[0]
    else:
        h, w = base.shape[-2:]

    if left < 0:
        delta = -left
        left = 0
        kernel = kernel[..., delta:]

    if right >= w:
        delta = right - w + 1
        right = w - 1
        kernel = kernel[..., :-delta]

    if top < 0:
        delta = -top
        top = 0
        kernel = kernel[..., delta:, :]

    if bottom >= h:
        delta = bottom - h + 1
        bottom = h - 1
        kernel = kernel[..., :-delta, :]

    sy, ey = top, bottom + 1
    sx, ex = left, right + 1

    if base.dim() == 1:
        base[sx:ex] = torch.maximum(base[sx:ex], kernel)

    else:
        base[..., sy:ey, sx:ex] = \
            torch.maximum(base[..., sy:ey, sx:ex], kernel)
    return base


def generate_gaussian_1d(
        radius: int,
        sigma: float = 1.0,
        eps: float = 1e-10,
        normalize=False,
        centralize=True) -> np.ndarray:
    d = 1 if centralize else 0
    dx = np.arange(-radius, radius + d)
    kernel = np.exp(-(dx * dx) / (2 * sigma * sigma + eps))
    if normalize:
        kernel /= sigma * (2 * np.pi) ** 0.5
    return kernel


def generate_gaussian_2d(
        radius: int,
        sigma: float = 1.0,
        eps: float = 1e-10,
        normalize=False,
        centralize=True,
        dtype=torch.float32,
        device=None) -> torch.Tensor:
    radius = int(radius)
    d = 1 if centralize else 0
    dy, dx = torch.meshgrid(
        torch.arange(-radius, radius + d, dtype=dtype, device=device),
        torch.arange(-radius, radius + d, dtype=dtype, device=device),
        indexing='ij')

    kernel = torch.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma + eps))
    if normalize:
        kernel /= 2 * np.pi * sigma * sigma
    return kernel


def create_heatmap(
        base,
        point,
        r,
        peak_val=1.0,
        sigma: float = None,
        eps: float = 1e-10):
    p = int(point[0]), int(point[1])
    if int(r) == 0:
        return base

    H, W = base.shape[-2:]
    if p[0] < 0 or p[1] < 0 or p[0] >= H or p[1] >= W:
        return base

    base[..., p[0], p[1]] = float(peak_val)
    if sigma is None:
        sigma = float(r) / 3

    is_tensor = isinstance(base, torch.Tensor)
    if not is_tensor:
        base = torch.tensor(base)

    kernel = peak_val * generate_gaussian_2d(
        int(r), sigma=sigma, eps=eps,
        dtype=base.dtype, device=base.device)

    hm = overlay_at_center(base, kernel, *p)
    return hm if is_tensor else hm.detach().cpu().numpy()


def create_vp_offset_map(
        vanish_point_y,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu')) -> torch.Tensor:

    height, width = shape[-2:]
    y_vp = int(vanish_point_y)

    num_under_vanish = height - y_vp

    offset_line = torch.zeros(height, dtype=dtype, device=device)
    offset_line[y_vp:] = torch.tensor(list(range(0, num_under_vanish)))

    offset_map = offset_line[None, :, None].repeat(1, 1, width)

    return offset_map


def create_vp_relative_depth(
        vanish_point: tuple,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu')) -> torch.Tensor:
    offset_map = create_vp_offset_map(
            vanish_point,
            shape,
            dtype=dtype,
            device=device)
    return offset_map / offset_map[0, -1, 0]


# Non-maximum-suppression for heatmap
def heatmap_nms(
        indices: torch.Tensor,
        scores: torch.Tensor,
        radii: torch.Tensor) -> tuple:

    if indices.numel() == 0:
        return indices, scores, radii

    pool = list(range(indices.shape[0]))
    nolap = []
    while len(pool) > 0:
        # sampling a point having the maximum score
        max_i = scores[torch.as_tensor(pool)].argmax()

        target_idx = pool.pop(max_i)
        nolap.append(target_idx)

        target = indices[target_idx]
        radius = radii[target_idx]

        alive = []
        for idx in pool:
            pt = indices[idx]
            d = (target - pt).pow(2).sum().sqrt()

            if d > radius:
                alive.append(idx)

        pool = alive
    return sorted(nolap)


def extract_heatmap_peak(heatmap, kernel_size=3, threshold=0.5):
    """
    Extract the peak of heatmap via the max-pooling.

    Args:
      heatmap: the Tensor object of the shape (B, C, H, W).
      kernel_size: the kernel size for extraction.
    """

    expand_heatmap = F.max_pool2d(
        heatmap,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2)

    peak_mask = expand_heatmap == heatmap
    thres_mask = heatmap >= threshold

    inds = (peak_mask & thres_mask).nonzero()
    vals = \
        heatmap[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

    peak_indices = [[] for _ in range(heatmap.size(0))]
    peak_values = [[] for _ in range(heatmap.size(0))]
    for index, value in zip(inds, vals):
        peak_indices[index[0]].append(index[1:])
        peak_values[index[0]].append(value)

    for i, (indices, values) in enumerate(
            zip(peak_indices, peak_values)):
        if len(indices) == 0:
            peak_indices[i] = torch.tensor([], device=heatmap.device)
            peak_values[i] = torch.tensor([], device=heatmap.device)
        else:
            peak_indices[i] = torch.stack(indices)
            peak_values[i] = torch.stack(values)

    return peak_values, peak_indices
