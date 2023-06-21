from PIL import Image

import numpy as np
import cv2
import torch


def to_u8(x: np.array) -> np.array:
    if x.ndim == 2:
        H, W = x.shape[-2:]
        x = x.reshape(1, H, W)
    x = x.transpose(1, 2, 0)
    x = (x * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return x


def to_pil(x: np.array) -> Image:
    if x.ndim == 3 and x.shape[-1] == 1:
        return Image.fromarray(x[:, :, 0])
    return Image.fromarray(x)


def to_numpy(x: torch.Tensor) -> np.array:
    return x.detach().cpu().numpy()


def alpha_blend(img, hm, cmap=cv2.COLORMAP_JET):
    """
    Apply alpha blending.

    If the heatmap has already colors, color should be set None.
    """
    H, W = hm.shape[:2]

    if cmap is not None:
        hm = hm.reshape(H, W, 1)
        color_hm = np.uint8(hm * 255)
        color_hm = cv2.applyColorMap(color_hm, cmap)
        color_hm = cv2.cvtColor(color_hm, cv2.COLOR_BGR2RGB)
        a = hm.repeat(3, axis=2)

    else:
        a = hm.max(axis=2, keepdims=True).repeat(3, axis=2) > 0
        color_hm = hm
    out = (1 - a) * img + a * color_hm
    return out.astype(img.dtype)


def pickup_color(score: float, cmap=cv2.COLORMAP_TURBO) -> tuple:
    val = np.full((1, 1), score * 255, dtype=np.uint8)
    color_hm = cv2.applyColorMap(val, cmap)
    color_hm = cv2.cvtColor(color_hm, cv2.COLOR_BGR2RGB)
    color = color_hm[0][0]
    return (int(color[0]), int(color[1]), int(color[2]))


def put_colorized_points(height, width, points, scores=1.0) -> np.array:
    hm = np.zeros((height, width, 3), dtype=np.uint8)
    if len(points) == 0:
        return hm

    if isinstance(scores, (int, float)):
        scores = [scores] * points.shape[0]

    prior_eye_level = height // 2
    r_factor = 0.1
    min_r = max(height // 50, 6)

    idxs = np.argsort([p[0] for p in points])
    for idx in idxs:
        s = scores[idx]
        p = points[idx]

        v = int(p[0] * (height - 1))
        u = int(p[1] * (width - 1))
        delta = v - prior_eye_level
        r = max(r_factor * delta, min_r)

        color = pickup_color(s)
        cv2.circle(hm, (u, v), int(r), color, thickness=-1)
        b_color = (255, 255, 255)
        cv2.circle(hm, (u, v), int(r), b_color, thickness=2)

    return hm
