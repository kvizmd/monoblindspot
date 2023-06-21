from PIL import Image, ImageOps
import contextlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from bse.utils.projector import \
    project_to_2d, \
    transform_point_cloud
from bse.utils.figure import \
    to_numpy, \
    to_pil, \
    to_u8, \
    put_colorized_points, \
    alpha_blend

DPI = 100
# DPI = 300  # High Resolution


@contextlib.contextmanager
def make_depth_figure(inputs, outputs) -> plt.Figure:
    fig = plt.figure(figsize=(12, 5), tight_layout=True, dpi=DPI)

    color = to_numpy(inputs['color', 0, 0][0])
    disp = to_numpy(outputs['disp', 0, 0][0])
    height, width = color.shape[-2:]

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_axis_off()
    ax1.set_title('Input Image')
    ax1.imshow(to_u8(color))

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_axis_off()
    ax2.set_title('Inversed Depth')
    disp = to_pil(to_u8(disp))
    disp = disp.resize((width, height), Image.BILINEAR)
    ax2.imshow(disp, cmap='magma')

    try:
        yield fig
    finally:
        plt.clf()
        plt.close()


@contextlib.contextmanager
def make_bs_figure(inputs, outputs) -> plt.Figure:
    color = to_numpy(inputs['color', 0, 0][0])
    pred_points = to_numpy(outputs['bs_point', 0, 0][0])
    pred_scores = to_numpy(outputs['bs_confidence', 0, 0][0])

    bs_gt = to_numpy(inputs['bs_gt'][0])
    if bs_gt.shape[-1] == 3:
        # Generated labels
        gt_points = bs_gt[:, :2]
        gt_scores = bs_gt[:, 2]
    else:
        # Annotated labels
        gt_points = bs_gt
        gt_scores = np.ones_like(bs_gt[:, 0])

    height, width = color.shape[-2:]
    color = to_u8(color)

    gt_hm = put_colorized_points(height, width, gt_points, gt_scores)
    gt_color = alpha_blend(color, gt_hm, None)

    pred_hm = put_colorized_points(height, width, pred_points, pred_scores)
    pred_color = alpha_blend(color, pred_hm, None)

    fig = plt.figure(figsize=(12, 5), tight_layout=True, dpi=DPI)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title('Ground Truth')
    ax1.set_axis_off()
    ax1.imshow(gt_color)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('Prediction')
    ax2.set_axis_off()
    ax2.imshow(pred_color)

    try:
        yield fig
    finally:
        plt.clf()
        plt.close()


@contextlib.contextmanager
def make_bsgen_figure(inputs: dict, outputs: dict) -> plt.Figure:
    color = to_numpy(inputs['color', 0, 0][0])
    height, width = color.shape[-2:]
    color = to_u8(color)

    pred_points = to_numpy(outputs['bs_point', 0, 0][0])
    pred_scores = to_numpy(outputs['bs_confidence', 0, 0][0])

    if 'bs_gt' in inputs:
        gt_points = to_numpy(inputs['bs_gt'][0][..., :2])
    else:
        gt_points = []

    gt_hm = put_colorized_points(height, width, gt_points)
    gt_color = alpha_blend(color, gt_hm, None)

    pred_hm = put_colorized_points(height, width, pred_points, pred_scores)
    pred_color = alpha_blend(color, pred_hm, None)

    ogm = to_u8(to_numpy(outputs['OGM', 0, 0].data[0].sigmoid()))
    ogm = ImageOps.flip(to_pil(ogm))

    proj_ogm, ogm_alpha = make_perspective_grid(
        outputs['OGM', 0, 0], outputs['T_ogm2cam', 0, 0],
        inputs['K', 0], height, width, cmap=cv2.COLORMAP_JET)
    ogm_alpha = ogm_alpha.clip(0.0, 0.7)
    proj_color = ogm_alpha * proj_ogm + (1 - ogm_alpha) * color
    proj_color = proj_color.astype(np.uint8)

    fig = plt.figure(figsize=(10, 5), tight_layout=True, dpi=DPI)

    l_ax1 = fig.add_subplot(3, 2, 1)
    l_ax1.set_axis_off()
    l_ax1.set_title('Annotated Ground Truth')
    l_ax1.imshow(gt_color)

    l_ax2 = fig.add_subplot(3, 2, 3)
    l_ax2.set_axis_off()
    l_ax2.set_title('Generated Ground Truth')
    l_ax2.imshow(pred_color)

    l_ax3 = fig.add_subplot(3, 2, 5)
    l_ax3.set_axis_off()
    l_ax3.set_title('Projected Occupancy Grid Map')
    l_ax3.imshow(proj_color)

    r_ax1 = fig.add_subplot(1, 2, 2)
    r_ax1.set_title('Occupancy Grid Map')
    r_ax1.set_axis_off()
    r_ax1.imshow(ogm, vmin=0, vmax=255, cmap='binary')

    try:
        yield fig
    finally:
        plt.clf()
        plt.close()


def make_perspective_grid(
        ogm,
        T_ogm2cam: torch.Tensor,
        K: torch.Tensor,
        height: int,
        width: int,
        cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    corner_points = ogm.create_grid_coords()[0][:1]
    corner_points = transform_point_cloud(T_ogm2cam[:1], corner_points)
    img_points = project_to_2d(corner_points, K[:1])
    img_points = img_points.view(2, -1).transpose(0, 1)
    img_points = img_points.cpu().numpy().astype(int)

    ogm_prob = ogm.data[0].sigmoid().view(-1).cpu().numpy()

    zs = np.arange(1, ogm.size)
    xs = np.arange(0, ogm.size - 1)
    meshgrid = np.meshgrid(xs, zs, indexing='xy')
    x, z = np.stack(meshgrid)
    indices = x + z * ogm.size

    lt_idx = indices[:-1, :-1].flatten()
    rt_idx = indices[1:, :-1].flatten()
    rb_idx = indices[1:, 1:].flatten()
    lb_idx = indices[:-1, 1:].flatten()

    lt = img_points[lt_idx]
    rt = img_points[rt_idx]
    rb = img_points[rb_idx]
    lb = img_points[lb_idx]
    all_pts = np.stack((lt, rt, rb, lb, lt), axis=1)

    probs = ogm_prob[lt_idx]

    proj_ogm = np.zeros((height, width), dtype=np.uint8)
    for pts, prob in zip(all_pts, probs):
        color = (max(int(255 * prob), 1), )
        cv2.fillConvexPoly(proj_ogm, pts, color)

    alpha = proj_ogm > 0

    proj_ogm_color = cv2.applyColorMap(proj_ogm, cmap)
    proj_ogm_color = cv2.cvtColor(proj_ogm_color, cv2.COLOR_BGR2RGB)

    return proj_ogm_color, alpha[:, :, None]
