import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
from torch.nn import functional as F

from .projector import project_to_3d


def compute_depth_errors(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }


def compute_eigen_depth_errors(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    pred = F.interpolate(
        pred, gt.shape[-2:], mode='bilinear', align_corners=False)
    pred = torch.clamp(pred, 1e-3, 80)

    mask = gt > 0

    # garg/eigen crop
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = True
    mask = mask * crop_mask

    gt = gt[mask]
    pred = pred[mask]

    scale = torch.median(gt) / torch.median(pred)
    pred = scale * pred
    pred = torch.clamp(pred, min=1e-3, max=80)

    depth_errors = compute_depth_errors(
        gt.cpu().numpy(), pred.cpu().numpy())
    return depth_errors


def count_blindspot_confusion(
        pred_bss: np.ndarray,
        gt_bss: np.ndarray,
        threshold: float = 1.0) -> dict:
    """
    Compute the confusion matrix between the predictions and ground-truth.

    Args:
      pred_bss: N x 3
      gt_bss: M x 3
    """
    Np = pred_bss.shape[0]
    Ng = gt_bss.shape[0]

    if Np == 0 and Ng == 0:
        return {}

    if Np == 0:
        return {'FN': Ng}

    if Ng == 0:
        return {'FP': Np}

    D = gt_bss.shape[-1]

    cost_row = pred_bss.reshape(Np, 1, D).repeat(Ng, axis=1)
    cost_col = gt_bss.reshape(1, Ng, D).repeat(Np, axis=0)
    diff = cost_row - cost_col
    cost = (diff ** 2).sum(axis=2) ** 0.5

    judges = cost <= threshold

    # Select the pair with the most correct true points.
    pred_indices, gt_indices = linear_sum_assignment(judges, maximize=True)

    # matched_cost = cost[pred_indices, gt_indices]
    match_result = judges[pred_indices, gt_indices]

    TP = match_result.sum()
    FP = Np - TP
    FN = Ng - TP

    results = {
        'TP': TP,
        'FP': FP,
        'FN': FN
    }
    return results


def count_blindspot_zsubset_confusion(
        pred_bss: np.ndarray,
        gt_bss: np.ndarray,
        ranges: dict = {'all': (None, None)},
        threshold: float = 1) -> dict:
    """
    Compute the confusion matrix between the predictions and ground-truth.
    It returns results divided into a range of evaluation distances in Z.

    pred_bss: N x 3
    gt_bss: M x 3
    """

    def _cap(zs, lower, upper):
        mask = np.ones_like(zs, dtype=bool)
        if lower is not None:
            mask &= zs >= lower

        if upper is not None:
            mask &= zs < upper

        return mask

    out_metrics = {}
    for key, (lower, upper) in ranges.items():
        # pred_distance = (pred_bss ** 2).sum(-1) ** 0.5
        # gt_distance = (gt_bss ** 2).sum(-1) ** 0.5
        pred_distance = pred_bss[:, 2]
        gt_distance = gt_bss[:, 2]

        pred_bss_mask = _cap(pred_distance, lower, upper)
        gt_bss_mask = _cap(gt_distance, lower, upper)

        pred_bss_subset = pred_bss[pred_bss_mask]
        gt_bss_subset = gt_bss[gt_bss_mask]

        metrics = count_blindspot_confusion(
            pred_bss_subset, gt_bss_subset, threshold=threshold)

        for name, val in metrics.items():
            out_metrics[key + '/' + name] = val

    return out_metrics


def compute_binary_metrics(TP: int, FP: int, FN: int) -> dict:
    metrics = {}

    if (TP + FN) > 0:
        recall = TP / (TP + FN)
        metrics['recall'] = recall
    else:
        recall = None

    if (TP + FP) > 0:
        precision = TP / (TP + FP)
        metrics['precision'] = precision
    else:
        precision = None

    if TP > 0 and recall is not None and precision is not None:
        f1 = 2 * recall * precision / (recall + precision)
        metrics['f1'] = f1

    return metrics


def ignore_negative(points: torch.Tensor) -> torch.Tensor:
    """
    Ignore negative points

    Args:
      points: shape: (N, 2)
    """

    # Pack grount-truth from batchs
    mask = points.amin(-1) >= 0
    points = points[mask.view(-1), :]
    return points


def ignore_on_mask(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Ignore points if the mask value at the point is true.

    Args:
      points: shape: (N, 2)
      mask: shape: (1, H, W)
    """
    if len(points) == 0:
        return points

    H, W = mask.shape[-2:]
    y_idx = (points[:, 0] * (H - 1)).long()
    x_idx = (points[:, 1] * (W - 1)).long()
    ignore_mask = mask.view(-1)[x_idx + W * y_idx]
    return points.view(-1, 2)[~ignore_mask]


def project_to_campoints(
        points_yx: torch.Tensor,
        dense_depth: torch.Tensor,
        inv_K: torch.Tensor,
        ignore_zero_depth: bool = True) -> torch.Tensor:
    """
    Project the points at image coordinates to the camera coordinates.

    Args:
      normalized_yx: normalized yx image coordinates  (shape: N x 2)
      dense_depth: The dense absolute depth           (shape: 1 x H x W)
      inv_K: Inverse matrix of intrinsics             (shape: 4 x 4)
      ignore_zero_depth: Ignore the points having invalud depth.
    """
    H, W = dense_depth.shape[-2:]
    ys = points_yx[:, 0] * (H - 1)
    xs = points_yx[:, 1] * (W - 1)
    points = torch.stack((xs, ys))

    depth = dense_depth.view(1, -1)[:, xs.long() + W * ys.long()]

    if ignore_zero_depth:
        mask = (depth > 0).view(-1)
        points = points[:, mask]
        depth = depth[:, mask]

    cam_points = project_to_3d(
        points.unsqueeze(0), depth.unsqueeze(0), inv_K.unsqueeze(0))
    cam_points = cam_points[0].permute(1, 0)

    return cam_points


def evaluate_blindspot(
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        depth: torch.Tensor,
        inv_K: torch.Tensor,
        depth_ranges: dict = {'all': (None, None)},
        threshold: float = 1.0) -> dict:
    """
    Evaluate blind spot using the ground depth map.

    Args:
      pred_points: the prediction    (shape: Np x 2)
      gt_points:  the ground truth   (shape: Ng x 2)
      depth: the dense depth         (shape: 1 x H x W)
      inv_k:
      depth_ranges:
      threshold:
    """
    pred_bss = project_to_campoints(
        pred_points, depth, inv_K, ignore_zero_depth=True)

    gt_bss = project_to_campoints(
        gt_points, depth, inv_K, ignore_zero_depth=True)

    metrics = count_blindspot_zsubset_confusion(
        pred_bss.cpu().numpy(),
        gt_bss.cpu().numpy(),
        ranges=depth_ranges,
        threshold=threshold)

    return metrics
