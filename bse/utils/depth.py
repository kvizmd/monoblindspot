import numpy as np
import scipy
import torch


def complement_sparse_depth(
        sparse_depth: np.ndarray,
        fill_value=0,
        mode='linear'):
    """
    convert lidar sparse points to dense depth map
    """

    H, W = sparse_depth.shape[-2:]
    keep_shape = sparse_depth.shape
    keep_dtype = sparse_depth.dtype
    sparse_depth = sparse_depth.reshape(H, W)

    mask = sparse_depth > 0

    sparse_y_coords, sparse_x_coords = np.where(mask)
    sparse_coords = np.stack((sparse_x_coords, sparse_y_coords), -1)

    values = sparse_depth[mask]

    x_coord = np.linspace(0, W - 1, W)
    y_coord = np.linspace(0, H - 1, H)
    xx, yy = np.meshgrid(x_coord, y_coord)

    dense_depth = scipy.interpolate.griddata(
        sparse_coords, values, (xx, yy),
        method=mode, fill_value=fill_value)

    dense_depth = dense_depth.reshape(*keep_shape).astype(keep_dtype)

    return dense_depth


def disp_to_depth(
        disp: torch.Tensor,
        min_depth: float,
        max_depth: float) -> tuple:
    """
    Convert the inversed depth map into the relative depth.
    """

    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(
        depth: torch.Tensor,
        min_depth: float,
        max_depth: float) -> torch.Tensor:
    """
    Convert the relative depth into the inversed representation.
    """
    depth = depth.clamp(min=min_depth, max=max_depth)

    min_disp = 1 / max_depth
    max_disp = 1 / min_depth

    disp = 1 / depth
    scaled_disp = (disp - min_disp) / (max_disp - min_disp)
    scaled_disp = scaled_disp.clamp(min=0.0, max=1.0)

    return scaled_disp
