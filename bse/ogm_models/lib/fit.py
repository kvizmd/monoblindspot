# from functools import lru_cache

import torch
from torch import nn

from bse.utils.ransac import \
    RANSAC, \
    is_inlier, \
    do_faster_ransac


class GroundPlaneFit(nn.Module):
    """
    Fit the plane of the ground using prior infomations.
    """

    def __init__(
            self,
            height: int,
            width: int,
            max_iterations: int = 200):
        """
        Args:
          height                : Height of image.
          width                 : Width of image.
          max_iterations        : Max iteration of RANSAC.
        """

        super().__init__()

        self.H = height
        self.W = width

        self.max_iterations = max_iterations

    def get_ground_prior(self, K, prior_resolution: int = 8) -> torch.Tensor:
        """
        Args:
          prior_resolution: the number of horizontal divisions.
        """
        assert prior_resolution % 2 == 0

        ys = K[:, 1, 2].mean(0).to(int)
        xs = (self.W * (prior_resolution // 2 - 1)) // prior_resolution
        xe = (self.W * (prior_resolution // 2 + 1)) // prior_resolution
        prior_mask = torch.zeros((1, self.H, self.W), dtype=torch.bool)
        prior_mask[..., ys:, xs:xe] = True
        return prior_mask

    def forward(
            self,
            ptcloud: torch.Tensor,
            K: torch.Tensor,
            threshold: float = None) -> tuple:
        """
        Computes plane parameters from a point cloud.

        Args:
          ptcloud   : Point cloud
          threshold : Threshold for the distance to be considered as inliers.
                      If None, adaptively and automatically chosen.
          invert_y_axis: Reverse the y-axis direction.
        """
        ground_prior_mask = self.get_ground_prior(K).to(ptcloud.device)
        prior_ptcloud = ptcloud[..., ground_prior_mask.view(-1)]

        prior_ptcloud = prior_ptcloud.transpose(1, 2)

        if threshold is None:
            threshold = compute_mad_threshold(prior_ptcloud[..., 1])

        plane = do_faster_ransac(
            prior_ptcloud,
            max_iterations=self.max_iterations,
            plane_distance_thr=threshold)

        # The ground mask is used for detecting the lowest points
        # However, the mask is not perfectly.
        inliner = is_inlier(plane, ptcloud.transpose(1, 2), threshold)

        batch_size = ptcloud.shape[0]
        inlier_mask = inliner.view(batch_size, self.H, self.W)

        return plane, inlier_mask


def compute_mad_threshold(data: torch.Tensor) -> torch.Tensor:
    """
    Based on scikit-learn's RANSACRegressor, threshold is chosen as MAD.
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html

    For simplicity of implementation, mutual exclusion for
    multi-threading is not performed.

    data: batched data for specific dimension (B, N)
    """
    med, _ = torch.median(data, dim=1, keepdims=True)
    abs_deviation = torch.abs(med - data)

    # The residual threshold is the median of the absolute deviation.
    ada_thr, _ = torch.median(abs_deviation, dim=1, keepdims=True)
    return ada_thr


def compute_gflat_rotmat(normal: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrix to make the normal (0, 1, 0)

    Args:
      normal: The normal vector of the plane.
    """

    # The points under a half of image height is used for RANSAC.
    normal = normal.view(-1, 3, 1)

    # The points under a half of image height is used for RANSAC.
    # correct z-axis angle error of ground
    xy_theta = xy_angle_of(normal)
    z_rotmat = create_z_rotmat(xy_theta)
    normal = torch.matmul(z_rotmat, normal)

    # correct x-axis angle error of ground
    yz_theta = yz_angle_of(normal)
    x_rotmat = create_x_rotmat(yz_theta)
    normal = torch.matmul(x_rotmat, normal)

    T = torch.eye(4, dtype=normal.dtype, device=normal.device)
    T = T.view(1, 4, 4).expand(normal.shape[0], -1, -1).clone()
    T[:, :3, :3] = torch.matmul(x_rotmat, z_rotmat)
    return T


def xy_angle_of(vector: torch.Tensor) -> torch.Tensor:
    """
    Compute angle between x-axis and y-axis.
    Args:
      vector: The vector, shape: (N x 3)
    """

    return torch.atan2(vector[:, 0], vector[:, 1]).view(-1)


def yz_angle_of(vector: torch.Tensor) -> torch.Tensor:
    """
    Compute angle between y-axis and z-axis.
    Args:
      vector: The vector, shape: (N x 3)
    """

    return -torch.atan2(vector[:, 2], vector[:, 1]).view(-1)


def create_z_rotmat(theta: torch.Tensor) -> torch.Tensor:
    """
    Create z-rotation matrix from the angle.

    Args:
      theta: Angle (N)
    """

    z_rotate = torch.zeros(
        (theta.shape[0], 3, 3),
        dtype=theta.dtype, device=theta.device)
    z_rotate[:, 0, 0] = torch.cos(theta)
    z_rotate[:, 0, 1] = -torch.sin(theta)
    z_rotate[:, 1, 0] = torch.sin(theta)
    z_rotate[:, 1, 1] = torch.cos(theta)
    z_rotate[:, 2, 2] = 1.0
    return z_rotate


def create_x_rotmat(theta: torch.Tensor) -> torch.Tensor:
    """
    Create x-rotation matrix from the angle.

    Args:
      theta: Angle (N)
    """

    x_rotate = torch.zeros(
        (theta.shape[0], 3, 3),
        dtype=theta.dtype, device=theta.device)
    x_rotate[:, 0, 0] = 1.0
    x_rotate[:, 1, 1] = torch.cos(theta)
    x_rotate[:, 1, 2] = -torch.sin(theta)
    x_rotate[:, 2, 1] = torch.sin(theta)
    x_rotate[:, 2, 2] = torch.cos(theta)
    return x_rotate


def get_rectified_plane(plane: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        torch.zeros_like(plane[:, 0]),
        torch.ones_like(plane[:, 1]),
        torch.zeros_like(plane[:, 2]),
        plane[:, 3]], dim=-1)
