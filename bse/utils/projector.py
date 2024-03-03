import numpy as np

import torch
from torch import nn


class BackprojectDepth(nn.Module):
    def __init__(self, batch_size: int, height: int, width: int):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(
            range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(
            self,
            depth: torch.Tensor,
            inv_K: torch.Tensor) -> torch.Tensor:
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(-1, 1, self.height * self.width) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    def __init__(
            self,
            batch_size: int,
            height: int,
            width: int,
            eps: float = 1e-7):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(
            self,
            points: torch.Tensor,
            K: torch.Tensor,
            T: torch.Tensor) -> torch.Tensor:
        # Project the 3D points into the image coordinates.
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)

        # Normalize the scale factor.
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2:3, :] + self.eps)

        # Normalize the coordinates to apply F.grid_sample().
        pix_coords = pix_coords.view(-1, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def project_to_2d(
        xyz: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor = None,
        eps: float = 1e-10) -> torch.Tensor:
    """
    Projects the 3D points into the image plane.
    """
    if T is None:
        P = K[:, :3, :]
    else:
        P = torch.matmul(K, T)[:, :3, :]

    ones = torch.ones_like(xyz[:, :1], requires_grad=False)
    xyza = torch.cat([xyz, ones], dim=-2)

    uv = torch.matmul(P, xyza)
    uv = uv[:, :2, :] / (uv[:, 2:3, :] + eps)
    return uv


def project_to_3d(
        uv: torch.Tensor,
        depth: torch.Tensor,
        inv_K: torch.Tensor) -> torch.Tensor:
    """
    Backprojects the depth map into the 3D coordinates.
    """
    ones = torch.ones_like(uv[:, :1], requires_grad=False)
    uva = torch.cat((uv, ones), dim=1)

    xyz = torch.matmul(inv_K[:, :3, :3], uva)
    xyz = depth.view(depth.shape[0], 1, -1) * xyz
    return xyz


def transform_point_cloud(
        T: torch.Tensor,
        point_cloud: torch.Tensor,
        invert: bool = False) -> torch.Tensor:
    ones = torch.ones_like(point_cloud[:, :1, :])
    xyzs = torch.cat((point_cloud, ones), 1)

    if invert:
        return torch.linalg.solve(T, xyzs)[:, :3]
    return torch.bmm(T, xyzs)[:, :3]


class PlaneToDepth(nn.Module):
    """
    Layer to transform a plane parameter to depth map
    """

    def __init__(
            self,
            batch_size: int,
            height: int,
            width: int,
            min_depth: float = 0.0,
            max_depth: float = None,
            negative_depth_value: float = None):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth

        if negative_depth_value is None:
            assert self.max_depth is not None
            self.neg_depth = self.max_depth
        else:
            self.neg_depth = negative_depth_value

        meshgrid = np.meshgrid(
            range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(
            self,
            plane_param: torch.Tensor,
            inv_K: torch.Tensor) -> torch.Tensor:
        """
        plane_param: The batched plane parameters, such as [a, b, c, d]
                     in ax + by + cz + d = 0.
                     The normal vector [a,b,c] assumes the negative direction
                     of the camera y-coordinate. Therefore, it prefers the
                     normal vector at the point cloud with inverted
                     y-coordinate of the back-projected point cloud.

        inv_K: inversed camera intrinsic mateix.
        """

        # unscaled direction vector
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points[:, 1, :] *= -1

        normal, d = plane_param[:, None, :3], plane_param[:, None, 3:]
        depth = -torch.abs(d) / torch.bmm(normal, cam_points)

        # Since the normal vector of the ground assumes the negative
        # direction of the y-axis, coordinates that intersect behind
        # the camera, such as the sky, have a negative depth, while
        # points on the ground in front have a positive depth.
        depth[depth < 0] = self.neg_depth
        depth = depth.clamp(min=self.min_depth, max=self.max_depth)

        depth = depth.view(-1, 1, self.height, self.width)

        return depth
