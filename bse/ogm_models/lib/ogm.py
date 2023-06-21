import copy
from functools import lru_cache

import torch


class OGM:
    """
    Class holding basic infomation and a tensor about a OGM.
    """

    def __init__(
            self,
            batch_size: int,
            size: int = 128,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device('cpu')):
        self.factory_args = {
            'dtype': dtype,
            'device': device
        }
        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.size = size
        self.shape = (self.batch_size, 1, self.size, self.size)
        self.data = torch.zeros(self.shape, **self.factory_args)
        self.height = torch.zeros((self.batch_size,), **self.factory_args)
        self.scale = None
        self.shift = None

    @property
    @lru_cache(maxsize=1)
    def pose(self):
        pose = torch.eye(4, **self.factory_args)
        pose = pose.view(1, 4, 4).expand(self.batch_size, -1, -1).clone()
        pose[:, :3, :3] *= self.scale.view(-1, 1, 1)
        pose[:, :3, 3] = self.shift.view(-1, 3)
        return pose

    def setup_resolution(self, *args, **kwargs):
        raise NotImplementedError()

    def clone(self):
        other = copy.deepcopy(self)
        return other

    def clear_occupancy(self):
        self.data = torch.zeros_like(self.data)

    def create_grid_coords(self, subgrid_num: int = 1) -> tuple:
        """
        Create a grid 3D corrdinates for projection.

        Args:
          subgrid_num: Number of divisions per side of a cell
        """

        # 0, 1, 2, 3...
        mesh = torch.tensor(list(range(self.size)), **self.factory_args)
        grid_num = self.size
        if subgrid_num > 1:
            # 0.0, 0.1, 0.2, 0.3, ...
            floating_mesh = torch.tensor(
                list(range(subgrid_num)), **self.factory_args) / subgrid_num

            # [0.0, 0.1, 0.2, ..., 0.9], [0.0, 0.1, 0.2, ..., 0.9], ...
            floating_mesh = floating_mesh.view(1, -1).expand(self.size, -1)

            # [0, 0, 0, 0...], [1, 1, 1, 1, ...], [2, 2, 2, 2, ...], ...
            mesh = mesh.view(1, -1).expand(subgrid_num, -1).transpose(0, 1)

            floating_mesh = floating_mesh.flatten()
            mesh = mesh.flatten()

            # 0.0, 0.1, ..., 1.0, 1.1, ..., 2.0, 2.1, ...
            mesh += floating_mesh
            grid_num *= subgrid_num

        meshgrid = torch.meshgrid(mesh, mesh, indexing='xy')
        meshgrid = torch.stack(meshgrid)
        meshgrid = meshgrid.view(1, 2, -1).expand(self.batch_size, -1, -1)

        cell_height = self.height.view(-1, 1).expand(-1, meshgrid.shape[-1])
        cam_points = torch.stack(
            (meshgrid[:, 0], cell_height, meshgrid[:, 1]), dim=1)

        return cam_points, grid_num


class RelativeOGM(OGM):
    def setup_resolution(
            self,
            ptcloud: torch.Tensor,
            inlier_mask: torch.Tensor,
            median_scale: float = None,
            scale: torch.Tensor = None,
            shift: torch.Tensor = None):
        """
        Calculate sufficient scales and shifts to quantize
        continuous valued point clouds as a grid.

        Args:
          ptcloud       : Point cloud rotated with respect to a plane
          inlier_mask   : The inlier mask to be considered as ground
          median_scale  : Number of OGM cells mapping coordinates corresponding
                          to the median of the inliers' z-coordinates
        """
        if shift is not None and scale is not None:
            self.scale = scale
            self.shift = shift
            return

        if scale is not None:
            self.scale = scale
        else:
            self.scale = torch.zeros((self.batch_size,), **self.factory_args)

        if shift is not None:
            self.shift = shift
        else:
            self.shift = torch.zeros((self.batch_size, 3), **self.factory_args)

        for b in range(self.batch_size):
            ground_idxs = inlier_mask[b].view(-1)

            if scale is None:
                # Quantize the float coordinates to the int coordinates
                # while keeping the appropriate significant digits.
                self.scale[b] = \
                    median_scale / ptcloud[b, 2, ground_idxs].median()

            if shift is not None:
                continue

            scaled = self.scale[b] * ptcloud[b]

            # The x-axis is moved to the center of the OGM, since the center of
            # the camera is 0 before shifting.
            x_shift = self.size / 2

            # The median of inlier is considered as the height of the road.
            y_shift = -float(scaled[1, ground_idxs].median())

            # The negative z-coordinates must be correctly shifted forward
            # because the point clouds back-projected from the image are all
            # forward point clouds.
            z_shift = torch.relu(-scaled[2].min())   # only negative

            self.shift[b] = torch.tensor(
                [x_shift, y_shift, z_shift], **self.factory_args)


class AbsoluteOGM(OGM):
    def setup_resolution(
            self,
            ptcloud: torch.Tensor,
            inlier_mask: torch.Tensor,
            max_depth: float):
        """
        Computes scales and shifts in absolute point cloud.

        Args:
          ptcloud     : Point cloud rotated with respect to a plane
          inlier_mask : The inlier mask to be considered as ground
          max_depth   : Distance of the point cloud to be considered.
        """
        proj_scale = self.size / max_depth
        self.scale = torch.full(
            (self.batch_size,), proj_scale, **self.factory_args)
        self.shift = torch.zeros((self.batch_size, 3), **self.factory_args)
        for b in range(self.batch_size):
            scaled = self.scale[b] * ptcloud[b]

            # The x-axis is moved to the center of the OGM, since the center of
            # the camera is 0 before shifting.
            x_shift = self.size / 2

            # The median of inlier is considered as the height of the road.
            ground_idxs = inlier_mask[b].view(-1)
            y_shift = -float(scaled[1, ground_idxs].median())

            # The negative z-coordinates must be correctly shifted forward
            # because the point clouds back-projected from the image are all
            # forward point clouds.
            z_shift = torch.relu(-scaled[2].min())   # only negative

            self.shift[b] = torch.tensor(
                [x_shift, y_shift, z_shift], **self.factory_args)
