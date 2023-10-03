import torch
from torch import nn
from torch.nn import functional as F

from bse.utils.projector import project_to_2d, transform_point_cloud

from .ogm import OGM
from .util import convert_to_sampler


def stack_list(x):
    return torch.stack(x) if len(x) > 0 else torch.tensor([])


class ImageProjector(torch.nn.Module):
    """
    Project the OGM peaks as the image coordinates.
    """

    def __init__(
            self,
            prob_thr: float = 0.5,
            pool_kernel: float = 9):
        """
        Args:
          threshold_prob : Threshold of occupancy probability to be extacted.
          pool_kernel    : Kernel size of max pooling applied to the OGM.
        """
        super().__init__()
        self.prob_thr = prob_thr
        self.pool_kernel = pool_kernel

        self.proj_left_offset = nn.Parameter(
            torch.tensor([1.0, 0, 0.0], dtype=torch.float32),
            requires_grad=False)
        self.proj_right_offset = nn.Parameter(
            torch.tensor([0.0, 0, 0.0], dtype=torch.float32),
            requires_grad=False)

        self.sample_left_offset = nn.Parameter(
            torch.tensor([2.0, 0, 1.0], dtype=torch.float32),
            requires_grad=False)
        self.sample_right_offset = nn.Parameter(
            torch.tensor([-1.0, 0, 1.0], dtype=torch.float32),
            requires_grad=False)

    def forward(
            self,
            ogm: OGM,
            sampling_mask: torch.Tensor,
            K: torch.Tensor,
            T_ogm2cam: torch.Tensor) -> tuple:
        """
        Args:
          ogm:
          sampling_mask(N, 1, H, W) : The image mask for the visible ground
          K(N, 3, 3)                : Camera intrinsic matrix.
          T_ogm2cam(N, 3, 3)        : Transformation matrix from the ogm
                                      coordinates to the camera coordinates.
        """
        batch_size = ogm.batch_size
        height, width = sampling_mask.shape[-2:]
        factory_args = {'dtype': ogm.dtype, 'device': ogm.device}

        ogm_prob = ogm.data.sigmoid()

        visible_mask = self.masking_visible_cell(
                ogm, sampling_mask, K, T_ogm2cam)

        peak_cells = [[] for _ in range(batch_size)]
        peak_probs = [[] for _ in range(batch_size)]
        extracted_cells = self.extract_occupancy_peak(ogm_prob, visible_mask)
        for cell in extracted_cells:
            b, _, z, x = cell

            peak_probs[b].append(ogm_prob[b, 0, z, x])

            y = ogm.height[b]
            x = x.to(y)
            z = z.to(y)
            cell_ptcloud = torch.stack((x, y, z))

            # To reduce quantization error, the coordinates of the grid are
            # mapped to the back side near the center of road.
            if x < ogm.size // 2:
                cell_ptcloud += self.proj_left_offset
            else:
                cell_ptcloud += self.proj_right_offset

            peak_cells[b].append(cell_ptcloud)

        peak_cells = [stack_list(c) for c in peak_cells]
        peak_probs = [stack_list(p) for p in peak_probs]
        peak_img_points = [
            torch.tensor([], **factory_args) for _ in range(batch_size)]

        nums = [len(cell) for cell in peak_cells]
        total_num = int(torch.tensor(nums, **factory_args).sum())
        if total_num == 0:
            return peak_img_points, peak_probs, peak_cells

        # To reduce the computational cost of the projection, all peaks
        # between batches are combined to be considered as a batch and
        # finally decomposed.

        # (B + Ngt) x 3 x 1
        cells = torch.cat([
            cell.view(-1, 3, 1)
            for i, cell in enumerate(peak_cells) if nums[i] > 0])

        # (B + Ngt) x 3 x 3
        T = torch.cat([
            _T.view(1, 4, 4).expand(nums[i], -1, -1)
            for i, _T in enumerate(T_ogm2cam) if nums[i] > 0])

        # (B + Ngt) x 4 x 4
        K = torch.cat([
            _K.view(1, 4, 4).expand(nums[i], -1, -1)
            for i, _K in enumerate(K) if nums[i] > 0])

        cells = transform_point_cloud(T, cells)
        points = project_to_2d(cells, K)
        points[:, 0] /= width - 1
        points[:, 1] /= height - 1
        points = points.clamp(min=0.0, max=1.0)
        points = torch.index_select(
            points, 1, torch.tensor([1, 0], device=factory_args['device']))

        peak_img_points = points.view(-1, 2).split(nums)
        return peak_img_points, peak_probs, peak_cells

    def masking_visible_cell(
            self,
            ogm: OGM,
            sampling_mask: torch.Tensor,
            K: torch.Tensor,
            T_ogm2cam: torch.Tensor) -> torch.Tensor:
        """
        Create maks for visible cells.

        In order to obtain a robustly sampled mask, an expanded
        sampling is performed. However, there is no guarantee that
        the horizontal center of the OGM matches the center of the road, so
        expansion is performed on both sides.
        """

        batch_size = ogm.batch_size
        height, width = sampling_mask.shape[-2:]

        ogm_points, N = ogm.create_grid_coords()

        ogm_points = ogm_points.view(batch_size, 3, N, N)

        # Reduce quantization error by over-sampling in the center direction.
        ogm_points[..., :N // 2] += self.sample_left_offset.view(1, 3, 1, 1)
        ogm_points[..., N // 2:] += self.sample_right_offset.view(1, 3, 1, 1)
        ogm_points = ogm_points.view(batch_size, 3, -1)

        # Grid -> Point Cloud -> Image
        cam_points = transform_point_cloud(T_ogm2cam, ogm_points)
        img_points = project_to_2d(cam_points, K)

        # Divide into the two sampler
        sampler, img_range_mask = \
            convert_to_sampler(img_points, height, width)

        sampling_mask = sampling_mask.view(batch_size, -1)
        sampling_mask = sampling_mask.gather(1, sampler)

        mask = img_range_mask & sampling_mask
        mask = mask.view(batch_size, 1, N, N)
        return mask

    def extract_occupancy_peak(
            self,
            ogm_prob: torch.Tensor,
            visible_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract the occupancy peak using max-pooling.
        """
        # Max pool over the heatmap
        local_max = F.max_pool2d(
            ogm_prob, kernel_size=self.pool_kernel,
            stride=1, padding=self.pool_kernel // 2)
        mask = visible_mask & (local_max == ogm_prob)

        # Extract peak cells
        cells = ((ogm_prob * mask) > self.prob_thr).nonzero()
        return cells
