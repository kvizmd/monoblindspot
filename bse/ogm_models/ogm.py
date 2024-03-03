import torch
from torch import nn

from bse.utils.projector import \
    BackprojectDepth, \
    Project3D, \
    PlaneToDepth, \
    project_to_2d, \
    transform_point_cloud
from bse.utils.pose import transformation_from_parameters
from bse.utils.depth import disp_to_depth, depth_to_disp

from .lib.fit import GroundPlaneFit
from .lib.imgproj import ImageProjector
from .lib.invsnr import InverseSensorModel
from .lib.ogm import OGM
from .lib.util import \
    convert_to_sampler, \
    prob2logodds, \
    batch_masked_sampling


class OGMIntegrator(nn.Module):
    def __init__(
            self,
            batch_size: int,
            height: int,
            width: int,
            frame_indices: list,
            depth_net: nn.Module,
            pose_net: nn.Module,
            min_depth: float = 0.1,
            max_depth: float = 100,
            ogm_size: int = 128,
            ogm_num_subgrids: int = 8,
            ogm_median_scale: float = 16,
            ogm_ransac_iterations: int = 100,
            ogm_invsnr_grad_thr: float = 0.0005,
            ogm_invsnr_count_thr: int = 3,
            ogm_invsnr_min_prob: float = 0.35,
            ogm_invsnr_max_prob: float = 0.65,
            ogm_invsnr_free_prob: float = 1e-10,
            ogm_invsnr_height_quantile: float = 0.75,
            ogm_update_prior_prob: float = 0.5,
            ogm_project_prob_thr: float = 0.55,
            ogm_project_pool_kernel: int = 7,
            ogm_sampling_mask_acceptable_offset: float = 0.2,
            **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.ogm_update_prior_odds = prob2logodds(ogm_update_prior_prob)
        self.ogm_num_subgrids = ogm_num_subgrids
        self.ogm_size = ogm_size
        self.ogm_sampling_mask_offset = ogm_sampling_mask_acceptable_offset
        self.ogm_median_scale = ogm_median_scale

        self.depth_net = depth_net
        self.depth_net.eval()

        self.pose_net = pose_net
        self.pose_net.eval()

        self.backproject = BackprojectDepth(batch_size, height, width)
        self.project = Project3D(batch_size, height, width)

        self.plane2depth = PlaneToDepth(
            batch_size, height, width,
            min_depth=min_depth, max_depth=max_depth)

        self.planefit = GroundPlaneFit(
            height, width,
            max_iterations=ogm_ransac_iterations)

        self.invmodel = InverseSensorModel(
            grad_thr=ogm_invsnr_grad_thr,
            count_thr=ogm_invsnr_count_thr,
            min_prob=ogm_invsnr_min_prob,
            max_prob=ogm_invsnr_max_prob,
            height_quantile=ogm_invsnr_height_quantile)

        self.ogm2img = ImageProjector(
            prob_thr=ogm_project_prob_thr,
            pool_kernel=ogm_project_pool_kernel)

        self.frame_indices = frame_indices
        self.forward_indices = []
        self.backward_indices = []
        for frame_idx in self.frame_indices:
            if frame_idx == 0:
                continue

            if frame_idx < 0:
                self.forward_indices.append(frame_idx)
            else:
                self.backward_indices.append(frame_idx)

        R_y_invert = torch.tensor([
            1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1], dtype=torch.float32).view(1, 4, 4)
        R_y_invert = R_y_invert.expand(self.batch_size, -1, -1)
        self.R_y_invert = nn.Parameter(R_y_invert, requires_grad=False)

    def forward(self, inputs: dict) -> dict:
        outputs = self.integrate(inputs)

        ogm = outputs['OGM', 0]
        T_ogm2cam = outputs['T_ogm2cam', 0]

        points, probs, peak_cells = self.ogm2img(
            ogm,
            outputs['sampling_mask', 0],
            inputs['K', 0],
            T_ogm2cam)

        outputs.update({
            ('bs_point', 0): points,
            ('bs_confidence', 0): probs,
        })

        # The transformation matrix and the peak cell coordinates are
        # normalized with the grid size (e.g. 128).
        T_norm2ogm = torch.eye(4, dtype=ogm.dtype, device=ogm.device)
        T_norm2ogm[:3, :3] *= ogm.size
        T_norm2cam = torch.matmul(T_ogm2cam, T_norm2ogm.view(1, 4, 4))

        peak_cells = [c / (ogm.size - 1) for c in peak_cells]

        outputs.update({
            ('T_norm2cam', 0): T_norm2cam,
            ('bs_peak_cell', 0): peak_cells,
        })

        return outputs

    def integrate(self, inputs: dict) -> dict:
        raise NotImplementedError()

    @torch.inference_mode()
    def predict_depth(self, frame: torch.Tensor) -> torch.Tensor:
        outputs = self.depth_net(frame)
        disp = outputs['disp', 0]
        return disp

    @torch.inference_mode()
    def predict_pose(
            self,
            frame_1: torch.Tensor,
            frame_2: torch.Tensor) -> tuple:
        pose_inputs = torch.cat([frame_1, frame_2], 1)
        axisangle, translation = self.pose_net(pose_inputs)

        T = transformation_from_parameters(
            axisangle, translation, invert=False)

        inv_T = transformation_from_parameters(
            axisangle, translation, invert=True)

        return T, inv_T

    def disp_to_depth(self, disp: torch.Tensor) -> torch.Tensor:
        _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        return depth

    def depth_to_disp(self, depth: torch.Tensor) -> torch.Tensor:
        disp = depth_to_disp(depth, self.min_depth, self.max_depth)
        return disp

    def multiply_confidence(
            self,
            ogm: OGM,
            plane_disp: torch.Tensor,
            K: torch.Tensor,
            T_ogm2cam: torch.Tensor):
        """
        The point cloud projected from the image becomes less
        reliable as it becomes sparser with distance. Therefore,
        weighting is performed with normalized inverse depth.

        Since locations that appear larger (closer) in the image
        are more reliable, the value sampled from the inversed
        depth map is used as the confidence value. However, since
        the depth map is a relative depth and the nearest ground
        is not 1.0, it is normalized by the maximum value minimum
        value.
        """

        # Generate sampling points in the target cell to update.
        ogm_points, _ = ogm.create_grid_coords(self.ogm_num_subgrids)
        ogm_shape_sub = (ogm.size, self.ogm_num_subgrids) * 2

        # Transform these points into the original camera coordinates(OGM->cam)
        cam_points = transform_point_cloud(T_ogm2cam, ogm_points)

        # Warp the 3d points from previous frame to the current frame
        # and projects the points into the image coordinates.
        img_points = project_to_2d(cam_points, K)

        sampling_indices, img_range_mask = \
            convert_to_sampler(img_points, self.height, self.width)

        # Only consider points succeeded projection.
        subcell_mask = img_range_mask

        # The number of the warped into the ground from the first cells.
        subcell_num = \
            subcell_mask.view(-1, 1, *ogm_shape_sub).sum(dim=5).sum(dim=3)

        # Samples the depth values in 2D coordinates because the
        # self-supervised depth does not gurantee scale-consistency.
        sample_disp = batch_masked_sampling(
            plane_disp.view(self.batch_size, -1),
            sampling_indices, img_range_mask)
        sample_disp = sample_disp.view(-1, 1, *ogm_shape_sub)

        # apply confidence weighting with the average of inversed depth
        sample_disp = sample_disp.view(-1, 1, *ogm_shape_sub).sum(dim=5).sum(3)
        sample_disp /= subcell_num.clamp(min=1)

        min_disp = sample_disp.amin((2, 3), True)
        max_disp = sample_disp.amax((2, 3), True)
        disp_range = (max_disp - min_disp).clamp(min=1e-10)
        z_confidence = (sample_disp - min_disp) / disp_range

        ogm.data *= z_confidence
