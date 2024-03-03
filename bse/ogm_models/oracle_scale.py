import torch
from torch import nn

from bse.utils.projector import transform_point_cloud

from .lib.fit import compute_gflat_rotmat
from .lib.ogm import AbsoluteOGM, RelativeOGM
from .lib.util import normalize_with_mean, DepthScaleEstimator
from .ogm import OGMIntegrator


class OracleScaleOGMIntegrator(OGMIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # [1, 2, 3, ..., 50]
        self.backward_indices.sort()

        # [-1, -2, ..., -5]
        self.forward_indices.sort()
        self.forward_indices.reverse()

        T_eye = torch.eye(4, 4, dtype=torch.float32).view(1, 4, 4)
        self.T_eye = nn.Parameter(
            T_eye.expand(self.batch_size, -1, -1), requires_grad=False)

        self.compute_scale = DepthScaleEstimator(
                min_rel_depth=self.min_depth,
                max_rel_depth=self.max_depth,
                min_metric_depth=1e-3,
                max_metric_depth=80,
                scaling_ransac_iters=1000,
                scaling_ransac_thr=0.01)

    def integrate(self, inputs: dict) -> dict:
        outputs = {}

        disp = self.predict_depth(inputs['color', 0, 0])
        outputs['disp', 0] = disp
        outputs['scale', 0] = self.compute_scale(disp, inputs['depth_gt', 0])

        self.initialize_occupancy(inputs, outputs)

        # Update target occupancy grid map
        # index: -1, -2
        for i, frame_idx in enumerate(self.forward_indices):
            next_idx = 0 if i == 0 else self.forward_indices[i - 1]

            disp = self.predict_depth(inputs['color', frame_idx, 0])
            outputs['disp', frame_idx] = disp

            outputs['scale', frame_idx] = \
                self.compute_scale(disp, inputs['depth_gt', frame_idx])

            # -1 -> 0(next)
            inv_T, _ = self.predict_pose(
                inputs['color', frame_idx, 0],
                inputs['color', next_idx, 0])
            outputs['inv_T', frame_idx] = inv_T

            self.update_occupancy(inputs, outputs, frame_idx, next_idx)

        # index: 1, 2, ..., 50
        for i, frame_idx in enumerate(self.backward_indices):
            prev_idx = 0 if i == 0 else self.backward_indices[i - 1]

            disp = self.predict_depth(inputs['color', frame_idx, 0])
            outputs['disp', frame_idx] = disp

            outputs['scale', frame_idx] = \
                self.compute_scale(disp, inputs['depth_gt', frame_idx])

            # 1 -> 0(prev)
            _, inv_T = self.predict_pose(
                inputs['color', prev_idx, 0],
                inputs['color', frame_idx, 0])
            outputs['inv_T', frame_idx] = inv_T

            self.update_occupancy(inputs, outputs, frame_idx, prev_idx)

        return outputs

    @torch.no_grad()
    def initialize_occupancy(self, inputs: dict, outputs: dict):
        K = inputs['K', 0]
        inv_K = inputs['inv_K', 0]
        disp = outputs['disp', 0]
        scale = outputs['scale', 0]

        depth = self.disp_to_depth(disp)
        cam_points = self.backproject(depth, inv_K)
        cam_points = cam_points[:, :3, :]  # absolute scale point cloud
        metric_cam_points = scale.view(-1, 1, 1) * cam_points

        # Normalize with mean inversed depth
        norm_disp, mean_scale = normalize_with_mean(disp)
        norm_depth = self.disp_to_depth(norm_disp)

        outputs['inv_T_cascade', 0] = self.T_eye

        invcam_points = transform_point_cloud(self.R_y_invert, cam_points)
        plane, inlier_mask = self.planefit(invcam_points, K)
        R_flat = compute_gflat_rotmat(plane[:, :3])

        plane_disp = self.depth_to_disp(self.plane2depth(plane, inv_K))

        # Compute abs-rel to make sampling masks
        norm_plane_disp = plane_disp * mean_scale
        ground_error = (norm_plane_disp - norm_disp).abs() / norm_disp
        sampling_mask = ground_error < self.ogm_sampling_mask_offset
        outputs['sampling_mask', 0] = sampling_mask

        # Align point clouds to plane.
        T_cam2plane = torch.bmm(R_flat, self.R_y_invert)
        plane_points = transform_point_cloud(T_cam2plane, metric_cam_points)

        ogm = RelativeOGM(
            self.batch_size, self.ogm_size,
            dtype=plane_points.dtype, device=plane_points.device)
        ogm.setup_resolution(
            plane_points, inlier_mask, self.ogm_median_scale)

        ogm_points = transform_point_cloud(ogm.pose, plane_points)
        T_cam2ogm = torch.bmm(ogm.pose, T_cam2plane)
        T_ogm2cam = T_cam2ogm.inverse()
        outputs['T_cam2ogm', 0] = T_cam2ogm
        outputs['T_ogm2cam', 0] = T_ogm2cam

        self.invmodel.store_occupancy(
            ogm, norm_depth, inv_K, ogm_points, inlier_mask)

        self.multiply_confidence(ogm, plane_disp, K, T_ogm2cam)
        outputs['OGM', 0] = ogm

    @torch.no_grad()
    def update_occupancy(
            self,
            inputs: dict,
            outputs: dict,
            frame_idx: int,
            trg_frame_idx: int):
        """
        """
        K = inputs['K', 0]
        inv_K = inputs['inv_K', 0]
        disp = outputs['disp', frame_idx]
        scale = outputs['scale', frame_idx]
        T_scale = self.T_eye.clone()
        T_scale[:, :3] *= scale.view(-1, 1, 1)
        trg_ogm = outputs['OGM', 0]

        depth = self.disp_to_depth(disp)
        cam_points = self.backproject(depth, inv_K)
        cam_points = cam_points[:, :3, :]  # absolute scale

        norm_disp, mean_scale = normalize_with_mean(disp)
        norm_depth = self.disp_to_depth(norm_disp)

        plane, _ = self.planefit(
            transform_point_cloud(self.R_y_invert, cam_points), K)
        plane_disp = self.depth_to_disp(self.plane2depth(plane, inv_K))

        # Warp the current points clound into the target point cloud.
        T_cam2cam = torch.bmm(
            outputs['inv_T_cascade', trg_frame_idx],
            outputs['inv_T', frame_idx])
        outputs['inv_T_cascade', frame_idx] = T_cam2cam
        T_cam2cam = torch.bmm(T_scale, T_cam2cam)

        T_cam2invcam = torch.bmm(self.R_y_invert, T_cam2cam)
        invcam_points = transform_point_cloud(T_cam2invcam, cam_points)

        trg_plane, inlier_mask = self.planefit(invcam_points, K)
        R_flat = compute_gflat_rotmat(trg_plane[:, :3])

        plane_points = transform_point_cloud(R_flat, invcam_points)
        T_cam2plane = torch.bmm(R_flat, T_cam2invcam)

        src_ogm = RelativeOGM(
            self.batch_size, self.ogm_size,
            dtype=plane_points.dtype, device=plane_points.device)
        src_ogm.setup_resolution(
            plane_points, inlier_mask,
            scale=trg_ogm.scale, shift=trg_ogm.shift)

        ogm_points = transform_point_cloud(src_ogm.pose, plane_points)
        T_cam2ogm = torch.bmm(src_ogm.pose, T_cam2plane)
        T_ogm2cam = T_cam2ogm.inverse()

        # Normalize with mean inversed depth
        self.invmodel.store_occupancy(
            src_ogm, norm_depth, inv_K, ogm_points, inlier_mask)

        self.multiply_confidence(src_ogm, plane_disp, K, T_ogm2cam)

        # Update occpancy grid map with odds theorem
        src_ogm.data[~inputs['valid_indices', frame_idx]] = 0
        trg_ogm.data += src_ogm.data + self.ogm_update_prior_odds

        outputs['OGM', frame_idx] = src_ogm
