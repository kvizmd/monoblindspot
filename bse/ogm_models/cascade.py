import torch

from bse.utils.projector import \
    project_to_2d, \
    project_to_3d, \
    transform_point_cloud

from .lib.fit import compute_gflat_rotmat
from .lib.ogm import RelativeOGM
from .lib.util import \
    convert_to_sampler, \
    batch_masked_sampling, \
    normalize_with_mean
from .ogm import OGMIntegrator


class CascadeOGMIntegrator(OGMIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # [50, 49, ..., 1]
        self.backward_indices.sort()
        self.backward_indices.reverse()

        # [-5, -4, ..., -1]
        self.forward_indices.sort()

    def integrate(self, inputs: dict) -> dict:
        outputs = {}

        outputs['disp', 0] = self.predict_depth(inputs['color', 0, 0])
        self.predict_occupancy(inputs, outputs, 0)

        # index: -2, -1
        last_idx = len(self.forward_indices) - 1
        for i, frame_idx in enumerate(self.forward_indices):
            outputs['disp', frame_idx] = \
                self.predict_depth(inputs['color', frame_idx, 0])
            self.predict_occupancy(inputs, outputs, frame_idx)

            if i > 0:
                # prev_idx -> farme_idx
                prev_idx = self.forward_indices[i - 1]
                self.update_occupancy(inputs, outputs, prev_idx, frame_idx)

            next_idx = 0 if i == last_idx else self.forward_indices[i + 1]
            inv_T, T = self.predict_pose(
                inputs['color', frame_idx, 0], inputs['color', next_idx, 0])
            outputs['T', frame_idx] = T   # such as 0 -> -1
            outputs['inv_T', frame_idx] = inv_T  # such as -1 -> 0

        # -1 -> 0
        self.update_occupancy(inputs, outputs, frame_idx, 0)

        # index: 50, 49, 48, ..., 1
        last_idx = len(self.backward_indices) - 1
        for i, frame_idx in enumerate(self.backward_indices):
            outputs['disp', frame_idx] = \
                self.predict_depth(inputs['color', frame_idx, 0])
            self.predict_occupancy(inputs, outputs, frame_idx)

            if i > 0:
                # next_idx -> frame_idx
                next_idx = self.backward_indices[i - 1]
                self.update_occupancy(inputs, outputs, next_idx, frame_idx)

            prev_idx = 0 if i == last_idx else self.backward_indices[i + 1]
            T, inv_T = self.predict_pose(
                inputs['color', prev_idx, 0], inputs['color', frame_idx, 0])
            outputs['T', frame_idx] = T   # such as 0 -> 1
            outputs['inv_T', frame_idx] = inv_T  # such as 1 -> 0

        # 1 -> 0
        self.update_occupancy(inputs, outputs, frame_idx, 0)

        return outputs

    @torch.no_grad()
    def predict_occupancy(
            self,
            inputs: dict,
            outputs: dict,
            frame_idx: int):
        K = inputs['K', 0]
        inv_K = inputs['inv_K', 0]

        disp = outputs['disp', frame_idx]
        depth = self.disp_to_depth(disp)
        cam_points = self.backproject(depth, inv_K)
        cam_points = cam_points[:, :3, :]

        # Normalize with mean inversed depth
        norm_disp, mean_scale = normalize_with_mean(disp)
        norm_depth = self.disp_to_depth(norm_disp)

        cam_points = transform_point_cloud(self.R_y_invert, cam_points)
        plane, inlier_mask = self.planefit(cam_points, K)
        R_flat = compute_gflat_rotmat(plane[:, :3])

        plane_depth = self.plane2depth(plane, inv_K)
        outputs['plane_depth', frame_idx] = plane_depth
        plane_disp = self.depth_to_disp(plane_depth)

        # Compute abs-rel to make sampling masks
        norm_plane_disp = plane_disp * mean_scale
        ground_error = (norm_plane_disp - norm_disp).abs() / norm_disp
        sampling_mask = ground_error < self.ogm_sampling_mask_offset
        outputs['sampling_mask', frame_idx] = sampling_mask

        # Align point clouds to plane.
        plane_points = transform_point_cloud(R_flat, cam_points)
        T_cam2plane = torch.bmm(R_flat, self.R_y_invert)

        ogm = RelativeOGM(
            self.batch_size, self.ogm_size,
            dtype=plane_points.dtype, device=plane_points.device)
        ogm.setup_resolution(plane_points, inlier_mask, self.ogm_median_scale)

        ogm_points = transform_point_cloud(ogm.pose, plane_points)
        T_cam2ogm = torch.bmm(ogm.pose, T_cam2plane)
        T_ogm2cam = T_cam2ogm.inverse()
        outputs['T_cam2ogm', frame_idx] = T_cam2ogm
        outputs['T_ogm2cam', frame_idx] = T_ogm2cam

        self.invmodel.store_occupancy(
            ogm, norm_depth, inv_K, ogm_points, inlier_mask)

        self.multiply_confidence(ogm, plane_disp, K, T_ogm2cam)

        outputs['OGM', frame_idx] = ogm

        return outputs

    @torch.no_grad()
    def update_occupancy(
            self,
            inputs: dict,
            outputs: dict,
            src_frame_idx: int,
            tgt_frame_idx: int):
        """
        """
        K = inputs['K', 0]
        inv_K = inputs['inv_K', 0]

        src_ogm = outputs['OGM', src_frame_idx]
        src_plane_depth = outputs['plane_depth', src_frame_idx]
        src_T_cam2ogm = outputs['T_cam2ogm', src_frame_idx]
        T_tgt2src = outputs['T', src_frame_idx]

        tgt_ogm = outputs['OGM', tgt_frame_idx]
        tgt_T_ogm2cam = outputs['T_ogm2cam', tgt_frame_idx]

        # Generate sampling points in the target cell to update.
        ogm_points, _ = tgt_ogm.create_grid_coords(self.ogm_num_subgrids)
        ogm_shape_sub = (tgt_ogm.size, self.ogm_num_subgrids) * 2

        # Transform these points into the original camera coordinates(OGM->cam)
        cam_points = transform_point_cloud(tgt_T_ogm2cam, ogm_points)

        # Warp the 3d points from previous frame to the current frame
        # and projects the points into the image coordinates.
        img_points = project_to_2d(cam_points, K, T_tgt2src)

        sampling_indices, img_range_mask = \
            convert_to_sampler(img_points, self.height, self.width)

        # Only consider points located on the ground and succeeded projection.
        subcell_mask = img_range_mask
        subcell_mask = subcell_mask.view(-1, 1, *ogm_shape_sub)

        # The number of the warped into the ground from the first cells.
        subcell_num = subcell_mask.sum(dim=5).sum(dim=3)

        # Samples the depth values in 2D coordinates because the
        # self-supervised depth does not gurantee scale-consistency.
        src_depth = batch_masked_sampling(
            src_plane_depth.view(self.batch_size, -1),
            sampling_indices, img_range_mask)
        src_depth = src_depth.view(-1, 1, *ogm_shape_sub)

        # Transform the image points at t+1 into the OGM points at t+1.
        src_cam_points = project_to_3d(img_points, src_depth, inv_K)
        src_ogm_points = transform_point_cloud(src_T_cam2ogm, src_cam_points)
        src_ogm_points = src_ogm_points.view(-1, 3, *ogm_shape_sub)

        # The warped cells are computed as the mean position on grounds.
        src_ogm_points *= subcell_mask.view(-1, 1, *ogm_shape_sub)
        src_ogm_points = src_ogm_points.sum(dim=5).sum(dim=3)
        src_ogm_points /= subcell_num.clamp(min=1)

        src_sample_points = torch.index_select(
            src_ogm_points, 1,
            torch.tensor([0, 2], device=src_ogm_points.device))
        src_sample_grids, src_sample_mask = \
            convert_to_sampler(src_sample_points, *src_ogm.shape[-2:])

        # For the out of range, set zero odds.
        warped_odds = batch_masked_sampling(
            src_ogm.data.view(self.batch_size, -1),
            src_sample_grids.view(self.batch_size, -1),
            src_sample_mask.view(self.batch_size, -1))
        warped_odds = warped_odds.view(*tgt_ogm.shape)

        new_odds = torch.full_like(warped_odds, self.ogm_update_prior_odds)
        cell_mask = subcell_num > 0
        new_odds[cell_mask] += warped_odds[cell_mask]
        new_odds[~inputs['valid_indices', src_frame_idx]] = 0

        # Update occpancy grid map with odds theorem
        tgt_ogm.data += new_odds
