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


class SequentialOGMIntegrator(OGMIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # [1, 2, ...., 50]
        self.backward_indices.sort()

        # [-1, -2, ...., -5]
        self.forward_indices.sort()
        self.forward_indices.reverse()

        self.reuse_first_ogm_scale = True

    def integrate(self, inputs: dict) -> dict:
        outputs = {}

        outputs['disp', 0] = self.predict_depth(inputs['color', 0, 0])
        self.predict_occupancy(inputs, outputs, 0)

        sampler = self.create_sampler(inputs, outputs)
        # index: -1, -2
        for i, frame_idx in enumerate(self.forward_indices):
            outputs['disp', frame_idx] = \
                self.predict_depth(inputs['color', frame_idx, 0])
            self.predict_occupancy(inputs, outputs, frame_idx)

            next_idx = 0 if i == 0 else self.forward_indices[i - 1]
            inv_T, T = self.predict_pose(
                inputs['color', frame_idx, 0], inputs['color', next_idx, 0])
            outputs['T', frame_idx] = T   # such as 0 -> -1
            outputs['inv_T', frame_idx] = inv_T  # such as -1 -> 0

            sampler = self.update_occupancy(
                inputs, outputs, frame_idx, sampler)

        # index: 1, 2, ..., 50
        sampler = self.create_sampler(inputs, outputs)
        for i, frame_idx in enumerate(self.backward_indices):
            outputs['disp', frame_idx] = \
                self.predict_depth(inputs['color', frame_idx, 0])
            self.predict_occupancy(inputs, outputs, frame_idx)

            prev_idx = 0 if i == 0 else self.backward_indices[i - 1]
            T, inv_T = self.predict_pose(
                inputs['color', prev_idx, 0], inputs['color', frame_idx, 0])
            outputs['T', frame_idx] = T   # such as 0 -> 1
            outputs['inv_T', frame_idx] = inv_T  # such as 1 -> 0

            sampler = self.update_occupancy(
                inputs, outputs, frame_idx, sampler)

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

        # Backproject the image points using a predicted depth.
        depth = self.disp_to_depth(disp)
        cam_points = self.backproject(depth, inv_K)
        cam_points = cam_points[:, :3, :]

        # Normalize with mean inversed depth
        norm_disp, mean_scale = normalize_with_mean(disp)
        norm_depth = self.disp_to_depth(norm_disp)

        # Plane fitting with RANSAC.
        invcam_points = transform_point_cloud(self.R_y_invert, cam_points)
        plane, inlier_mask = self.planefit(invcam_points, K)
        R_flat = compute_gflat_rotmat(plane[:, :3])

        # Compute the depth map of the plane based on the camera model.
        plane_depth = self.plane2depth(plane, inv_K)
        outputs['plane_depth', frame_idx] = plane_depth
        plane_disp = self.depth_to_disp(plane_depth)

        # Compute abs-rel to make sampling masks, which is used to sample only
        # the visible points. To detect all points, including even invisible
        # areas, all should be masked True like torch.ones_like(sampling_mask).
        norm_plane_disp = plane_disp * mean_scale
        ground_error = (norm_plane_disp - norm_disp).abs() / norm_disp
        sampling_mask = ground_error < self.ogm_sampling_mask_offset
        outputs['sampling_mask', frame_idx] = sampling_mask

        # Align the point cloud to the plane to become the XZ plane is
        # parallel to the obtained plane.
        plane_points = transform_point_cloud(R_flat, invcam_points)
        T_cam2plane = torch.bmm(R_flat, self.R_y_invert)

        ogm = RelativeOGM(
            self.batch_size, self.ogm_size,
            dtype=plane_points.dtype, device=plane_points.device)

        if frame_idx != 0 and self.reuse_first_ogm_scale:
            ogm.setup_resolution(
                plane_points, inlier_mask, scale=outputs['OGM', 0].scale)
        else:
            ogm.setup_resolution(
                plane_points, inlier_mask, median_scale=self.ogm_median_scale)

        # Scale the point cloud into the OGM coordinates.
        ogm_points = transform_point_cloud(ogm.pose, plane_points)
        T_cam2ogm = torch.bmm(ogm.pose, T_cam2plane)
        T_ogm2cam = T_cam2ogm.inverse()
        outputs['T_cam2ogm', frame_idx] = T_cam2ogm
        outputs['T_ogm2cam', frame_idx] = T_ogm2cam

        # Computes the occupancy probability.
        self.invmodel.store_occupancy(
            ogm, norm_depth, inv_K, ogm_points, inlier_mask)

        self.multiply_confidence(ogm, plane_disp, K, T_ogm2cam)

        outputs['OGM', frame_idx] = ogm
        return outputs

    @torch.no_grad()
    def create_sampler(self, inputs: dict, outputs: dict) -> dict:
        ogm = outputs['OGM', 0]
        T_ogm2cam = outputs['T_ogm2cam', 0]

        # To setup the internal state for the target frame,
        # Generate sub grids of frame-0 for warping.
        ogm_sampler, _ = ogm.create_grid_coords(self.ogm_num_subgrids)
        cam_sampler = transform_point_cloud(T_ogm2cam, ogm_sampler)
        return cam_sampler

    @torch.no_grad()
    def update_occupancy(
            self,
            inputs: dict,
            outputs: dict,
            frame_idx: int,
            sampler: torch.Tensor) -> torch.Tensor:
        K = inputs['K', 0]
        inv_K = inputs['inv_K', 0]
        src_ogm = outputs['OGM', frame_idx]
        src_cam2ogm = outputs['T_cam2ogm', frame_idx]
        T_tgt2src = outputs['T', frame_idx]
        depth = outputs['plane_depth', frame_idx]
        tgt_ogm = outputs['OGM', 0]
        ogm_shape_sub = (tgt_ogm.size, self.ogm_num_subgrids) * 2

        # Warp the 3d points from previous frame to the current frame
        # and projects the points into the image coordinates.
        img_points = project_to_2d(sampler, K, T_tgt2src)

        sampling_indices, img_range_mask = \
            convert_to_sampler(img_points, self.height, self.width)

        # Only consider points succeeded projection.
        subcell_mask = img_range_mask

        # Samples the depth values in 2D coordinates because the
        # self-supervised depth does not gurantee scale-consistency.
        src_depth = batch_masked_sampling(
            depth.view(self.batch_size, -1),
            sampling_indices, subcell_mask)
        src_depth = src_depth.view(-1, 1, *ogm_shape_sub)

        # Backproject the image points at t+1 into the camera points.
        sampler = project_to_3d(img_points, src_depth, inv_K)

        # Transform the camera points at t+1 into the OGM points at t+1.
        src_ogm_points = transform_point_cloud(src_cam2ogm, sampler)
        src_ogm_points = src_ogm_points.view(-1, 3, *ogm_shape_sub)

        # The number of the warped into the ground from the first cells.
        subcell_num = \
            subcell_mask.view(-1, 1, *ogm_shape_sub).sum(dim=5).sum(dim=3)
        cell_mask = subcell_num > 0

        # Compute the center point of warped points
        src_ogm_points *= subcell_mask.view(-1, 1, *ogm_shape_sub)
        src_ogm_points = src_ogm_points.sum(dim=5).sum(dim=3)
        src_ogm_points /= subcell_num.clamp(min=1)

        src_ogm_points = torch.index_select(
            src_ogm_points, 1,
            torch.tensor([0, 2], device=src_ogm_points.device))
        src_sample_grids, src_sample_mask = \
            convert_to_sampler(src_ogm_points, *src_ogm.shape[-2:])

        # Sampling the occpancy from the other OGM.
        warped_odds = batch_masked_sampling(
            src_ogm.data.view(self.batch_size, -1),
            src_sample_grids.view(self.batch_size, -1),
            src_sample_mask.view(self.batch_size, -1))
        warped_odds = warped_odds.view(*tgt_ogm.shape)

        # Update occpancy grid map with odds theorem
        new_odds = torch.full_like(warped_odds, self.ogm_update_prior_odds)
        new_odds[cell_mask] += warped_odds[cell_mask]

        # If a batch contains a scene for which no more frames can be
        # loaded, no update is performed for that dimension only.
        new_odds[~inputs['valid_indices', frame_idx]] = 0

        tgt_ogm.data += new_odds
        return sampler
