import torch
from torch.nn import functional as F

from bse.utils.diff import right_minus_left, left_minus_right
from .util import prob2logodds
from .ogm import OGM


class InverseSensorModel(torch.nn.Module):
    """
    Predict each occupancy probabilities which denotes,
    the cell is occupied with the high-gradient points,
    from the depth and point cloud.
    """

    def __init__(
            self,
            grad_thr: float = 0.01,
            count_thr: int = 3,
            min_prob: float = 0.1,
            max_prob: float = 0.9,
            free_prob: float = 1e-10,
            height_quantile: float = 0.7):
        """
        Args:
          grad_thr          : Threshold of the second-order depth gradient
          count_thr         : Threshold on the number of points at which a cell
                              is considered unknown
          min_prob          : Minimum occupancy probability of OGM
          max_prob          : Maximum occupancy probability of OGM
          free_prob         : Occpancy probability for free spaces
          height_quantile   : Upper ratio of height to be considered for
                              occupancy probability
        """

        super().__init__()

        # Although it would be possible to tune in more detail if different
        # thresholds were used for positive and negative, the same thresholds
        # are used for experimental purposes.
        self.p_grad_thr = grad_thr
        self.n_grad_thr = grad_thr

        self.p_count_thr = count_thr
        self.n_count_thr = count_thr

        self.p_min_prob = min_prob
        self.p_max_prob = max_prob
        self.n_max_prob = max_prob

        self.prior_free_prob = free_prob
        self.prior_free_odds = prob2logodds(self.prior_free_prob)

        self.p_min_odds = prob2logodds(self.p_min_prob)
        self.p_max_odds = prob2logodds(self.p_max_prob)
        self.n_max_odds = prob2logodds(self.n_max_prob)

        self.height_quantile = height_quantile

    def store_occupancy(
            self,
            ogm: OGM,
            depth: torch.Tensor,
            inv_K: torch.Tensor,
            scaled_ptcloud: torch.Tensor,
            ground_mask: torch.Tensor):
        """
        Predict occupancies from the depth and the point cloud.

        Args:
          ogm               : The target occupancy grid map.
          depth             : The depth map to compute its gradient.
          inv_K             : The inverted intrinsic matrix.
          scaled_ptcloud    : the point cloud in the OGM coordinates.
          ground_mask       : inlier mask to be considered as ground.
        """
        # Computes the gradient of the depth.
        grad_pos, grad_neg = self.assign_score(depth)

        ogm_pos_prob, ogm_free_mask, ogm_unknown_mask = \
            self.compute_occupancy_probability(
                ogm, scaled_ptcloud, ground_mask, grad_pos,
                grad_thr=self.p_grad_thr, count_thr=self.p_count_thr)

        ogm_pos_odds = prob2logodds(ogm_pos_prob)
        ogm_pos_odds[ogm_free_mask] = self.prior_free_odds
        ogm_pos_odds[ogm_unknown_mask] = 0  # probability: 0.5
        ogm_pos_odds = torch.clamp(
            ogm_pos_odds, self.p_min_odds, self.p_max_odds)

        # The negative occupancy grid to suppress thin/short objects
        ogm_neg_prob, _, _ = self.compute_occupancy_probability(
            ogm, scaled_ptcloud, ground_mask, grad_neg,
            grad_thr=self.n_grad_thr, count_thr=self.n_count_thr)

        ogm_neg_odds = prob2logodds(ogm_neg_prob)
        ogm_neg_odds[ogm_free_mask | ogm_unknown_mask] = 0
        ogm_neg_odds = torch.clamp(ogm_neg_odds, 0, self.n_max_odds)

        # (Experimental)
        # Spreading negative occupancy with max-pooling downsampling is robust
        # to its resolution.
        ogm_neg_odds = F.max_pool2d(
            ogm_neg_odds, kernel_size=3, stride=1, padding=1)

        ogm.data = ogm_pos_odds - ogm_neg_odds

    def assign_score(self, depth: torch.Tensor) -> tuple:
        """
        Assign a score to the 3D point cloud based on the relative
        second-order gradient of depth.
        """

        rml_dx = torch.relu(right_minus_left(depth))
        lmr_dx = torch.relu(left_minus_right(depth))

        rml_ddx = torch.relu(-1 * right_minus_left(rml_dx, 2))
        lmr_ddx = torch.relu(-1 * left_minus_right(lmr_dx, 2))

        W = depth.shape[-1]
        l_pos = rml_ddx[..., :W // 2]
        l_neg = lmr_ddx[..., :W // 2]
        r_pos = lmr_ddx[..., W // 2:]
        r_neg = rml_ddx[..., W // 2:]

        grad_pos = torch.cat((l_pos, r_pos), dim=-1)
        grad_neg = torch.cat((l_neg, r_neg), dim=-1)

        return grad_pos, grad_neg

    @torch.no_grad()
    def compute_occupancy_probability(
            self,
            ogm: OGM,
            scaled_ptcloud: torch.Tensor,
            ground_mask: torch.Tensor,
            discontinuity: torch.Tensor,
            grad_thr: float,
            count_thr: int) -> tuple:
        """
        Computes the occpunacy probability from the quantized point cloud.
        """
        batch_size = scaled_ptcloud.shape[0]

        ogm_occ = torch.zeros_like(ogm.data.view(batch_size, -1))
        normalizer = ogm_occ.clone()
        free_mask = ogm_occ.clone().to(torch.bool)

        # TODO: Support batch-style
        for b in range(batch_size):
            xyz = scaled_ptcloud[b]
            score = discontinuity[b].view(-1)
            g_mask = ground_mask[b].view(-1)

            # Ignore out of range
            x_range_mask = (xyz[0] >= 0) & (xyz[0] <= ogm.size - 1)
            z_range_mask = (xyz[2] >= 0) & (xyz[2] <= ogm.size - 1)
            range_mask = x_range_mask & z_range_mask

            # Consider only 75% heights in the point cloud to ignore
            # discontinuity between buildings and skys
            height_mask = xyz[1] < xyz[1].quantile(self.height_quantile)
            valid_mask = height_mask & range_mask

            # The zero weights implies invalid points (e.g. avobe eye level)
            # since perfect zeros are rare as a result of the computer
            # calculation, so ignore these points
            obj_mask = valid_mask & (~g_mask) & (score > 0)
            no_obj_mask = valid_mask & g_mask

            # Compute the number of points belonging to each cell.
            object_cells = xyz[..., obj_mask].long()
            if object_cells.numel() > 0:
                # To apply scatter_add_, convert the shape to flatten.
                indices = object_cells[0] + ogm.size * object_cells[2]

                # Count the number of high discontinuity points
                valid_points = (score[obj_mask] > grad_thr).to(ogm_occ.dtype)
                ogm_occ[b] = ogm_occ[b].scatter_add_(0, indices, valid_points)

                # Count the number of points corresponding to indices.
                counts = torch.ones_like(valid_points)
                normalizer[b] = normalizer[b].scatter_add_(0, indices, counts)

            ground_cells = xyz[..., no_obj_mask].long()
            if ground_cells.numel() > 0:
                # If there is no occupancy count in the ground areas,
                # they are assumed as completely free.
                indices = ground_cells[0] + ogm.size * ground_cells[2]
                free_mask[b, indices] = normalizer[b, indices] == 0

        # Unreliable cells with insufficient counts are considered unknown.
        unknown_mask = (~free_mask) & (normalizer < count_thr)

        # Computes the occupancy probability
        ogm_occ /= normalizer.clamp(min=1)

        # the OGM_occ has only occupied probabilities.
        ogm_occ[unknown_mask | free_mask] = 0.5

        ogm_occ = ogm_occ.view(*ogm.shape)
        free_mask = free_mask.view(*ogm.shape)
        unknown_mask = unknown_mask.view(*ogm.shape)

        return ogm_occ, free_mask, unknown_mask
