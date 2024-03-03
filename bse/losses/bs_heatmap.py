import torch
from torch import nn

from bse.utils import create_heatmap, create_vp_offset_map

from .crit import Criterion
from .functions import HeatmapFocalLoss


class HeatmapCriterion(Criterion):
    def __init__(
            self,
            factor: float = 1.0,
            cls_factor: float = 1.0,
            offset_factor: float = 1.0,
            heatmap_radius: float = 1.0,
            occ_weighting: bool = True,
            **kwargs):
        super().__init__(factor)

        self.factors = {
            'hm_loss': cls_factor,
            'offset_loss': offset_factor
        }

        self.hm_loss = HeatmapFocalLoss(alpha=2.0, beta=4.0)
        self.offset_loss = nn.L1Loss(reduction='none')

        self.radius = heatmap_radius
        self.occ_weighting = occ_weighting

    def compute_losses(self, inputs, outputs, losses):
        gt_hm, gt_offset, gt_conf = self.generate_gt_heatmap(inputs, outputs)

        pred_hm = outputs['pred_hm', 0, 0]
        hm_loss = self.hm_loss(
            pred_hm, gt_hm,
            weight=gt_conf if self.occ_weighting else None)
        losses['hm_loss'] = hm_loss

        pred_offset = outputs['pred_offset', 0, 0]
        inst_mask = gt_hm == 1.0
        offset_loss = self.offset_loss(pred_offset, gt_offset)
        offset_loss = inst_mask * offset_loss
        if self.occ_weighting:
            offset_loss *= gt_conf

        # Normalize with the number of instances.
        inst_num = inst_mask.sum()
        inst_num[inst_num == 0] = 1  # Avoid zero division
        offset_loss = offset_loss.sum() / inst_num.clamp(min=1)
        losses['offset_loss'] = offset_loss

        losses['loss'] = 0
        for key, factor in self.factors.items():
            losses[key] *= factor
            losses['loss'] += losses[key]

    def get_radius_map(self, RH, RW):
        vp_offset = create_vp_offset_map(RH // 2, (RH, RW))
        vp_offset *= self.radius
        return vp_offset.view(RH, RW).clamp(min=1)

    @torch.no_grad()
    def generate_gt_heatmap(self, inputs, outputs):
        gt_points = outputs['bs_point_gen', 0, 0]
        gt_probs = outputs['bs_confidence_gen', 0, 0]

        _, _, H, W = inputs['color', 0, 0].shape

        pred_hm = outputs['pred_hm', 0, 0]
        B, _, RH, RW = pred_hm.shape
        factory_args = {
            'device': pred_hm.device,
            'dtype': pred_hm.dtype,
        }

        gt_hm = torch.zeros((B, 1, RH, RW), **factory_args)
        gt_offset = torch.zeros((B, RH, RW, 2), **factory_args)
        gt_confidence = torch.full_like(gt_hm, 0.5)

        r_map = self.get_radius_map(RH, RW)
        for b, (points, probs) in enumerate(zip(gt_points, gt_probs)):
            if len(points) == 0:
                continue

            real_pts = points.clone()
            real_pts[:, 0] *= RH - 1
            real_pts[:, 1] *= RW - 1

            int_pts = real_pts.long()
            offsets = real_pts - int_pts

            for val, pt, offset in zip(probs, int_pts, offsets):
                hm = create_heatmap(gt_hm[b], pt, r_map[pt[0], pt[1]])
                gt_hm[b] = hm
                gt_confidence[b] = (hm > 0) * val
                gt_offset[b, pt[0], pt[1]] = offset

        gt_offset = gt_offset.permute(0, 3, 1, 2)
        return gt_hm.detach(), gt_offset.detach(), gt_confidence.detach()
