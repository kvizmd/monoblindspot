import torch
from torch import nn
from torch.nn import functional as F

from .crit import Criterion
from .functions import FocalLoss
from .util import bipartite_match


class BipartCriterion(Criterion):
    def __init__(
            self,
            factor=1.0,
            cls_factor=1.0,
            pos_2d_factor=1.0,
            score_match=1.0,
            pos_match=1.0,
            occ_weighting=True,
            **kwargs):
        super().__init__(factor)

        self.factors = {
            'cls_loss': cls_factor,
            'pos_2d_loss': pos_2d_factor,
            'score_match': score_match,
            'pos_match': pos_match,
        }

        self.factor = factor

        self.cls_loss = FocalLoss(alpha=0.25, gamma=2)
        self.pos_2d_loss = nn.L1Loss(reduction='none')

        self.occ_weighting = occ_weighting

    def match(self, inputs, outputs):
        # Confidence score
        pred_score = outputs['bs_confidence', 0, 0]

        # 2D position
        pred_point = outputs['bs_point', 0, 0]
        gt_point = outputs['bs_point_gen', 0, 0]

        # Compute the number of prediction and ground-truth.
        factory_args = {'dtype': torch.int64, 'device': pred_score.device}
        gt_nums = [len(pts) for pts in gt_point]
        Ngt_all = torch.tensor(gt_nums).sum()
        if Ngt_all == 0:
            return [(
                torch.tensor([], **factory_args),
                torch.tensor([], **factory_args))
                for _ in range(pred_point.shape[0])]

        B, Npred = pred_score.shape

        def reshape_gt(xs):
            # [(Ngt_b x H x W), ... (Ngt_B x H x W)] -> (N_gt_all x H x W)
            xs = torch.cat([x for x in xs if x.numel() > 0])
            xs = xs.view(1, Ngt_all, -1).expand(B * Npred, -1, -1)
            return xs

        def reshape_pred(xs):
            return xs.view(B * Npred, 1, -1).expand(-1, Ngt_all, -1)

        # Prediction score
        pred_score = reshape_pred(pred_score)

        # 2D position error
        pred_point = reshape_pred(pred_point)
        gt_point = reshape_gt(gt_point)

        pos_err = self.pos_2d_loss(pred_point, gt_point).mean(-1, True)

        match_score = \
            pred_score ** self.factors['score_match'] \
            * (1 - pos_err) ** self.factors['pos_match']

        match_score = match_score.view(B, Npred, -1)
        return bipartite_match(match_score, gt_nums, maximize=True)

    def compute_losses(self, inputs, outputs, losses):
        Npred = outputs['bs_confidence', 0, 0].shape[1]
        bipart_indices = self.match(inputs, outputs)
        for b, (pred_indices, gt_indices) in enumerate(bipart_indices):
            pred_cls_score = outputs['bs_confidence', 0, 0][b]

            gt_num = gt_indices.numel()
            if gt_num == 0:
                gt_cls_score = torch.zeros(Npred).to(pred_cls_score)
                losses['cls_loss'] += 0.1 * self.cls_loss(
                    pred_cls_score, gt_cls_score.detach())
                continue

            gt_occ_score = outputs['bs_confidence_gen', 0, 0][b][gt_indices]

            # Instance loss
            gt_cls_score = F.one_hot(
                pred_indices, Npred).amax(0).to(pred_cls_score)
            cls_confidence = torch.ones_like(gt_cls_score)
            cls_confidence[pred_indices] = gt_occ_score
            losses['cls_loss'] += self.cls_loss(
                pred_cls_score, gt_cls_score.detach(),
                weight=cls_confidence if self.occ_weighting else None)

            # 2D position loss
            pred_point = outputs['bs_point', 0, 0][b][pred_indices]
            gt_point = outputs['bs_point_gen', 0, 0][b][gt_indices].detach()
            pos_2d_loss = self.pos_2d_loss(pred_point, gt_point).mean(-1)
            if self.occ_weighting:
                pos_2d_loss *= gt_occ_score
            losses['pos_2d_loss'] += pos_2d_loss.mean()

        losses['loss'] = 0
        for key in losses.keys():
            if key in self.factors:
                losses[key] *= self.factors[key]
                losses['loss'] += losses[key]
