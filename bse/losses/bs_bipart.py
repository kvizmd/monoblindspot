import torch
from torch import nn
from torch.nn import functional as F

from .crit import Criterion
from .functions import FocalLoss
from .util import bipartite_match


class BipartCriterion(Criterion):
    def __init__(
            self,
            factor: float = 1.0,
            cls_factor: float = 1.0,
            pos_2d_factor: float = 1.0,
            score_match: float = 1.0,
            pos_match: float = 1.0,
            occ_weighting: bool = True,
            **kwargs):
        super().__init__(factor)

        self.factors = {
            'cls_loss': cls_factor,
            'pos_2d_loss': pos_2d_factor,
            'score_match': score_match,
            'pos_match': pos_match,
        }

        self.factor = factor

        self.focal_alpha = 0.25
        self.focal_gamma = 2
        self.cls_loss = FocalLoss(
            alpha=self.focal_alpha, gamma=self.focal_gamma)
        self.pos_2d_loss = nn.L1Loss(reduction='none')

        self.occ_weighting = occ_weighting

    def match(self, inputs: dict, outputs: dict, s_idx: int = 0):
        # Confidence score
        pred_score = outputs['bs_confidence', 0, s_idx]

        # 2D position
        pred_point = outputs['bs_point', 0, s_idx]
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

        # match_cost = self._compute_matching_costs_detr(
        #    pred_score, pred_point, gt_point)
        match_cost = self._compute_matching_costs_sparseinst(
            pred_score, pred_point, gt_point)
        match_cost = match_cost.view(B, Npred, -1)
        return bipartite_match(match_cost, gt_nums, maximize=False)

    def _compute_matching_costs_sparseinst(
            self,
            pred_score,
            pred_point_2d,
            gt_point_2d):
        pos_cost_2d = self.pos_2d_loss(
            pred_point_2d, gt_point_2d).mean(-1, True)

        match_score = \
            pred_score ** self.factors['score_match'] \
            * (1 - pos_cost_2d).clamp(min=0) ** self.factors['pos_match']

        return -match_score

    def _compute_matching_costs_detr(
            self,
            pred_score,
            pred_point_2d,
            gt_point_2d):

        pos_score_cost = \
            self.focal_alpha \
            * ((1 - pred_score) ** self.focal_gamma) \
            * (-torch.log(pred_score + 1e-8))
        neg_score_cost = \
            (1 - self.focal_alpha) \
            * (pred_score ** self.focal_gamma) \
            * (-torch.log(1 - pred_score + 1e-8))
        score_cost = pos_score_cost - neg_score_cost

        pos_cost_2d = self.pos_2d_loss(
            pred_point_2d, gt_point_2d).sum(-1, True)

        match_cost = \
            self.factors['cls_loss'] * score_cost \
            + self.factors['pos_2d_loss'] * pos_cost_2d

        return match_cost

    def compute_losses(self, inputs, outputs, losses):
        s_indices = []
        for key in outputs.keys():
            if key[0] == 'bs_confidence':
                s_indices.append(int(key[-1]))

        for s_idx in s_indices:
            s_confidence = 1 / 2 ** s_idx

            Npred = outputs['bs_confidence', 0, s_idx].shape[1]

            bipart_indices = self.match(inputs, outputs)

            for b, (pred_indices, gt_indices) in enumerate(bipart_indices):
                pred_cls_score = outputs['bs_confidence', 0, s_idx][b]

                gt_num = gt_indices.numel()
                if gt_num == 0:
                    gt_cls_score = torch.zeros(Npred).to(pred_cls_score)
                    losses['cls_loss'] += s_confidence * 0.5 * self.cls_loss(
                        pred_cls_score, gt_cls_score.detach())
                    continue

                gt_occ_score = \
                    outputs['bs_confidence_gen', 0, 0][b][gt_indices]

                # Instance loss
                gt_cls_score = F.one_hot(
                    pred_indices, Npred).amax(0).to(pred_cls_score)
                cls_confidence = torch.full_like(gt_cls_score, 0.5)  # Unknown
                cls_confidence[pred_indices] = gt_occ_score
                losses['cls_loss'] += s_confidence * self.cls_loss(
                    pred_cls_score, gt_cls_score.detach(),
                    weight=cls_confidence if self.occ_weighting else None)

                # 2D position loss
                pred_point = \
                    outputs['bs_point', 0, s_idx][b][pred_indices]
                gt_point = \
                    outputs['bs_point_gen', 0, 0][b][gt_indices].detach()
                pos_2d_loss = self.pos_2d_loss(pred_point, gt_point).sum(-1)
                if self.occ_weighting:
                    pos_2d_loss *= gt_occ_score
                pos_2d_loss = pos_2d_loss.sum() / gt_num
                losses['pos_2d_loss'] += s_confidence * pos_2d_loss

        losses['loss'] = 0
        for key in losses.keys():
            if key in self.factors:
                losses[key] *= self.factors[key]
                losses['loss'] += losses[key]
