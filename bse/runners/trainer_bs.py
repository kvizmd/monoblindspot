from collections import defaultdict

import numpy as np

import torch

from bse.utils.metric import \
    compute_binary_metrics, \
    evaluate_blindspot, \
    ignore_negative, \
    ignore_on_mask
from .trainer import Trainer


class BlindSpotTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depth_ranges = {
            'all': (None, None),
            'short': (
                self.cfg.EVAL.BS.SHORT_LOWER,
                self.cfg.EVAL.BS.SHORT_UPPER
            ),
            'middle': (
                self.cfg.EVAL.BS.MIDDLE_LOWER,
                self.cfg.EVAL.BS.MIDDLE_UPPER
            ),
            'long': (
                self.cfg.EVAL.BS.LONG_LOWER,
                self.cfg.EVAL.BS.LONG_UPPER
            ),
        }
        self.error_threshold = self.cfg.EVAL.BS.ERROR_THRESHOLD
        self.score_threshold = self.cfg.EVAL.BS.SCORE_THRESHOLD

    def is_available_gt(self, inputs: dict) -> bool:
        return 'bs_gt' in inputs

    def evaluate(self, inputs: dict, outputs: dict) -> dict:
        out_metrics = defaultdict(int)

        pred_points = outputs['bs_point', 0, 0]
        pred_scores = outputs['bs_confidence', 0, 0]
        masks = pred_scores >= self.score_threshold
        masks = masks.view(self.batch_size, -1)

        gt_points = inputs['bs_gt']
        ignore_mask = inputs['bs_ignore']
        for b in range(self.batch_size):
            gt = ignore_negative(gt_points[b])
            gt = ignore_on_mask(gt, ignore_mask[b])

            pred = ignore_on_mask(pred_points[b, masks[b]], ignore_mask[b])

            ground_depth = inputs['ground_depth'][b]
            inv_K = inputs['inv_K_calib'][b]
            metrics = evaluate_blindspot(
                pred, gt, ground_depth, inv_K,
                depth_ranges=self.depth_ranges,
                threshold=self.error_threshold)

            for key, val in metrics.items():
                out_metrics[key] += val

        return out_metrics

    def integrate_metrics(self, all_sample_metrics: dict) -> dict:
        out_metrics = {}
        for key in self.depth_ranges.keys():
            TP = np.asarray(all_sample_metrics[key + '/TP']).sum()
            FP = np.asarray(all_sample_metrics[key + '/FP']).sum()
            FN = np.asarray(all_sample_metrics[key + '/FN']).sum()

            metrics = compute_binary_metrics(TP, FP, FN)
            for _k, _v in metrics.items():
                out_metrics[key + '/' + _k] = _v

        return out_metrics

    def load_offline_label(self, inputs: dict, outputs: dict):
        outputs['bs_point_gen', 0, 0] = []
        outputs['bs_confidence_gen', 0, 0] = []

        for b in range(self.batch_size):
            bs_gt = inputs['bs_gt'][b]
            bs_gt = ignore_negative(bs_gt)

            if bs_gt.shape[-1] == 3:
                point = bs_gt[..., :2]
                score = bs_gt[..., 2]

            else:
                point = bs_gt
                score = torch.ones_like(bs_gt[..., 0])

            outputs['bs_point_gen', 0, 0].append(point)
            outputs['bs_confidence_gen', 0, 0].append(score)
