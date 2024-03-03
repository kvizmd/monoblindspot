from collections import defaultdict

import numpy as np
import mlflow

from bse.utils.metric import \
    compute_binary_metrics, \
    ignore_negative, \
    ignore_on_mask, \
    evaluate_blindspot

from .evaluator import Evaluator
from .figure import \
    make_bsgen_figure, \
    make_bs_figure


class BlindSpotEvaluator(Evaluator):
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

        self.data_dir = self.dataloader.dataset.data_path

        self.get_model_args = self.build_model_feeder()
        self.make_figure = self.build_figure_maker()

    def build_model_feeder(self):
        if 'gen' in self.cfg.TARGET_MODE.lower():
            return lambda inputs: (inputs,)
        return lambda inputs: (inputs['color', 0, 0],)

    def build_figure_maker(self):
        if 'gen' in self.cfg.TARGET_MODE.lower():
            return make_bsgen_figure

        return make_bs_figure

    def predict(self, inputs: dict) -> dict:
        return self.model(*self.get_model_args(inputs))

    def save_figure(self, inputs: dict, outputs: dict, basename: str):
        with self.make_figure(inputs, outputs, score_thr=0.1) as fig:
            mlflow.log_figure(
                fig, basename + '_{}.jpg'.format(self.ml_run.info.run_id))

    def evaluate(self, inputs: dict, outputs: dict, iter_idx: int) -> dict:
        out_metrics = defaultdict(int)

        if 'bs_gt' not in inputs:
            print(
                'Warning: \"' + inputs['filename'][0]
                + '\" does not have annotation!!')
            return out_metrics

        for b in range(self.batch_size):
            gt_point = inputs['bs_gt'][b]  # shape: N x 2
            bs_ignore_mask = inputs['bs_ignore'][b]  # shape: 1 x H x W
            ground_depth = inputs['ground_depth'][b]
            inv_K = inputs['inv_K_calib'][b]
            pred_score = outputs['bs_confidence', 0, 0][b]  # shape: N
            pred_point = outputs['bs_point', 0, 0][b]  # shape: N x 2

            # Extract instances having high confidence
            high_conf_mask = pred_score.view(-1) >= self.score_threshold
            pred_score = pred_score[high_conf_mask]
            pred_point = pred_point[high_conf_mask]

            gt_point = ignore_negative(gt_point)
            gt_point = ignore_on_mask(gt_point, bs_ignore_mask)

            pred_point = ignore_on_mask(pred_point, bs_ignore_mask)

            metrics = evaluate_blindspot(
                pred_point, gt_point, ground_depth, inv_K,
                depth_ranges=self.depth_ranges,
                threshold=self.error_threshold)

            for key, val in metrics.items():
                out_metrics[key] += val

        return out_metrics

    def integrate(self, all_sample_metrics: dict) -> dict:
        out_metrics = {}
        for key in self.depth_ranges.keys():
            TP = all_sample_metrics.get(key + '/TP', [0])
            TP = np.asarray(TP).sum()

            FP = all_sample_metrics.get(key + '/FP', [0])
            FP = np.asarray(FP).sum()

            FN = all_sample_metrics.get(key + '/FN', [0])
            FN = np.asarray(FN).sum()

            metrics = compute_binary_metrics(TP, FP, FN)
            for _k, _v in metrics.items():
                out_metrics[key + '/' + _k] = _v

        return out_metrics
