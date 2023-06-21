import os

import numpy as np
import cv2
import mlflow

from bse.utils import compute_depth_errors, disp_to_depth

from .evaluator import Evaluator
from .figure import make_depth_figure


class DepthEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.min_depth = 1e-3
        self.max_depth = 80

        self.use_eigen_crop = 'eigen' in self.cfg.DATA.TEST_SPLIT
        if self.use_eigen_crop:
            print('Enable eigen cropping')

        gt_path = os.path.join(
            os.path.dirname(self.cfg.DATA.TEST_SPLIT), 'gt_depths.npz')
        self.gt_depths = np.load(
            gt_path, fix_imports=True,
            encoding='latin1', allow_pickle=True)['data']

    def predict(self, inputs):
        return self.model(inputs['color', 0, 0])

    def evaluate(self, inputs, outputs, iter_idx):
        gt_depth = self.gt_depths[iter_idx]
        gt_height, gt_width = gt_depth.shape[:2]

        scaled_disp, _ = disp_to_depth(
            outputs['disp', 0, 0], self.min_depth, self.max_depth)
        pred_disp = scaled_disp[0, 0].cpu().numpy()

        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if self.use_eigen_crop:
            pred_depth, gt_depth = self.apply_eigen_crop(pred_depth, gt_depth)

        else:
            mask = gt_depth > 0
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

        metrics = {}

        pred_depth, ratio = self.median_scaling(pred_depth, gt_depth)
        metrics['ratio'] = ratio

        errors = compute_depth_errors(gt_depth, pred_depth)
        metrics.update(errors)

        return metrics

    def integrate(self, all_sample_metrics):
        metrics = {}
        for key, vals in all_sample_metrics.items():
            if key == 'ratio':
                metrics['scaling_ratio_mean'] = vals.mean()
                med = np.median(vals)
                metrics['scaling_ratio_med'] = med
                metrics['scaling_ratio_std'] = np.std(vals / med)
            else:
                metrics[key] = vals.mean()

        return metrics

    def apply_eigen_crop(self, pred_depth, gt_depth):
        gt_height, gt_width = gt_depth.shape[:2]
        mask = np.logical_and(
            gt_depth > self.min_depth, gt_depth < self.max_depth)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width])
        crop = crop.astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        return pred_depth, gt_depth

    def median_scaling(self, pred_depth, gt_depth):
        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= ratio

        pred_depth[pred_depth < self.min_depth] = self.min_depth
        pred_depth[pred_depth > self.max_depth] = self.max_depth
        return pred_depth, ratio

    def save_figure(self, inputs: dict, outputs: dict, basename: str):
        with make_depth_figure(inputs, outputs) as fig:
            mlflow.log_figure(fig, basename + '_depth.jpg')
