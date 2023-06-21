import torch
from torch.nn import functional as F
import mlflow

from bse.utils import \
    BackprojectDepth, \
    Project3D, \
    compute_eigen_depth_errors, \
    transformation_from_parameters, \
    disp_to_depth

from .trainer import Trainer
from .figure import make_depth_figure


class DepthTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.min_depth = self.cfg.MODEL.DEPTH.MIN_DEPTH
        self.max_depth = self.cfg.MODEL.DEPTH.MAX_DEPTH
        self.scales = self.cfg.MODEL.DEPTH.SCALES
        self.frame_idxs = self.cfg.DATA.FRAME_IDXS

        self.backproject = BackprojectDepth(
            self.batch_size, self.height, self.width)
        self.backproject.to(self.device)

        self.project = Project3D(
            self.batch_size, self.height, self.width)
        self.project.to(self.device)

    def build_optimizer(self, cfg, models: dict) -> torch.optim.Optimizer:
        param_lr_table = []

        param_lr_table += [{
            'params': models['depth'].parameters(),
            'lr': cfg.TRAIN.DEPTH.LR,
        }]

        param_lr_table += [{
            'params': models['pose'].parameters(),
            'lr': cfg.TRAIN.POSE.LR,
        }]

        optim = torch.optim.AdamW(
            param_lr_table,
            cfg.TRAIN.DEFAULT_LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        return optim

    def process_batch(self, inputs: dict) -> dict:
        outputs = {}
        for i, frame_idx in enumerate(self.frame_idxs):
            self.predict_depth(inputs, outputs, frame_idx)

            if frame_idx != 0:
                self.predict_adjacent_pose(inputs, outputs, frame_idx)

                for scale in self.scales:
                    self.warp_forward(inputs, outputs, frame_idx, scale)

        return outputs

    def predict_depth(
            self,
            inputs: dict,
            outputs: dict,
            frame_idx: int):
        depth_outputs = self.models['depth'](inputs['color_aug', frame_idx, 0])

        for s in self.scales:
            outputs['disp', frame_idx, s] = depth_outputs['disp', s]

    def predict_adjacent_pose(
            self,
            inputs: dict,
            outputs: dict,
            frame_idx: int):
        if frame_idx < 0:
            pose_inputs = [
                inputs['color_aug', frame_idx, 0],
                inputs['color_aug', 0, 0]
            ]

        else:
            pose_inputs = [
                inputs['color_aug', 0, 0],
                inputs['color_aug', frame_idx, 0]
            ]

        pose_inputs = torch.cat(pose_inputs, 1)
        axisangle, translation = self.models['pose'](pose_inputs)

        outputs['T', frame_idx, 0] = transformation_from_parameters(
            axisangle, translation, invert=frame_idx < 0)

    def warp_forward(
            self,
            inputs: dict,
            outputs: dict,
            frame_idx: int,
            scale: int):
        """
        It computes the correspondence from the target to the source.
        """
        if frame_idx == 0:
            return

        disp = F.interpolate(
            outputs['disp', 0, scale],
            (self.height, self.width), mode='bilinear', align_corners=False)
        _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

        cam_points = self.backproject(depth, inputs['inv_K', 0])
        pix_coords = self.project(
            cam_points, inputs['K', 0], outputs['T', frame_idx, 0])

        outputs['sample', frame_idx, scale] = pix_coords

        outputs['color', frame_idx, scale] = F.grid_sample(
            inputs['color', frame_idx, 0],
            outputs['sample', frame_idx, scale],
            padding_mode='border', align_corners=True)

    def is_available_gt(self, inputs: dict) -> bool:
        return ('depth_gt', 0) in inputs

    def evaluate(self, inputs: dict, outputs: dict) -> dict:
        gt_depth = inputs['depth_gt', 0].to(self.device)
        _, pred_depth = disp_to_depth(
            outputs['disp', 0, 0], self.min_depth, self.max_depth)
        metrics = compute_eigen_depth_errors(pred_depth, gt_depth)
        return metrics

    def integrate_metrics(self, metrics: dict) -> dict:
        out_metrics = {}
        for key, vals in metrics.items():
            out_metrics[key] = vals.mean()
        return out_metrics

    def save_figure(self, inputs: dict, outputs: dict, basename: str):
        with make_depth_figure(inputs, outputs) as fig:
            mlflow.log_figure(fig, basename + '_depth.jpg')
