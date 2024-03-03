import torch
import mlflow

from .trainer_bs import BlindSpotTrainer
from .figure import make_bs_figure


class BlindSpot2DTrainer(BlindSpotTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_optimizer(self, cfg, models: dict) -> torch.optim.Optimizer:
        param_lr_table = [
            {
                'params': self.get_model_module('bs').encoder.parameters(),
                'lr': cfg.TRAIN.BS.ENCODER_LR,
            },
            {
                'params': self.get_model_module('bs').decoder.parameters(),
                'lr': cfg.TRAIN.BS.DECODER_LR,
            }
        ]

        optim = torch.optim.AdamW(
            param_lr_table,
            cfg.TRAIN.DEFAULT_LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        return optim

    def process_batch(self, inputs: dict) -> dict:
        outputs = {}

        bs_outputs = self.models['bs'](inputs['color_aug', 0, 0])
        for key in bs_outputs.keys():
            new_key = [key[0]] + [0] + list(key[1:])
            outputs[tuple(new_key)] = bs_outputs[key]

        return outputs

    def save_figure(self, inputs: dict, outputs: dict, basename: str):
        with make_bs_figure(inputs, outputs) as fig:
            mlflow.log_figure(fig, basename + '_bs.jpg')
