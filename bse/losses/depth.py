import torch

from .crit import Criterion
from .functions import ReprojectionLoss, SmoothnessLoss


class DepthCriterion(Criterion):
    def __init__(
            self,
            frame_idxs,
            scales,
            factor=1.0,
            smooth_scaling=True,
            smooth_factor=0.001):
        super().__init__(factor)
        self.frame_idxs = frame_idxs
        self.scales = scales
        self.smooth_scaling = smooth_scaling
        self.smooth_factor = smooth_factor

        self.reproj_loss = ReprojectionLoss()
        self.smooth_loss = SmoothnessLoss()

    def compute_losses(self, inputs, outputs, losses):
        for scale in self.scales:
            target = inputs['color', 0, 0]

            _reproj = []
            _identity = []
            for frame_idx in self.frame_idxs:
                if frame_idx == 0:
                    continue
                source = inputs['color', frame_idx, 0]
                pred = outputs['color', frame_idx, scale]

                # reconstruction loss
                _reproj.append(self.reproj_loss(pred, target))
                # identity_loss
                _identity.append(self.reproj_loss(source, target))
            _reproj = torch.cat(_reproj, 1)
            _identity = torch.cat(_identity, 1)

            combined = torch.cat((_identity, _reproj), 1)
            reproj_loss, reproj_indices = torch.min(combined, 1)
            outputs['auto_mask'] = reproj_indices > _identity.shape[1] - 1

            reproj_loss = reproj_loss.mean()
            losses['reproj_loss'] += reproj_loss
            losses['loss'] += reproj_loss

            if self.smooth_factor > 0:
                disp = outputs['disp', 0, scale]
                color = inputs[("color", 0, scale)]
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)

                smooth_loss = self.smooth_loss(norm_disp, color)
                smooth_loss = self.smooth_factor * smooth_loss
                if self.smooth_scaling:
                    smooth_loss /= 2 ** scale

                smooth_loss = smooth_loss.mean()
                losses[f'smooth_loss/{scale}'] += smooth_loss
                losses['loss'] += smooth_loss

        losses['loss'] /= len(self.scales)
