import torch
from torch import nn

from .ssim import SSIM


class ReprojectionLoss(nn.Module):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.ssim = SSIM()
        self.alpha = alpha

    def forward(self, pred, target):
        l1_loss = torch.abs(target - pred).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)

        reprojection_loss = \
            self.alpha * ssim_loss + (1 - self.alpha) * l1_loss

        return reprojection_loss
