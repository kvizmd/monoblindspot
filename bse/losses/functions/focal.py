import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(
            self,
            alpha=-1,
            gamma=2,
            reduction='instance'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.reduction = reduction.lower()

    def forward(self, pred, target, weight=None):
        ploss = \
            -target \
            * torch.pow(1 - pred, self.gamma) \
            * torch.log(pred + 1e-10)

        nloss = \
            -(1 - target) \
            * torch.pow(pred, self.gamma) \
            * torch.log(1 - pred + 1e-10)

        if self.alpha >= 0:
            loss = self.alpha * ploss + (1 - self.alpha) * nloss
        else:
            loss = ploss + nloss

        if weight is not None:
            loss *= weight

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'instance':
            pnum = target.ge(1).sum()
            return loss.sum() / pnum.clamp(min=1)
        return loss
