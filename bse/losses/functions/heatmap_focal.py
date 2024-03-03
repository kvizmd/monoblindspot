import torch
from torch import nn


class HeatmapFocalLoss(nn.Module):
    """
    The gaussian heatmap focal loss proposed in CornerNet/CenterNet.
    """

    def __init__(
            self,
            alpha=2.0,
            beta=4.0,
            focal_alpha=-1,
            sigmoid_clamp=1e-4,
            positive_threshold=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_alpha = focal_alpha
        self.sigmoid_clamp = sigmoid_clamp
        self.threshold = positive_threshold

    def forward(self, pred, target, weight=None, eps=1e-6):
        pred = torch.clamp(
            pred,
            min=self.sigmoid_clamp,
            max=1 - self.sigmoid_clamp)

        pmask = target.ge(self.threshold).float()
        nmask = target.lt(self.threshold).float()

        ploss = \
            -torch.log(pred + eps) \
            * torch.pow(1 - pred, self.alpha) \
            * target \
            * pmask
        nloss = \
            -torch.log(1 - pred + eps) \
            * torch.pow(pred, self.alpha) \
            * torch.pow(1 - target, self.beta) \
            * nmask

        if self.focal_alpha >= 0:
            ploss *= self.focal_alpha
            nloss *= 1 - self.focal_alpha

        loss = ploss + nloss

        if weight is not None:
            loss *= weight

        pnum = pmask.sum().float()
        loss = loss.sum() / pnum.clamp(min=1.0)
        return loss
