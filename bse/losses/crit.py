from collections import defaultdict

from torch import nn


class Criterion(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def compute_losses(self, inputs, outputs, losses):
        raise NotImplementedError()

    def forward(self, inputs, outputs):
        losses = defaultdict(int)
        if self.factor <= 0:
            return losses

        losses['loss'] = 0
        self.compute_losses(inputs, outputs, losses)

        losses['loss'] *= self.factor
        return losses
