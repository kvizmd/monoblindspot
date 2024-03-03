import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from .util import resnet_multiimage_input


class ResNetEncoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            pretrained: bool,
            num_input_images: int = 1,
            **kwargs):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers))

        weights = 'IMAGENET1K_V1' if pretrained else None
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(
                num_layers, weights, num_input_images)
        else:
            self.encoder = resnets[num_layers](weights=weights)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.out_channels = self.num_ch_enc[-1]

    def forward(self, input_image: torch.Tensor) -> tuple:
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        f1 = self.encoder.relu(x)
        f2 = self.encoder.layer1(self.encoder.maxpool(f1))
        f3 = self.encoder.layer2(f2)
        f4 = self.encoder.layer3(f3)
        f5 = self.encoder.layer4(f4)
        return [f1, f2, f3, f4, f5]
