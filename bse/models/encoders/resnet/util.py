import torch
import torch.nn as nn
import torchvision.models as models


class ResNetMultiImageInput(models.ResNet):
    """
    Constructs a resnet model with varying number of input images.
    Adapted from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(
            self,
            block: int,
            layers: list,
            num_input_images: int = 1):
        super().__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def get_reset_weight(num_layers, weights):
    return {
        18: models.ResNet18_Weights,
        34: models.ResNet34_Weights,
        50: models.ResNet50_Weights,
        101: models.ResNet101_Weights,
        152: models.ResNet152_Weights}[num_layers].verify(weights)


def resnet_multiimage_input(
        num_layers: int,
        weights: str = None,
        num_input_images: int = 1) -> nn.Module:
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {
        18: [2, 2, 2, 2],
        50: [3, 4, 6, 3]
    }[num_layers]

    block_type = {
        18: models.resnet.BasicBlock,
        50: models.resnet.Bottleneck
    }[num_layers]

    model = ResNetMultiImageInput(
        block_type, blocks, num_input_images=num_input_images)

    if weights is not None:
        weights = get_reset_weight(num_layers, weights)
        loaded = weights.get_state_dict(progress=True)

        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model
