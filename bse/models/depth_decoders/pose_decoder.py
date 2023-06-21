import torch.nn as nn


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc: list, channels: int = 256):
        super().__init__()

        self.squeeze = nn.Conv2d(num_ch_enc[-1], channels, 1)

        self.conv1 = nn.Conv2d(
            channels, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, 6, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_feature: list) -> tuple:
        x = self.relu(self.squeeze(input_feature[-1]))

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        out = self.conv3(x).mean(3).mean(2)
        out = 0.01 * out.view(-1, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
