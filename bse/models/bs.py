from .model import ModelBase


class BSNetBase(ModelBase):
    def __init__(
            self,
            encoder,
            decoder,
            num_layers: int,
            inst_num: int = 32,
            prior_prob: float = 0.1,
            down_ratio: int = 8,
            group_num: int = 1,
            group_norm: bool = False,
            deform_conv: bool = False):
        super().__init__()

        self.encoder = encoder(
            num_layers,
            pretrained=True,
            down_ratio=down_ratio,
            group_norm=group_norm,
            ppm_channels=256,
            deform_conv=deform_conv)

        self.decoder = decoder(
            self.encoder.out_channels,
            head_channels=256,
            inst_num=inst_num,
            prior_prob=prior_prob,
            group_num=group_num)
