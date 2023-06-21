from .model import ModelBase
from .encoders import ResNetEncoder
from .depth_decoders import DepthDecoder


class DepthNet(ModelBase):
    def __init__(self, num_layers, scales=range(4)):
        super().__init__()
        self.encoder = ResNetEncoder(num_layers, True)
        self.decoder = DepthDecoder(
            self.encoder.num_ch_enc, scales=scales)
