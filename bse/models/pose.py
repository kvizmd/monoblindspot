from .model import ModelBase
from .encoders import ResNetEncoder
from .depth_decoders import PoseDecoder


class PoseNet(ModelBase):
    def __init__(self, num_layers: int):
        super().__init__()
        self.encoder = ResNetEncoder(num_layers, True, num_input_images=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc)
