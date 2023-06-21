from .depth import DepthNet
from .pose import PoseNet
from .bs import BSNetBase
from .encoders import \
    ResNetEncoder, \
    DLAEncoder, \
    SparseInstEncoder
from .bs_detectors import IAMDetector


def build_model(cfg):
    models = {}
    weights = {}

    if cfg.MODEL.DEPTH.ENABLED:
        models['depth'] = DepthNet(
            cfg.MODEL.DEPTH.NUM_LAYERS,
            scales=cfg.MODEL.DEPTH.SCALES)
        weights['depth'] = cfg.MODEL.DEPTH.WEIGHT

    if cfg.MODEL.POSE.ENABLED:
        models['pose'] = PoseNet(cfg.MODEL.POSE.NUM_LAYERS)
        weights['pose'] = cfg.MODEL.POSE.WEIGHT

    if cfg.MODEL.BS.ENABLED:
        bs_arch_keys = cfg.MODEL.BS.NAME.lower().split('_')
        enc_key = bs_arch_keys[0]
        dec_key = '_'.join(bs_arch_keys[1:])
        num_layers = cfg.MODEL.BS.NUM_LAYERS
        print('Encoder: {}-{}, Decoder: {}'.format(
            enc_key, num_layers, dec_key))

        bs_encoder = {
            'resnet': ResNetEncoder,
            'dla': DLAEncoder,
            'sparseinst': SparseInstEncoder
        }[enc_key]

        bs_detector = {
            'iam': IAMDetector,
        }[dec_key]

        models['bs'] = BSNetBase(
            bs_encoder, bs_detector, num_layers,
            inst_num=cfg.MODEL.BS.INSTANCES,
            down_ratio=cfg.MODEL.BS.DOWN_RATIO,
            prior_prob=cfg.MODEL.BS.PRIOR_PROB,
            group_num=cfg.MODEL.BS.GROUPS,
            group_norm=cfg.MODEL.BS.GROUP_NORM,
            deform_conv=cfg.MODEL.BS.DEFORMABLE)
        weights['bs'] = cfg.MODEL.BS.WEIGHT

    print('Model Initialization:')
    for key in models.keys():
        print('  {} weight:'.format(key), weights[key])
        if weights[key]:
            models[key].load_weight(weight=weights[key])

    return models
