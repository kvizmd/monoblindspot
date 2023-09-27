from glob import glob

from yacs.config import CfgNode

_C = CfgNode()

# General
_C._BASE_ = None
_C.BENCHMARK = False
_C.NUM_WORKERS = 4
_C.RANDOM_SEED = None
_C.DEVICE = 'cuda'
_C.MULTI_GPU = False

_C.TARGET_MODE = 'bs'

# Data
_C.DATA = CfgNode()
_C.DATA.NAME = 'kitti'
_C.DATA.ROOT_DIR = '@REQUIRED@'
_C.DATA.TRAIN_SPLIT = 'splits/eigen_zhou/train_files.txt'
_C.DATA.VAL_SPLIT = 'splits/eigen_zhou/val_files.txt'
_C.DATA.TEST_SPLIT = 'splits/eigen/test_files.txt'

_C.DATA.BATCH_SIZE = 16
_C.DATA.IMG_HEIGHT = 192
_C.DATA.IMG_WIDTH = 640

_C.DATA.SCALES = [0, 1, 2, 3]
_C.DATA.FRAME_IDXS = [0, -1, 1]

_C.DATA.NO_SHUFFLE = False

# Training
_C.TRAIN = CfgNode()
_C.TRAIN.MAX_EPOCHS = 20
_C.TRAIN.LOGGING_ITER = 1000
_C.TRAIN.FIGSAVE_ITER = 1000
_C.TRAIN.EVAL_BATCHES = int(1e10)
_C.TRAIN.LIMIT_BATCHES = int(1e10)

_C.TRAIN.WARMUP_ITERS = -1
_C.TRAIN.LR_MILESTONES = [15]
_C.TRAIN.GRAD_CLIP = -1.0
_C.TRAIN.WEIGHT_DECAY = 1e-4

_C.TRAIN.DEFAULT_LR = 1e-4

_C.TRAIN.REFERENCE_METRIC = 'middle/f1'

# Training DepthNet
_C.TRAIN.DEPTH = CfgNode()
_C.TRAIN.DEPTH.LR = 1e-4

# Training PoseNet
_C.TRAIN.POSE = CfgNode()
_C.TRAIN.POSE.LR = 1e-4

# Training BlindSpotNet
_C.TRAIN.BS = CfgNode()
_C.TRAIN.BS.ENCODER_LR = 1e-4
_C.TRAIN.BS.DECODER_LR = 1e-4
_C.TRAIN.BS.OFFLINE_LABELS = ''


# Model
_C.MODEL = CfgNode()

_C.MODEL.DEPTH = CfgNode()
_C.MODEL.DEPTH.ENABLED = False
_C.MODEL.DEPTH.NUM_LAYERS = 18
_C.MODEL.DEPTH.SCALES = [0, 1, 2, 3]
_C.MODEL.DEPTH.WEIGHT = ''
_C.MODEL.DEPTH.MIN_DEPTH = 0.1
_C.MODEL.DEPTH.MAX_DEPTH = 100

_C.MODEL.POSE = CfgNode()
_C.MODEL.POSE.ENABLED = False
_C.MODEL.POSE.NUM_LAYERS = 18
_C.MODEL.POSE.WEIGHT = ''

_C.MODEL.BS = CfgNode()
_C.MODEL.BS.ENABLED = True
_C.MODEL.BS.NAME = 'baseline'
_C.MODEL.BS.NUM_LAYERS = 18
_C.MODEL.BS.WEIGHT = ''
_C.MODEL.BS.DOWN_RATIO = 8
_C.MODEL.BS.INSTANCES = 24
_C.MODEL.BS.PRIOR_PROB = 0.01
_C.MODEL.BS.GROUPS = 1
_C.MODEL.BS.DEFORMABLE = False
_C.MODEL.BS.GROUP_NORM = False


# Loss
_C.LOSS = CfgNode()

_C.LOSS.DEPTH = CfgNode()
_C.LOSS.DEPTH.FACTOR = 1.0
_C.LOSS.DEPTH.SMOOTH_FACTOR = 0.001
_C.LOSS.DEPTH.FRAME_IDXS = _C.DATA.FRAME_IDXS
_C.LOSS.DEPTH.SCALES = _C.DATA.SCALES
_C.LOSS.DEPTH.SMOOTH_SCALING = True

_C.LOSS.BS = CfgNode()
_C.LOSS.BS.NAME = 'bipart'
_C.LOSS.BS.FACTOR = 1.0
_C.LOSS.BS.CLS_FACTOR = 1.0
_C.LOSS.BS.POS2D_FACTOR = 1.0
_C.LOSS.BS.OFFSET_FACTOR = 1.0

_C.LOSS.BS.SCORE_MATCH = 0.2
_C.LOSS.BS.POS_MATCH = 0.8

_C.LOSS.BS.OCC_WEIGHTING = True

# OGM
_C.OGM = CfgNode()

_C.OGM.NAME = 'sequential'
_C.OGM.SIZE = 128
_C.OGM.MEDIAN_SCALE = 16
_C.OGM.SUBGRIDS = 32
_C.OGM.HEIGHT_QUANTILE = 0.75

_C.OGM.THRESHOLD = 0.55
_C.OGM.COUNT_THRESHOLD = 3
_C.OGM.DEPTH_GRAD_THRESHOLD = 0.0005
_C.OGM.POOLING_KERNEL_SIZE = 9

_C.OGM.MIN_PRIOR_PROB = 0.35
_C.OGM.MAX_PRIOR_PROB = 0.65
_C.OGM.UNKNOWN_PRIOR_PROB = 0.5

_C.OGM.RANSAC_ITERATIONS = 100
_C.OGM.SAMPLING_ACCEPT_OFFSET = 0.25

_C.EVAL = CfgNode()
_C.EVAL.SET_NAME = 'val'

_C.EVAL.BS = CfgNode()
_C.EVAL.BS.ERROR_THRESHOLD = 1.5
_C.EVAL.BS.SCORE_THRESHOLD = 0.5
_C.EVAL.BS.MAX_DEPTH = 80.0

_C.EVAL.BS.SHORT_LOWER = 0.0
_C.EVAL.BS.SHORT_UPPER = 15.0

_C.EVAL.BS.MIDDLE_LOWER = 0.0
_C.EVAL.BS.MIDDLE_UPPER = 30.0

_C.EVAL.BS.LONG_LOWER = 0.0
_C.EVAL.BS.LONG_UPPER = 60.0


def get_cfg_defaults():
    return _C.clone()


def _load_config_recursive(config_file: str) -> CfgNode:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)

    if cfg._BASE_ is not None:
        base_config_file = glob(cfg._BASE_)[0]
        cfg = _load_config_recursive(base_config_file)
        cfg.merge_from_file(config_file)

    print('Config Loaded: ', config_file)
    return cfg


def load_config(
        config_file: str,
        override_opts: list = [],
        freeze: bool = True,
        check_requirements: bool = True) -> CfgNode:
    """
    Load config file.

    Args:
      config_file: the filename of yaml.
      override_opts: List of keys and values, such as [key1, val1, key2, val2]
      freeze: Whether the return value is unchangeable or not.
      check_requirements: Output error for parameters with value '@REQUIRED@'.
    """
    cfg = _load_config_recursive(config_file)

    if override_opts:
        cfg.merge_from_list(override_opts)

    def check_if_satisfied(cfg, parent_key='', unsatisfied_keys=[]):
        for key, val in cfg.items():
            key = parent_key + key

            if isinstance(val, dict):
                check_if_satisfied(val, key + '.', unsatisfied_keys)

            elif '@required@' in str(val).lower():
                unsatisfied_keys.append(key)

    if check_requirements:
        unsatisfied_keys = []
        check_if_satisfied(cfg, unsatisfied_keys=unsatisfied_keys)
        if len(unsatisfied_keys) > 0:
            raise RuntimeError(
                'The following parameters are not satistied: ' +
                str(unsatisfied_keys))

    if freeze:
        cfg.freeze()

    return cfg
