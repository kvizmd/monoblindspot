from .kitti import KITTIDataset
from .kitti_improved import KITTIImprovedDataset
from .kitti_bs import KITTIBlindSpotDataset
from .nusc import nuScenesDataset
from .nusc_bs import nuScenesBlindSpotDataset


def readlines(filename: str) -> list:
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def build_depth_dataset(cfg, keys: list = ['train', 'val', 'test']):
    train_dataset, val_dataset, test_dataset = {
        'kitti': (
            KITTIDataset,
            KITTIDataset,
            KITTIDataset
        ),
        'kitti_improved': (
            KITTIImprovedDataset,
            KITTIImprovedDataset,
            KITTIImprovedDataset,
        ),
        'nusc': (
            nuScenesDataset,
            nuScenesDataset,
            lambda *args, **kwargs: None
        ),
    }[str(cfg.DATA.NAME).lower()]

    scales = len(cfg.DATA.SCALES)
    train_filenames = readlines(cfg.DATA.TRAIN_SPLIT)
    val_filenames = readlines(cfg.DATA.VAL_SPLIT)
    test_filenames = readlines(cfg.DATA.TEST_SPLIT)

    datasets = {}
    if 'train' in keys:
        datasets['train'] = train_dataset(
            cfg.DATA.ROOT_DIR, train_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=True,
            require_depth_gt=False)

    if 'val' in keys:
        datasets['val'] = val_dataset(
            cfg.DATA.ROOT_DIR, val_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False)

    if 'test' in keys:
        datasets['test'] = test_dataset(
            cfg.DATA.ROOT_DIR, test_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False)

    if len(keys) == 1:
        return datasets[keys[0]]

    return datasets


def build_bsgen_dataset(cfg, keys: list = ['train', 'val', 'test']):
    train_dataset, val_dataset, test_dataset = {
        'kitti': (
            KITTIDataset,
            KITTIBlindSpotDataset,
            KITTIBlindSpotDataset
        ),
        'nusc': (
            nuScenesDataset,
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: None
        ),
    }[str(cfg.DATA.NAME).lower()]

    scales = len(cfg.DATA.SCALES)
    train_filenames = readlines(cfg.DATA.TRAIN_SPLIT)
    val_filenames = readlines(cfg.DATA.VAL_SPLIT)
    test_filenames = readlines(cfg.DATA.TEST_SPLIT)

    datasets = {}

    if 'train' in keys:
        datasets['train'] = train_dataset(
            cfg.DATA.ROOT_DIR, train_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False,
            require_depth_gt=False,
            require_adjacent_depth_gt=False,
            require_pose=False)

    if 'val' in keys:
        datasets['val'] = val_dataset(
            cfg.DATA.ROOT_DIR, val_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False,
            require_depth_gt=False,
            require_adjacent_depth_gt=False,
            require_pose=False)

    if 'test' in keys:
        datasets['test'] = test_dataset(
            cfg.DATA.ROOT_DIR, test_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False,
            require_depth_gt=False,
            require_adjacent_depth_gt=False,
            require_pose=False)

    if len(keys) == 1:
        return datasets[keys[0]]

    return datasets


def build_bs_dataset(cfg, keys: list = ['train', 'val', 'test']):
    train_dataset, val_dataset, test_dataset = {
        'kitti': (
            KITTIBlindSpotDataset,
            KITTIBlindSpotDataset,
            KITTIBlindSpotDataset,
        ),
        'nusc': (
            nuScenesBlindSpotDataset,
            nuScenesDataset,
            lambda *args, **kwargs: None
        ),
    }[str(cfg.DATA.NAME).lower()]

    scales = len(cfg.DATA.SCALES)
    train_filenames = readlines(cfg.DATA.TRAIN_SPLIT)
    val_filenames = readlines(cfg.DATA.VAL_SPLIT)
    test_filenames = readlines(cfg.DATA.TEST_SPLIT)

    datasets = {}

    if 'train' in keys:
        datasets['train'] = train_dataset(
            cfg.DATA.ROOT_DIR, train_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=True,
            bs_label_path=cfg.TRAIN.BS.OFFLINE_LABELS,
            require_depth_gt=False)

    if 'val' in keys:
        datasets['val'] = val_dataset(
            cfg.DATA.ROOT_DIR, val_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False,
            require_depth_gt=False)

    if 'test' in keys:
        datasets['test'] = test_dataset(
            cfg.DATA.ROOT_DIR, test_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False,
            require_depth_gt=False)

    if len(keys) == 1:
        return datasets[keys[0]]

    return datasets
