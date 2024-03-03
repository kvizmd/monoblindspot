from collections import defaultdict

from torch.utils.data import ConcatDataset

from .kitti import KITTIDataset
from .kitti_improved import KITTIImprovedDataset
from .kitti_bs import KITTIBlindSpotDataset
from .nusc import nuScenesDataset
from .nusc_bs import nuScenesBlindSpotDataset
from .cityscapes import CityscapesDataset
from .cityscapes_bs import CityScapesBlindSpotDataset
from .ddad import DDADDataset
from .ithaca import Ithaca365Dataset
from .ithaca_bs import Ithaca365BlindSpotDataset


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
            nuScenesDataset,
        ),
        'cityscapes': (
            CityscapesDataset,
            CityscapesDataset,
            CityscapesDataset,
        ),
        'ddad': (
            DDADDataset,
            DDADDataset,
            DDADDataset,
        ),
        'ithaca': (
            Ithaca365Dataset,
            Ithaca365Dataset,
            Ithaca365Dataset,
        )
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
            aug_color_prob=cfg.DATA.AUG.COLOR,
            aug_hflip_prob=cfg.DATA.AUG.H_FLIP,
            aug_rescale_crop_prob=cfg.DATA.AUG.RESCALE_CROP,
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
        'cityscapes': (
            CityscapesDataset,
            CityscapesDataset,
            lambda *args, **kwargs: None
        ),
        'ddad': (
            DDADDataset,
            DDADDataset,
            lambda *args, **kwargs: None
        ),
        'ithaca': (
            Ithaca365Dataset,
            Ithaca365Dataset,
            lambda *args, **kwargs: None
        )
    }[str(cfg.DATA.NAME).lower()]

    scales = len(cfg.DATA.SCALES)
    train_filenames = readlines(cfg.DATA.TRAIN_SPLIT)
    val_filenames = readlines(cfg.DATA.VAL_SPLIT)
    test_filenames = readlines(cfg.DATA.TEST_SPLIT)

    datasets = {}

    require_depth_gt = 'scale' in cfg.OGM.NAME.lower()
    require_oxts_pose = 'oxts' in cfg.OGM.NAME.lower()

    if 'train' in keys:
        datasets['train'] = train_dataset(
            cfg.DATA.ROOT_DIR, train_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False,
            require_depth_gt=require_depth_gt,
            require_adjacent_depth_gt=require_depth_gt,
            require_pose=require_oxts_pose)

    if 'val' in keys:
        datasets['val'] = val_dataset(
            cfg.DATA.ROOT_DIR, val_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False,
            require_depth_gt=require_depth_gt,
            require_adjacent_depth_gt=require_depth_gt,
            require_pose=require_oxts_pose)

    if 'test' in keys:
        datasets['test'] = test_dataset(
            cfg.DATA.ROOT_DIR, test_filenames,
            cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
            cfg.DATA.FRAME_IDXS, scales, is_train=False,
            require_depth_gt=require_depth_gt,
            require_adjacent_depth_gt=require_depth_gt,
            require_pose=require_oxts_pose)

    if len(keys) == 1:
        return datasets[keys[0]]

    return datasets


def build_bs_dataset(cfg, keys: list = ['train', 'val', 'test']):
    dataset_list = {
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
        'cityscapes': (
            CityScapesBlindSpotDataset,
            CityscapesDataset,
            lambda *args, **kwargs: None
        ),
        'ddad': (
            DDADDataset,
            DDADDataset,
            lambda *args, **kwargs: None
        ),
        'ithaca': (
            Ithaca365BlindSpotDataset,
            Ithaca365Dataset,
            lambda *args, **kwargs: None
        )
    }

    scales = len(cfg.DATA.SCALES)
    if len(cfg.DATA.NAMES) > 1:
        num_max_instances = 100
    else:
        num_max_instances = None

    if len(cfg.DATA.NAMES) > 0:
        data_names = cfg.DATA.NAMES
        train_splits = cfg.DATA.TRAIN_SPLITS
        val_splits = cfg.DATA.VAL_SPLITS
        test_splits = cfg.DATA.TEST_SPLITS
        bs_label_paths = cfg.DATA.BS_LABELS
        root_dirs = cfg.DATA.ROOT_DIRS
    else:
        data_names = [cfg.DATA.NAME]
        train_splits = [cfg.DATA.TRAIN_SPLIT]
        val_splits = [cfg.DATA.VAL_SPLIT]
        test_splits = [cfg.DATA.TEST_SPLIT]
        bs_label_paths = [cfg.DATA.BS_LABEL]
        root_dirs = [cfg.DATA.ROOT_DIR]

    print('Datasets:')
    datasets = defaultdict(list)
    for i, (name, bs_label_path, root_dir) in enumerate(
            zip(data_names, bs_label_paths, root_dirs)):
        name = str(name).lower()
        train_filenames = readlines(train_splits[i])
        val_filenames = readlines(val_splits[i])
        test_filenames = readlines(test_splits[i])

        train_dataset, val_dataset, test_dataset = dataset_list[name]

        if 'train' in keys:
            datasets['train'].append(train_dataset(
                root_dir, train_filenames,
                cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
                cfg.DATA.FRAME_IDXS, scales, is_train=True,
                bs_label_path=bs_label_path,
                aug_color_prob=cfg.DATA.AUG.COLOR,
                aug_hflip_prob=cfg.DATA.AUG.H_FLIP,
                aug_rescale_crop_prob=cfg.DATA.AUG.RESCALE_CROP,
                num_max_instances=num_max_instances,
                require_depth_gt=False))

        if 'val' in keys:
            datasets['val'].append(val_dataset(
                root_dir, val_filenames,
                cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
                cfg.DATA.FRAME_IDXS, scales, is_train=False,
                num_max_instances=num_max_instances,
                require_depth_gt=False))

        if 'test' in keys:
            datasets['test'].append(test_dataset(
                root_dir, test_filenames,
                cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH,
                cfg.DATA.FRAME_IDXS, scales, is_train=False,
                num_max_instances=num_max_instances,
                require_depth_gt=False))

        print('[  OK  ] ' + name)

    _datasets = {}
    for key, sources in datasets.items():
        if len(cfg.DATA.NAMES) > 1 and key == 'train':
            _datasets[key] = ConcatDataset(
                [s for s in sources if s is not None])
        else:
            _datasets[key] = sources[0]
    datasets = _datasets

    if len(keys) == 1:
        return datasets[keys[0]]

    return datasets
