import os

import torch
import numpy as np

from .nusc import nuScenesDataset
from .utils import \
    BlindSpotLabelLoader, \
    RandomRescaleCropCreator, \
    HorizontalFlipBlindSpot


class nuScenesBlindSpotDataset(nuScenesDataset):
    def __init__(
            self,
            *args,
            bs_label_path: str = '',
            num_max_instances: int = None,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.label_path = bs_label_path if bs_label_path else self.data_path

        self.bs_loader = BlindSpotLabelLoader()
        for i in range(len(self.filenames)):
            line = self.filenames[i]
            folder, frame_idx, side = self.parse_filename(line)
            bs_filename = os.path.join(
                self.label_path,
                folder,
                '{:010d}.json'.format(frame_idx))

            cache_key = (folder, frame_idx)
            self.bs_loader.load_labels(bs_filename, cache_key)
        self.bs_loader.concat_labels(num_max_instances)

        # We choice the greater scale than 1.0, because the lower scale
        # than 1.0 creates obstacle-like objects around images due to padding.
        self.rescale_crop_creator = RandomRescaleCropCreator(
                out_height=self.height,
                out_width=self.width,
                lower_scale=1.0,
                upper_scale=1.2)

        self.hflip_bs = HorizontalFlipBlindSpot()

    def get_color(
            self,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict) -> np.ndarray:
        color = super().get_color(folder, frame_index, side, augments)

        if augments['rescale_crop']:
            if 'rescale_crop_fn' not in augments:
                augments['rescale_crop_fn'] = \
                    self.rescale_crop_creator.create(color.height, color.width)

            color = augments['rescale_crop_fn'].apply_to_iamge(color)

        return color

    def get_intrinsic(
            self,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict) -> np.ndarray:
        K = super().get_intrinsic(folder, frame_index, side, augments)

        if augments['rescale_crop']:
            K = augments['rescale_crop_fn'].apply_to_intrinsic(K)

        return K

    def store_additional(
            self,
            inputs: dict,
            folder: str,
            frame_idx: int,
            side: str,
            augments: dict):

        super().store_additional(
            inputs, folder, frame_idx, side, augments)

        self.store_blindspot(inputs, folder, frame_idx, side, augments)

    def store_blindspot(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict):
        cache = self.bs_loader[folder, frame_index]
        point = cache['blindspot']
        T_ogm2cam = cache.get('ogm2cam', [np.eye(4)])[0]

        if augments['rescale_crop']:
            point = augments['rescale_crop_fn'].apply_to_blindspot(point)

        if augments['flip']:
            point, T_ogm2cam = self.hflip_bs(point, T_ogm2cam)

        inputs['bs_gt'] = torch.from_numpy(point)
        inputs['ogm2cam'] = torch.from_numpy(T_ogm2cam.astype(np.float32))
