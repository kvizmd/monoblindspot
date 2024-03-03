import os

import numpy as np
from PIL import Image, ImageDraw
import torch

from .kitti import KITTIDataset
from .utils import \
    BlindSpotLabelLoader, \
    RandomRescaleCropCreator, \
    HorizontalFlipBlindSpot


class KITTIBlindSpotDataset(KITTIDataset):
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
            line = self.filenames[i].split()
            scene_name = line[0]
            frame_index = int(line[1])
            side = line[2]
            bs_filename = os.path.join(
                self.label_path,
                scene_name,
                'image_0{}'.format(self.side_map[side]),
                'bs',
                '{:010d}.json'.format(frame_index))

            cache_key = (scene_name, frame_index, side)
            self.bs_loader.load_labels(bs_filename, cache_key)
        self.bs_loader.concat_labels(num_max_instances)

        # We choice the greater scale than 1.0, because the lower scale
        # than 1.0 creates obstacle-like objects around images due to padding.
        self.rescale_crop_creator = RandomRescaleCropCreator(
                out_height=self.height,
                out_width=self.width,
                lower_scale=0.8,
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

        if not self.is_train:
            self.store_ground_depth(inputs, folder, frame_idx, side)
            self.store_calib_intrinsic(inputs, folder, frame_idx, side)
            self.store_ignore_mask(inputs, folder, frame_idx, side)

    def store_blindspot(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict):

        cache = self.bs_loader[folder, frame_index, side]
        point = cache['blindspot']
        T_ogm2cam = cache.get('ogm2cam', [np.eye(4)])[0]

        if augments['rescale_crop']:
            point = augments['rescale_crop_fn'].apply_to_blindspot(point)

        if augments['flip']:
            point, T_ogm2cam = self.hflip_bs(point, T_ogm2cam)

        inputs['bs_gt'] = torch.from_numpy(point)
        inputs['ogm2cam'] = torch.from_numpy(T_ogm2cam.astype(np.float32))

    def store_ignore_mask(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str):
        shapes = self.bs_loader[folder, frame_index, side].get('ambiguous', [])

        W, H = self.full_res_shape
        mask = np.zeros((H, W), dtype=bool)
        for data in shapes:
            points = data['points']
            if data['type'] == 'box':
                bbox = np.stack(points).reshape(2, 2)
                bbox[:, 0] *= W - 1
                bbox[:, 1] *= H - 1
                bbox = bbox.astype(int).flatten()
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = True

            elif data['type'] == 'polygon':
                points = [
                    (int(p[1] * (W - 1)), int(p[0] * (H - 1)))
                    for p in points
                ]

                area = Image.new('L', (W, H))
                draw = ImageDraw.Draw(area)
                draw.polygon(points, fill=1)
                area = np.array(area, dtype=bool)
                mask |= area

        mask = np.expand_dims(mask, 0)
        inputs['bs_ignore'] = torch.from_numpy(mask)

    def store_ground_depth(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str):
        depth_path = os.path.join(
            self.data_path,
            folder,
            'image_0{}'.format(self.side_map[side]),
            'ground_depth',
            '{:010d}.png'.format(frame_index))

        depth_gt = Image.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, Image.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        depth_gt = np.expand_dims(depth_gt, 0)
        inputs['ground_depth'] = torch.from_numpy(depth_gt)

    def store_calib_intrinsic(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str) -> tuple:
        K = self._get_calib_intrinsic(folder, side)
        W, H = self.full_res_shape
        K[0, :3] *= W
        K[1, :3] *= H

        inputs['K_calib'] = torch.from_numpy(K)

        inv_K = np.linalg.inv(K)
        inputs['inv_K_calib'] = torch.from_numpy(inv_K)
