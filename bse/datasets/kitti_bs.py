import os
import json

import numpy as np
from PIL import Image, ImageDraw
import torch

from .kitti import KITTIDataset
from .utils import \
    load_annotated_blindspot, \
    load_generated_blindspot


class KITTIBlindSpotDataset(KITTIDataset):
    def __init__(self, *args, bs_label_path: str = '', **kwargs):
        super().__init__(*args, **kwargs)

        self.label_path = bs_label_path if bs_label_path else self.data_path

        self.caches = self.load_blindspot_label()

    def store_additional(
            self,
            inputs: dict,
            folder: str,
            frame_idx: int,
            side: str,
            do_flip: bool,
            do_colorjit: bool):

        super().store_additional(
            inputs, folder, frame_idx, side, do_flip, do_colorjit)

        self.store_blindspot(inputs, folder, frame_idx, side, do_flip)

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
            do_flip: bool):
        point = self.caches[folder, frame_index, side]['blindspot']

        if do_flip:
            point[:, 1] = 1.0 - point[:, 1]

        inputs['bs_gt'] = torch.from_numpy(point)

    def store_ignore_mask(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str):
        shapes = self.caches[folder, frame_index, side].get('ambiguous', [])

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

    def load_blindspot_label(self) -> dict:
        caches = {}
        bs_nums = []
        bs_dims = []
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

            with open(bs_filename, 'r') as f:
                obj = json.load(f)

            if obj['flags'].get('gtgen', False):
                points = load_generated_blindspot(obj)
            else:
                points = load_annotated_blindspot(obj)

            if 'blindspot' in points:
                bs_pt = np.stack(points['blindspot'], axis=0)
                bs_nums.append(len(bs_pt))
                bs_dims.append(bs_pt.shape[-1])
                points['blindspot'] = bs_pt

            caches[scene_name, frame_index, side] = points

        for key, val in caches.items():
            # Expand points with -1 to create mini-batch.
            expand = np.full(
                (max(bs_nums), max(bs_dims)), -1, dtype=np.float32)
            if 'blindspot' in val:
                pts = val['blindspot']
                expand[:pts.shape[0]] = pts
            caches[key]['blindspot'] = expand

        return caches
