import os
import json

import torch
import numpy as np

from .nusc import nuScenesDataset
from .utils import load_generated_blindspot


class nuScenesBlindSpotDataset(nuScenesDataset):
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
                side,
                '{:010d}.json'.format(frame_index))

            with open(bs_filename, 'r') as f:
                obj = json.load(f)

            if obj['flags'].get('gtgen', False):
                points = load_generated_blindspot(obj)
            else:
                raise NotImplementedError()

            if 'blindspot' in points:
                bs_pt = np.stack(points['blindspot'], axis=0)
                bs_nums.append(len(bs_pt))
                bs_dims.append(bs_pt.shape[-1])
                points['blindspot'] = bs_pt

            caches[scene_name, frame_index, side] = points

        # Expand points to build mini-batch
        for key, val in caches.items():
            expand = np.full(
                (max(bs_nums), max(bs_dims)), -1, dtype=np.float32)
            if 'blindspot' in val:
                pts = val['blindspot']
                expand[:pts.shape[0]] = pts
            caches[key]['blindspot'] = expand

        return caches
