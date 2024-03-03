import os
import json
from PIL import Image
from collections import defaultdict

import numpy as np

from .dataset import Dataset


class Ithaca365Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_calib_intrinsic = True

        # Averaged intrinsic matrix
        self.K = np.array([[0.957, 0, 0.5, 0],
                           [0, 1.492, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.raw_shape = (1920, 1208)
        self.full_res_shape = (1920, 1208)

        self.file_table = self._setup_file_table()

    def _setup_file_table(self, split_dirs=['v2.2']):
        """
        To get filenames and intrinsic from the frame index, whose type
        is int, the table of filenames are pre-constructed.
        """

        folders = {}

        for split_dir in split_dirs:
            # Load image paths
            sample_file = os.path.join(
                self.data_path, split_dir, 'sample_data.json')
            with open(sample_file, 'r') as f:
                _sample_obj = json.load(f)
            sampletoken2tokens = defaultdict(dict)
            sample_obj = {}
            for sample in _sample_obj:
                if sample['fileformat'] != 'png' \
                        or '/cam0/' not in sample['filename']:
                    continue

                sample_token = sample['sample_token']
                token = sample['token']
                timestamp = sample['timestamp']
                sampletoken2tokens[sample_token][timestamp] = token
                filename = sample['filename']
                next_token = sample['next']
                calib_token = sample['calibrated_sensor_token']

                sample_obj[token] = {
                    'next_token': next_token,
                    'calib_token': calib_token,
                    'filename': os.path.splitext(filename)[0] + self.img_ext,
                }

            # Load intrinsic parameters
            calib_file = os.path.join(
                self.data_path, split_dir, 'calibrated_sensor.json')
            with open(calib_file, 'r') as f:
                _calib_file = json.load(f)
            calib_file = {}
            for calib in _calib_file:
                if not calib['camera_intrinsic']:
                    continue
                calib_file[calib['token']] = calib['camera_intrinsic']

            # Load scenes
            scene_file = os.path.join(self.data_path, split_dir, 'scene.json')
            with open(scene_file, 'r') as f:
                scene_obj = json.load(f)
            for scene in scene_obj:
                scene_name = scene['name']
                # Since the first_sample_token denotes frame indentifier of
                # sample_data, all sensor has the same sample_token.
                # To get the unique token of cam0 sensor, convert it.
                tokens = sampletoken2tokens[scene['first_sample_token']]

                # Load the first token
                token = sorted(list(tokens.values()))[0]

                frames = {}
                i = 0
                while token:
                    sample = sample_obj[token]
                    filename = sample['filename']
                    calib_token = sample['calib_token']
                    intrinsic = calib_file[calib_token]

                    frames[i] = {
                        'filename': filename,
                        'intrinsic': intrinsic
                    }

                    token = sample['next_token']
                    i += 1
                folders[scene_name] = frames
        return folders

    def parse_filename(self, filename: str) -> tuple:
        line = filename.split()
        folder = line[0]
        frame_idx = int(line[1])
        return folder, frame_idx, None

    def get_color(
            self,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict) -> np.ndarray:
        folder_table = self.file_table.get(folder, {})
        if not folder_table:
            return None

        frame = folder_table.get(frame_index, {})
        if not frame:
            return None

        filepath = os.path.join(self.data_path, frame['filename'])
        if not os.path.isfile(filepath):
            return None

        color = self.load_image(filepath)
        color, _ = self.hcrop_pil(color, self.height / self.width)
        color = color.resize((self.width, self.height), Image.BILINEAR)
        return color

    def get_intrinsic(
            self,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict) -> np.ndarray:
        if self.use_calib_intrinsic:
            K = self._get_calib_intrinsic(folder, frame_index)
        else:
            K = self.K.copy()

        return K

    def _get_calib_intrinsic(self, folder, frame_index) -> np.ndarray:
        data = self.file_table[folder][frame_index]

        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = np.array(data['intrinsic'], dtype=np.float32)

        W, H = self.raw_shape
        K, new_H = self.hcrop_intrinsic(K, W, H, self.height / self.width)

        K[0, :] /= W
        K[1, :] /= new_H

        return K
