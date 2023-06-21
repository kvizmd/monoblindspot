import os
import json
from PIL import Image

import numpy as np

from .dataset import Dataset


class nuScenesDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_calib_intrinsic = True

        # Averaged intrinsic matrix
        self.K = np.array([[0.786, 0, 0.513, 0],
                           [0, 1.398, 0.533, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.raw_shape = (1600, 900)
        self.full_res_shape = (1600, 900)

        self.file_table = self._setup_file_table()

    def _setup_file_table(self):
        """
        To get filenames and intrinsic from the frame index, whose type
        is int, the table of filenames are pre-constructed.
        """

        folders = {}
        for split_dir in ['v1.0-test', 'v1.0-trainval']:
            # Load image paths
            sample_file = os.path.join(
                self.data_path, split_dir, 'sample_data.json')
            with open(sample_file, 'r') as f:
                _sample_obj = json.load(f)
            sampletoken2token = {}
            sample_obj = {}
            for sample in _sample_obj:
                if sample['fileformat'] != 'jpg' \
                        or '/CAM_FRONT/' not in sample['filename']:
                    continue

                sampletoken2token[sample['sample_token']] = sample['token']
                sample_obj[sample['token']] = {
                    'next_token': sample['next'],
                    'calib_token': sample['calibrated_sensor_token'],
                    'filename':  sample['filename']
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
                # To get the unique token of CAMERA_FORNT sensor, convert it.
                token = sampletoken2token[scene['first_sample_token']]

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

    def get_intrinsic(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool) -> np.ndarray:
        if self.use_calib_intrinsic:
            K = self._get_calib_intrinsic(folder, frame_index)
        else:
            K = self.K.copy()

        if do_flip:
            K[0, 2] = 1.0 - K[0, 2]

        return K

    def get_color(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool) -> np.ndarray:
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
        color = color.resize(self.full_res_shape, Image.BILINEAR)

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def _get_calib_intrinsic(self, folder, frame_index) -> np.ndarray:
        data = self.file_table[folder][frame_index]

        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = np.array(data['intrinsic'], dtype=np.float32)

        K[0, :] /= self.raw_shape[0]
        K[1, :] /= self.raw_shape[1]

        return K
