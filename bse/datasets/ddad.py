import os
import json
from PIL import Image

import numpy as np

from .dataset import Dataset


class DDADDataset(Dataset):
    """
    Assumes the foolwing directory structure.

    ddad(self.data_path)/
      ddad_train_val/
        000000/
          calibration/
          point_cloud/
          rgb/
          scene_5c5616b2e583b5fb4f25013580172d1df43b8a31.json
        000001/
        000002/
        000003/
        000004/
        ..
        LICENSE.md
        ddad.json
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.K = np.array([[1.125, 0, 0.5, 0],
                           [0, 1.794, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.use_calib_intrinsic = True

        self.raw_shape = (1936, 1216)
        self.full_res_shape = (1936, 1162)  # To divide by 32.

        self.file_table = self._setup_file_table()

    def _setup_file_table(self):
        """
        To get filenames and intrinsic from the frame index, whose type
        is int, the table of filenames are pre-constructed.
        """

        filename = os.path.join(self.data_path, 'ddad_train_val/ddad.json')
        with open(filename, 'r') as f:
            obj = json.load(f)

        scene_files = []
        for split in obj['scene_splits'].values():
            scene_files += split['filenames']

        folders = {}
        for scene_file in scene_files:
            scene_path = os.path.join(
                self.data_path, 'ddad_train_val', scene_file)
            scene_id = int(scene_file.split('/')[0])
            with open(scene_path, 'r') as f:
                scene_obj = json.load(f)

            cam_data = {}
            for data in scene_obj['data']:
                if data['id']['name'] != 'CAMERA_01':
                    continue

                cam_data[data['key']] = data['datum']['image']['filename']

            frames = {}
            for frame_idx, sample in enumerate(scene_obj['samples']):
                calib_path = os.path.join(
                    os.path.dirname(scene_path),
                    'calibration/{}.json'.format(sample['calibration_key']))

                with open(calib_path, 'r') as f:
                    calib_obj = json.load(f)

                intrinsic = None
                for i, name in enumerate(calib_obj['names']):
                    if name == 'CAMERA_01':
                        intrinsic = calib_obj['intrinsics'][i]

                img_filename = None
                for datum_key in sample['datum_keys']:
                    if datum_key in cam_data:
                        img_filename = cam_data[datum_key]

                frames[frame_idx] = {
                    'intrinsic': intrinsic,
                    'filename': img_filename.split('/')[-1].split('.')[0]
                }

            folders[scene_id] = frames
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
            do_flip: str) -> np.ndarray:

        folder_table = self.file_table.get(int(folder), {})
        if not folder_table:
            return None

        frame = folder_table.get(frame_index, {})
        if not frame:
            return None

        filepath = os.path.join(
            self.data_path, 'ddad_train_val', folder,
            'rgb/CAMERA_01', frame['filename'] + self.img_ext)

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
            do_flip: bool) -> np.ndarray:
        if self.use_calib_intrinsic:
            K = self._get_calib_intrinsic(folder, frame_index)
        else:
            K = self.K.copy()
        return K

    def _get_calib_intrinsic(self, folder, frame_index) -> np.ndarray:
        data = self.file_table[int(folder)][frame_index]['intrinsic']

        fx = data['fx']
        fy = data['fy']
        cx = data['cx']
        cy = data['cy']

        K = np.array([[fx, 0, cx, 0],
                      [0, fy, cy, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]).astype(np.float32)

        W, H = self.raw_shape
        K, new_H = self.hcrop_intrinsic(K, W, H, self.height / self.width)

        K[0, :] /= W
        K[1, :] /= new_H
        return K
