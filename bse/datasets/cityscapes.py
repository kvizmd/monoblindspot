import os
import json
import glob
from PIL import Image

import numpy as np

from .dataset import Dataset


class CityscapesDataset(Dataset):
    """
    This assumes the following directory structure.
    The source data is based on leftImg8bit_sequence_trainvaltest.zip and
    the format of these image is converted to the JPEG to save memories.

    leftImg8bit_sequence/
        aachen/
            aachen_000000_000000_leftImg8bit.jpg
            aachen_000000_000001_leftImg8bit.jpg
            aachen_000000_000002_leftImg8bit.jpg
            aachen_000000_000003_leftImg8bit.jpg
            aachen_000000_000004_leftImg8bit.jpg
            aachen_000000_000005_leftImg8bit.jpg
            aachen_000000_000006_leftImg8bit.jpg
            aachen_000000_000007_leftImg8bit.jpg
            aachen_000000_000008_leftImg8bit.jpg
            aachen_000000_000009_leftImg8bit.jpg
            ...
        berlin/
        bielefeld/
        bochum/
        bonn/
        bremen/
        ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.crop_ratio = 0.75
        self.full_res_shape = (1024, 384)
        self.raw_shape = (2048, 1024)

        self.K = np.array([[1.105, 0, 0.5269, 0],
                           [0, 2.198, 0.6706, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.use_calib_intrinsic = True

    def parse_filename(self, filename: str) -> tuple:
        line = filename.split()

        city = line[0]
        filename = line[1]
        _, snippets_idx, frame_idx = filename.split('_')

        frame_idx = int(frame_idx)
        city_snippets = city + '_' + snippets_idx

        return city_snippets, frame_idx, None

    def get_color(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: str) -> np.array:
        city_snippets = folder.split('_')
        city = city_snippets[0]
        snippets_idx = int(city_snippets[1])

        filename = os.path.join(
            self.data_path, 'leftImg8bit_sequence', city,
            '{}_{:06}_{:06}_leftImg8bit{}'.format(
                city, snippets_idx, frame_index, self.img_ext))

        if not os.path.isfile(filename):
            return None

        color = self.load_image(filename)

        # To remove the ego vehicle, the crops off the bottom 25% of the image.
        color = color.crop(
            (0, 0, color.width, int(color.height * self.crop_ratio)))

        color, _ = self.hcrop_pil(color, self.height / self.width)
        color = color.resize((self.width, self.height), Image.BILINEAR)

        return color

    def get_intrinsic(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool) -> np.array:
        """
        Get an intrinsic of the camera pre-defined in the constructer.

        Args:
          folder: the directory of the dataset.
          frame_index: the index of the target frame
          side: the string of side tag, such as 'l' or 'r'.
          do_flip: the boolean flag for the horizontal flipping augmentation.
        """

        if self.use_calib_intrinsic:
            K = self._get_calib_intrinsic(folder, frame_index)
        else:
            K = self.K.copy()
        return K

    def _get_calib_intrinsic(self, folder: str, frame_index: int) -> np.array:
        city_snippets = folder.split('_')
        city = city_snippets[0]
        snippets_idx = int(city_snippets[1])
        filename = glob.glob(os.path.join(
            self.data_path, 'camera', city,
            '{}_{:06}_*_camera.json'.format(city, snippets_idx)))[0]

        if not os.path.isfile(filename):
            return None

        with open(filename, 'r') as f:
            data = json.load(f)['intrinsic']

        fx = data['fx']
        fy = data['fy']
        u0 = data['u0']
        v0 = data['v0']

        K = np.array([[fx, 0, u0, 0],
                      [0, fy, v0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]).astype(np.float32)

        parent_dir = os.path.basename(os.path.dirname(filename))
        basename = os.path.basename(filename)
        city, snippets, frames, sensor = basename.split('_')
        img_filename = os.path.join(
            self.data_path,
            'leftImg8bit_sequence', parent_dir,
            '{}_{}_{}_leftImg8bit.jpg'.format(city, snippets, frames))
        img = Image.open(img_filename)

        W = img.width
        H = img.height * self.crop_ratio

        K, new_H = self.hcrop_intrinsic(K, W, H, self.height / self.width)

        K[0, :] /= W
        K[1, :] /= new_H

        return K
