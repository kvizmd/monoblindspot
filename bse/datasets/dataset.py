import random

import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(
            self,
            data_dir: str,
            filenames: list,
            height: int,
            width: int,
            frame_indices: list,
            num_scales: int,
            is_train: bool = False,
            img_ext: str = '.jpg',
            **kwargs):
        super().__init__()

        self.data_path = data_dir
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.frame_indices = frame_indices
        self.is_train = is_train
        self.img_ext = img_ext

        self.to_tensor = transforms.ToTensor()

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.colorjit = transforms.ColorJitter()

        self.dummy_color = {}
        self.resize = {}
        self.resize_noanti = {}
        for i in range(self.num_scales):
            s = 2 ** i
            h, w = self.height // s, self.width // s
            self.resize[i] = transforms.Resize((h, w), antialias=True)
            self.resize_noanti[i] = transforms.Resize(
                (h, w), antialias=False,
                interpolation=transforms.InterpolationMode.NEAREST)
            self.dummy_color[i] = torch.zeros((3, h, w), dtype=torch.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        do_colorjit = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[idx]
        folder, frame_idx, side = self.parse_filename(line)

        inputs = {
            'filename': self.filenames[idx],
            'do_cjit': torch.full((1, ), do_colorjit, dtype=torch.bool),
            'do_flip': torch.full((1, ), do_flip, dtype=torch.bool)
        }

        self.store_color(inputs, folder, frame_idx, side, do_flip, do_colorjit)
        self.store_intrinsic(inputs, folder, frame_idx, side, do_flip)

        self.store_additional(
            inputs, folder, frame_idx, side, do_flip, do_colorjit)

        return inputs

    def parse_filename(self, filename: str) -> tuple:
        raise NotImplementedError()

    def get_color(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool) -> np.ndarray:
        raise NotImplementedError()

    def get_intrinsic(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool) -> np.ndarray:
        raise NotImplementedError()

    def store_color(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool,
            do_colorjit: bool):

        # We create the color_aug object in advance and apply the same
        # augmentation to all images in this item. This ensures that all images
        # input to the pose network receive the same augmentation.
        if do_colorjit:
            colorjit_params = self.colorjit.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
            colorjit = transforms.ColorJitter()
            colorjit.get_params = lambda *args, **kwargs: colorjit_params

        else:
            def colorjit(x): return x

        for i in self.frame_indices:
            color = self.get_color(folder, frame_index + i, side, do_flip)

            if color is None:
                if i == 0:
                    raise RuntimeError(
                        'The target frame ({}, {}, {}) does not exist.'.format(
                            folder, frame_index, side))

                inputs[('valid_indices', i)] = torch.zeros((1,), dtype=bool)
            else:
                inputs[('valid_indices', i)] = torch.ones((1,), dtype=bool)

            for s in range(self.num_scales):
                if color is None:
                    # To make a batch, append dummpy tensors.
                    inputs['color', i, s] = self.dummy_color[s]
                    inputs['color_aug', i, s] = self.dummy_color[s]
                else:
                    img_s = self.resize[s](color)
                    inputs['color', i, s] = self.to_tensor(img_s)
                    inputs['color_aug', i, s] = self.to_tensor(colorjit(img_s))

    def store_intrinsic(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool):
        K = self.get_intrinsic(folder, frame_index, side, do_flip)

        # Adjusting intrinsics to match each scale in the pyramid
        for s in range(self.num_scales):
            K = K.copy()

            K[0, :] *= self.width // (2 ** s)
            K[1, :] *= self.height // (2 ** s)

            inv_K = np.linalg.inv(K)

            inputs['K', s] = torch.from_numpy(K)
            inputs['inv_K', s] = torch.from_numpy(inv_K)

    def store_additional(
            self,
            inputs: dict,
            folder: str,
            frame_idx: int,
            side: str,
            do_flip: bool,
            do_colorjit: bool):
        return

    def load_image(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
