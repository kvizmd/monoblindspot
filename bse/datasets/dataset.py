import random

import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms

from .utils import ColorJitterCreator


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
            aug_color_prob: float = 0.5,
            aug_hflip_prob: float = 0.5,
            aug_rescale_crop_prob: float = 0.5,
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
        self.aug_color_prob = aug_color_prob
        self.aug_hflip_prob = aug_hflip_prob
        self.aug_rescale_crop_prob = aug_rescale_crop_prob

        self.to_tensor = transforms.ToTensor()
        self.colorjit_creator = ColorJitterCreator(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.1, 0.1))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        augments = {
            'colorjit': random.random() < self.aug_color_prob,
            'flip': random.random() < self.aug_hflip_prob,
            'rescale_crop': random.random() < self.aug_rescale_crop_prob,
        }
        if not self.is_train:
            for key in augments.keys():
                augments[key] = False

        line = self.filenames[idx]
        folder, frame_idx, side = self.parse_filename(line)

        inputs = {
            'filename': self.filenames[idx]
        }

        self.store_color(inputs, folder, frame_idx, side, augments)
        self.store_intrinsic(inputs, folder, frame_idx, side, augments)

        self.store_additional(inputs, folder, frame_idx, side, augments)

        return inputs

    def parse_filename(self, filename: str) -> tuple:
        raise NotImplementedError()

    def get_color(
            self,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict) -> np.ndarray:
        raise NotImplementedError()

    def get_intrinsic(
            self,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict) -> np.ndarray:
        raise NotImplementedError()

    def store_color(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict):

        if augments['colorjit']:
            augments['colorjit_fn'] = self.colorjit_creator.create()

        dummy_color = None
        for i in self.frame_indices:
            color = self.get_color(folder, frame_index + i, side, augments)

            if i == 0:
                if color is None:
                    raise RuntimeError(
                        'The target frame ({}, {}, {}) does not exist.'.format(
                            folder, frame_index, side))

                dummy_color = color

            if color is None:
                inputs[('valid_indices', i)] = torch.zeros((1,), dtype=bool)
                color = dummy_color  # To make a batch, append dummpy tensors.
            else:
                inputs[('valid_indices', i)] = torch.ones((1,), dtype=bool)

            if augments['flip']:
                color = color.transpose(Image.FLIP_LEFT_RIGHT)

            color_aug = color
            if augments['colorjit']:
                color_aug = augments['colorjit_fn'](color_aug)

            base_width, base_height = color.size
            for s in range(self.num_scales):
                h = base_height // (2 ** s)
                w = base_width // (2 ** s)

                resized_color = color.resize((w, h), Image.BILINEAR)
                resized_color_aug = color_aug.resize((w, h), Image.BILINEAR)

                inputs['color', i, s] = self.to_tensor(resized_color)
                inputs['color_aug', i, s] = self.to_tensor(resized_color_aug)

    def store_intrinsic(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str,
            augments: dict):
        K = self.get_intrinsic(folder, frame_index, side, augments)

        # Adjusting intrinsics to match each scale in the pyramid
        for s in range(self.num_scales):
            K = K.copy()

            ref_image = inputs['color', 0, s]
            height, width = ref_image.shape[-2:]

            if augments['flip']:
                K[0, 2] = 1.0 - K[0, 2]

            K[0, :] *= width
            K[1, :] *= height

            inv_K = np.linalg.inv(K)

            inputs['K', s] = torch.from_numpy(K)
            inputs['inv_K', s] = torch.from_numpy(inv_K)

    def store_additional(
            self,
            inputs: dict,
            folder: str,
            frame_idx: int,
            side: str,
            augments: dict):
        return

    def load_image(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _compute_crop_coordinates(
            self,
            src_width,
            src_height,
            h_ratio_per_w: float) -> tuple:
        new_H = int(h_ratio_per_w * src_width)
        dy = (src_height - new_H) // 2
        upper = dy
        lower = src_height - dy
        return upper, lower, new_H

    def hcrop_pil(
            self,
            pil_img: Image.Image,
            h_ratio_per_w: float) -> tuple:
        W, H = pil_img.size
        if H / W == h_ratio_per_w:
            return pil_img, H

        upper, lower, new_H = \
            self._compute_crop_coordinates(W, H, h_ratio_per_w)
        pil_img = pil_img.crop((0, upper, W, lower))
        return pil_img, new_H

    def hcrop_intrinsic(
            self,
            K: np.ndarray,
            src_width: int,
            src_height: int,
            h_ratio_per_w: float) -> tuple:
        if src_height / src_width == h_ratio_per_w:
            return K, src_height

        upper, lower, new_H = self._compute_crop_coordinates(
            src_width, src_height, h_ratio_per_w)

        out_K = K.copy()
        out_K[1, 2] -= upper
        return out_K, new_H
