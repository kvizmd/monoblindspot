import random
from PIL import Image

import numpy as np

from torchvision import transforms


class Creator:
    def __call__(self, *args, **kwargs):
        return self.create(*args, **kwargs)

    def create(self, *args, **kwargs):
        raise NotImplementedError()


class FixedColorJitter(transforms.ColorJitter):
    def __init__(self, params):
        super().__init__()
        self.fixed_params = params

    def get_params(self, *args, **kwargs):
        return self.fixed_params


class ColorJitterCreator(Creator):
    def __init__(
            self,
            brightness: tuple = (0.8, 1.2),
            contrast: tuple = (0.8, 1.2),
            saturation: tuple = (0.8, 1.2),
            hue: tuple = (-0.1, 0.1)):
        super().__init__()

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self._colorjit = transforms.ColorJitter()

    def create(self) -> FixedColorJitter:
        params = self._colorjit.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)

        # We create the color_aug object in advance and apply the same
        # augmentation to all images in this item. This ensures that all images
        # input to the pose network receive the same augmentation.
        return FixedColorJitter(params)


class FixedRescaleCrop:
    def __init__(
            self,
            img_height: int,
            img_width: int,
            out_height: int,
            out_width: int,
            lower_scale: float = 0.8,
            upper_scale: float = 1.2):

        self.img_height = img_height
        self.img_width = img_width
        self.out_height = out_height
        self.out_width = out_width
        self.lower_scale = lower_scale
        self.upper_scale = upper_scale

        # Parameters for rescaling
        self.scale = random.uniform(self.lower_scale, self.upper_scale)
        self.scaled_width = int(self.img_width * self.scale)
        self.scaled_height = int(self.img_height * self.scale)

        # Parameters for cropping
        crop_dx = max(self.scaled_width - self.out_width, 0)
        crop_dy = max(self.scaled_height - self.out_height, 0)

        self.left = random.randint(0, crop_dx)
        self.upper = random.randint(0, crop_dy)
        self.right = self.left + self.out_width
        self.lower = self.upper + self.out_height

    def apply_to_iamge(self, pil_image: Image) -> Image:
        """
        pil_image: Pillow Image object
        """
        # Rescaling
        pil_image = pil_image.resize(
            (self.scaled_width, self.scaled_height))

        pil_image = pil_image.crop(
            (self.left, self.upper, self.right, self.lower))

        return pil_image

    def apply_to_intrinsic(self, K: np.ndarray) -> np.ndarray:
        """
        K: Normalized camera intrinsic
        """
        K = K.copy()

        K[0, :3] *= self.img_width
        K[1, :3] *= self.img_height

        # Rescaling
        K[:2, :] *= self.scale

        # Cropping
        K[0, 2] -= self.left
        K[1, 2] -= self.upper

        K[0, :3] /= self.out_width
        K[1, :3] /= self.out_height

        return K

    def apply_to_blindspot(self, point: np.ndarray) -> np.ndarray:
        point = point.copy()
        y, x = point[:, 0], point[:, 1]

        y = y * (self.scaled_height - 1) - self.upper
        x = x * (self.scaled_width - 1) - self.left

        mask_img_range = \
            (y >= 0) \
            & (y < min(self.lower, self.out_height)) \
            & (x >= 0) \
            & (x < min(self.right, self.out_width))

        point[:, 0] = (y / (self.out_height - 1)).clip(0.0, 1.0)
        point[:, 1] = (x / (self.out_width - 1)).clip(0.0, 1.0)
        point[~mask_img_range, :] = -1  # ignore
        return point


class RandomRescaleCropCreator(Creator):
    def __init__(
            self,
            out_height: int,
            out_width: int,
            lower_scale: float = 0.8,
            upper_scale: float = 1.2):
        super().__init__()

        self.out_height = out_height
        self.out_width = out_width
        self.lower_scale = lower_scale
        self.upper_scale = upper_scale

    def create(self, img_height, img_width) -> FixedRescaleCrop:
        return FixedRescaleCrop(
                img_height,
                img_width,
                self.out_height,
                self.out_width,
                lower_scale=self.lower_scale,
                upper_scale=self.upper_scale)


class HorizontalFlipBlindSpot:
    def __init__(self):
        super().__init__()

        self.T_grid_flip = np.array([
            -1, 0, 0, 1,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1], dtype=np.float32).reshape(4, 4)

        self.T_cam_flip = np.array([
            -1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1], dtype=np.float32).reshape(4, 4)

    def __call__(self, points, T_ogm2cam) -> np.ndarray:
        points = points.copy()

        mask = points[:, 1] >= 0
        points[mask, 1] = 1.0 - points[mask, 1]

        if points.shape[-1] > 3:
            points[mask, 3] = 1.0 - points[mask, 3]

        T_ogm2cam = self.T_cam_flip @ T_ogm2cam @ self.T_grid_flip

        return points, T_ogm2cam
