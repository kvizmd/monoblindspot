import os

import skimage.transform
from PIL import Image
import numpy as np

import torch

from .dataset import Dataset
from .utils import \
    generate_depth_map, \
    load_imu2cam_pose, \
    load_imu2global_pose, \
    read_calib_file


class KITTIDataset(Dataset):
    def __init__(
            self,
            *args,
            require_depth_gt: bool = True,
            require_adjacent_depth_gt: bool = False,
            require_pose: bool = False,
            **kwargs):
        super().__init__(*args, **kwargs)

        # Averaged intrinsic matrix
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {'2': 2, '3': 3, 'l': 2, 'r': 3}

        self.require_depth_gt = require_depth_gt
        self.require_adjacent_depth_gt = require_adjacent_depth_gt
        self.require_pose = require_pose

        self.dummy_depth_full = torch.zeros(
            [1] + list(self.full_res_shape[::-1]), dtype=torch.float32)

        self.dummy_depth = {}
        for i in range(self.num_scales):
            s = 2 ** i
            h, w = self.height // s, self.width // s
            self.dummy_depth[i] = torch.zeros((1, h, w), dtype=torch.float32)

        self.dummy_pose = torch.eye(4, dtype=torch.float32)

        self.use_calib_intrinsic = True

    def parse_filename(self, filename: str) -> tuple:
        line = filename.split()
        folder = line[0]
        frame_idx = int(line[1])
        side = line[2]
        return folder, frame_idx, side

    def get_color(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: str) -> np.ndarray:
        filepath = os.path.join(
            self.data_path, folder,
            'image_0{}/data'.format(self.side_map[side]),
            '{:010d}{}'.format(frame_index, self.img_ext))

        if not os.path.isfile(filepath):
            return None

        color = self.load_image(filepath)
        color = color.resize(self.full_res_shape, Image.BILINEAR)

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def _get_calib_intrinsic(self, folder: str, side: str) -> np.ndarray:
        # Use correct calibrated intrinsic instead of mean.
        calib_path = os.path.join(self.data_path, folder.split('/')[0])
        cam2cam = read_calib_file(
            os.path.join(calib_path, 'calib_cam_to_cam.txt'))

        raw_shape = cam2cam['S_rect_02'].astype(np.int32)
        K_raw = cam2cam['P_rect_0' + str(self.side_map[side])]
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = K_raw.reshape(3, 4)[:, :3].astype(np.float32)

        K[0, :3] /= raw_shape[0]
        K[1, :3] /= raw_shape[1]

        return K

    def get_intrinsic(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool) -> np.ndarray:
        if self.use_calib_intrinsic:
            K = self._get_calib_intrinsic(folder, side)
        else:
            K = self.K.copy()

        if do_flip:
            K[0, 2] = 1.0 - K[0, 2]

        return K

    def store_additional(
            self,
            inputs: dict,
            folder: str,
            frame_idx: int,
            side: str,
            do_flip: bool,
            do_colorjit: bool):

        if self.require_depth_gt:
            self.store_depth(
                inputs, folder, frame_idx, side, do_flip)

        if self.require_pose:
            self.store_pose(inputs, folder, frame_idx, side)

    def get_depth(
            self,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool) -> np.ndarray:
        calib_path = os.path.join(self.data_path, folder.split('/')[0])
        velo_filename = os.path.join(
            self.data_path,
            folder,
            'velodyne_points',
            'data',
            '{:010d}.bin'.format(int(frame_index)))

        if not os.path.exists(velo_filename):
            return None

        depth_gt = generate_depth_map(
            calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1],
            order=0, preserve_range=True, mode='constant')
        depth_gt = depth_gt.astype(np.float32)

        if do_flip:
            depth_gt = np.fliplr(depth_gt).copy()

        return depth_gt

    def store_depth(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str,
            do_flip: bool):
        for i in self.frame_indices:
            depth_gt = self.get_depth(folder, frame_index + i, side, do_flip)

            if depth_gt is not None:
                depth_gt = torch.from_numpy(np.expand_dims(depth_gt, 0))
                inputs['depth_gt', i] = depth_gt
            else:
                inputs['depth_gt', i] = self.dummy_depth_full

            if i == 0 and not self.require_adjacent_depth_gt:
                break

    def store_pose(
            self,
            inputs: dict,
            folder: str,
            frame_index: int,
            side: str):
        calib_dir = os.path.join(self.data_path, folder.split('/')[0])
        if not os.path.isdir(calib_dir):
            raise RuntimeError('Not found calibration files')
        T_imu2rect = load_imu2cam_pose(calib_dir).astype(np.float32)
        T_rect2imu = np.linalg.inv(T_imu2rect)
        inputs['T_imu2cam'] = torch.from_numpy(T_imu2rect)
        inputs['T_cam2imu'] = torch.from_numpy(T_rect2imu)

        oxts_dir = os.path.join(self.data_path, folder, 'oxts/data')

        src_oxts = os.path.join(
            oxts_dir, '{:010d}.txt'.format(frame_index))
        if not os.path.isfile(src_oxts):
            raise RuntimeError('Not found GPS/IMU files')

        for i in self.frame_indices:
            trg_oxts = os.path.join(
                oxts_dir, '{:010d}.txt'.format(frame_index + i))

            if os.path.isfile(trg_oxts):
                T_imu2global = load_imu2global_pose(src_oxts, trg_oxts)
                T_imu2global = T_imu2global.astype(np.float32)
                inputs['T_imu2global', i] = torch.from_numpy(T_imu2global)
            else:
                inputs['T_imu2global', i] = self.dummy_pose
