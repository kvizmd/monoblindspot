import os
from argparse import ArgumentParser
import json
from PIL import Image, ImageDraw

import cv2
import numpy as np

import torch
import tqdm

from bse.datasets import read_calib_file
from bse.utils import \
    RANSAC, \
    BackprojectDepth, \
    PlaneToDepth, \
    fix_random_state


RANDOM_SEED = 42


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str,
        default='data/kitti',
        help='Dataset root dir')
    parser.add_argument(
        '--split', type=str,
        default='splits/kitti/bs/val_files.txt',
        help='Split filename')
    parser.add_argument(
        '--out_dir', type=str,
        default='outputs/ground_depth',
        help='Split filename')
    parser.add_argument(
        '--device', type=str, default='cpu')
    args = parser.parse_args()

    generate_ground_depth(
        args.out_dir, args.data_dir, args.split,
        device=args.device)


def export_split_files(split_filename, folder, frame_index, side):
    with open(split_filename, mode='a') as f:
        f.write(' '.join((folder, frame_index, side)) + '\n')


@torch.no_grad()
def generate_ground_depth(
        output_dir,
        data_dir,
        split_file,
        inlier_meter=0.04,
        max_depth=120,
        max_trial=1000,
        trim_mean_ratio=0.25,
        device='cpu'):
    """
    From the ground mask and LiDAR infomation, the ground plane is
    approximately fitted and the depth of the ground is determined
    by ray tracing.

    Args:
      outputs_dir: The directory to save the 16bit depth png
      data_dir: KITTI root
      split_file: The filename to split KITTI
      img_height: The height of the image
      img_width: The width of the image
      inlier_meter: Distance from the plane that is considered correct
      max_depth: The maximum depth of output depth map.
    """
    fix_random_state(RANDOM_SEED)

    device = torch.device(device)

    with open(split_file, 'r') as f:
        filenames = f.read().splitlines()

    ransac = RANSAC(
        sampling_num=3,
        max_iterations=1000,
        goal_inliers_ratio=0.95,
        inlier_plane_distance=inlier_meter,
        prior_d=None)  # 1.65)
    ransac.to(device)

    os.makedirs(output_dir, exist_ok=True)
    out_splitfile = os.path.join(output_dir, os.path.basename(split_file))
    if os.path.isfile(out_splitfile):
        os.remove(out_splitfile)

    side_map = {'2': 2, '3': 3, 'l': 2, 'r': 3}
    for filename in tqdm.tqdm(filenames, leave=False):
        folder, frame_index, side = filename.split()
        bs_filename = os.path.join(
            data_dir,
            folder,
            'image_0{}'.format(side_map[side]),
            'bs',
            '{:010d}.json'.format(int(frame_index)))

        depth_path = os.path.join(
            data_dir,
            folder,
            'proj_depth',
            'groundtruth',
            'image_0{}'.format(side_map[side]),
            '{:010d}.png'.format(int(frame_index)))

        if not os.path.isfile(depth_path):
            continue

        export_split_files(out_splitfile, folder, frame_index, side)

        depth = Image.open(depth_path)
        depth = np.array(depth).astype(np.float32) / 256
        depth = torch.from_numpy(depth).to(device)

        calib_path = os.path.join(data_dir, folder.split('/')[0])
        cam2cam = read_calib_file(
            os.path.join(calib_path, 'calib_cam_to_cam.txt'))

        raw_shape = cam2cam['S_rect_02'].astype(np.int32)
        img_width, img_height = raw_shape[0], raw_shape[1]

        K = cam2cam['P_rect_0' + str(side_map[side])]
        K = K.reshape(3, 4)[:, :3].astype(np.float32)

        backproject = BackprojectDepth(1, img_height, img_width)
        backproject.to(device)

        plane2depth = PlaneToDepth(
            1, img_height, img_width,
            min_depth=0, max_depth=max_depth,
            negative_depth_value=0)
        plane2depth.to(device)

        inv_K = np.expand_dims(np.linalg.pinv(K), 0).astype(np.float32)
        inv_K = torch.from_numpy(inv_K).to(device)

        # R_rect2cam = np.eye(4, dtype=np.float32)
        # R_rect2cam[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3).T
        # R_rect2cam = torch.from_numpy(R_rect2cam).to(device)

        ground_mask = load_ground_mask(bs_filename, img_height, img_width)
        ground_mask = torch.from_numpy(ground_mask).to(device)

        # Backproject
        cam_points = backproject(depth, inv_K)
        # cam_points = torch.matmul(R_rect2cam, cam_points)

        cam_points = cam_points[:, :3, :]
        cam_points[:, 1, :] *= -1

        # Sampling points on the ground and fit plane.
        ground_mask = ground_mask > 0
        depth_mask = depth > 0
        mask = ground_mask.view(-1) & depth_mask.view(-1)

        ground_cam_points = cam_points[..., mask].transpose(-1, -2)

        ground_depth_list = []
        for mi in tqdm.tqdm(range(max_trial), leave=False):
            plane = ransac.run(ground_cam_points)
            ground_depth = plane2depth(plane, inv_K)

            # Save the depth as the 16bit png.
            ground_depth_list.append(ground_depth.view(-1))

        sample_num = max_trial
        ground_depths = torch.stack(ground_depth_list)

        # To evaluate the quality of the ground depth map, calculate abs-rel.
        depth = depth.view(-1)
        depth_errors = torch.abs(
            ground_depths[:, mask] - depth[mask]) / depth[mask]
        depth_errors = depth_errors.mean(1)

        # Compute trimmed mean in the trial.
        trim = max(1, int(sample_num * trim_mean_ratio))
        inds = torch.argsort(depth_errors)[trim:-trim]
        stacked_ground_depth = ground_depths[inds]

        valid_mask = torch.all(stacked_ground_depth > 0, 0)
        ground_depth = (valid_mask * stacked_ground_depth).mean(0)
        ground_depth = ground_depth.view(img_height, img_width)

        ground_depth = ground_depth.detach().cpu().numpy()
        ground_depth = np.clip(ground_depth, 0, 65535)
        ground_depth = np.uint16(ground_depth * 256)

        out_filename = os.path.join(
            output_dir,
            folder,
            'image_0{}'.format(side_map[side]),
            'ground_depth',
            '{:010d}.png'.format(int(frame_index)))
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        cv2.imwrite(out_filename, ground_depth)


def load_ground_mask(filename, out_height, out_width):
    with open(filename, 'r') as f:
        obj = json.load(f)

    height = obj['imageHeight']
    width = obj['imageWidth']
    shapes = obj['shapes']
    points = []
    for shape in shapes:
        if shape['label'] != 'ground':
            continue

        if shape['shape_type'] != 'polygon':
            continue

        points = shape['points']
        break

    if len(points) == 0:
        return None

    points = [(int(p[0]), int(p[1])) for p in points]

    mask = Image.new('L', (width, height))
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=1)
    mask = mask.resize(
        (out_width, out_height), resample=Image.Resampling.NEAREST)
    mask = np.array(mask, dtype=int)
    return mask


def print_depth_error(depth_errors):
    depth_errors = depth_errors.cpu().numpy()
    print(
        'Depth Error: ',
        'mean: {:.3f} m,'.format(depth_errors.mean()),
        'med: {:.3f} m,'.format(np.median(depth_errors)),
        'min: {:.3f} m,'.format(depth_errors.min()),
        'max: {:.3f} m'.format(depth_errors.max()))


if __name__ == '__main__':
    main()
