import os
from collections import Counter

import numpy as np


def read_calib_file(path: str) -> dict:
    '''
    Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    '''
    float_chars = set('0123456789.e+- ')
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data


def load_velodyne_points(filename: str) -> np.ndarray:
    '''
    Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    '''
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def generate_depth_map(
        calib_dir: str,
        velo_filename: str,
        cam: int = 2) -> np.ndarray:
    """
    Generate a depth map by projecting LiDAR points into the image plane.
    """
    cam2cam = read_calib_file(
        os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(
        os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    T_velo2cam = np.eye(4)
    T_velo2cam[:3, :3] = velo2cam['R'].reshape(3, 3)
    T_velo2cam[:3, 3] = velo2cam['T'].flatten()

    raw_shape = cam2cam['S_rect_02'].astype(np.int32)
    img_width, img_height = raw_shape[0], raw_shape[1]

    # compute projection matrix from velodyne to image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0{}'.format(cam)].reshape(3, 4)
    P_velo2img = np.dot(np.dot(P_rect, R_cam2rect), T_velo2cam)

    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]  # only consider forward points

    img_points = np.dot(P_velo2img, velo.T).T
    img_points[:, :2] = img_points[:, :2] / img_points[:, 2:3]

    # Masking to remove invalid points
    img_points[:, 0] = np.round(img_points[:, 0]) - 1
    img_points[:, 1] = np.round(img_points[:, 1]) - 1
    mask = \
        (img_points[:, 0] >= 0) \
        & (img_points[:, 1] >= 0) \
        & (img_points[:, 0] < img_width) \
        & (img_points[:, 1] < img_height)
    img_points = img_points[mask, :].astype(int)

    # project to image
    depth = np.zeros((img_height, img_width))
    x_loc = img_points[:, 0]
    y_loc = img_points[:, 1]
    depth[y_loc, x_loc] = img_points[:, 2]

    # Find the duplicate points and choose the closest depth
    inds = img_points[:, 1] * (depth.shape[1] - 1) + img_points[:, 0] - 1
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(img_points[pts[0], 0])
        y_loc = int(img_points[pts[0], 1])
        depth[y_loc, x_loc] = img_points[pts, 2].min()
    depth[depth < 0] = 0

    return depth


def rotx(t) -> np.ndarray:
    c = np.cos(t)
    s = np.sin(t)
    return np.array(
        [[1,  0,  0],
         [0,  c, -s],
         [0,  s,  c]])


def roty(t) -> np.ndarray:
    c = np.cos(t)
    s = np.sin(t)
    return np.array(
        [[c,  0,  s],
         [0,  1,  0],
         [-s, 0,  c]])


def rotz(t) -> np.ndarray:
    c = np.cos(t)
    s = np.sin(t)
    return np.array(
        [[c, -s,  0],
         [s,  c,  0],
         [0,  0,  1]])


def construct_transform_matrix(
        odometry: np.ndarray,
        scale: np.ndarray) -> np.ndarray:
    """
    Convert the location data measured with OxTS sensors into the matrix.
    It is based on the development kit of kitti.
    """
    lat, lon, alt, roll, pitch, yaw = odometry[:6]

    R = 6378137  # Earth's radius in metres

    # To get translation matrix, use Mercator projection
    tx = scale * lon * np.pi * R / 180
    ty = scale * R * np.log(np.tan((90 + lat) * np.pi / 360))
    tz = alt
    t = np.array([tx, ty, tz])

    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))

    return R, t


def combine_rigid(R=None, t=None) -> np.ndarray:
    """
    Combine a rotation matrix and a translation matrix into a
    rigit transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def load_oxts(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        line = f.readline().strip('\n')
        odometry = list(map(float, line.split(' ')))
    return np.array(odometry)


def load_imu2global_pose(
        oxts_txt_origin: str,
        oxts_txt: str,
        combine: bool = True) -> np.ndarray:
    odom_0 = load_oxts(oxts_txt_origin)
    odom_1 = load_oxts(oxts_txt)

    ref_lat = odom_0[0]
    scale = np.cos(ref_lat * np.pi / 180)

    # R0_body2navi, t0_navi = construct_transform_matrix(odom_0, scale)
    R_body2navi, t_navi = construct_transform_matrix(odom_1, scale)

    if not combine:
        return combine_rigid(R_body2navi), combine_rigid(None, t_navi)
    return combine_rigid(R_body2navi, t_navi)


def load_imu2cam_pose(calib_dir: str) -> np.ndarray:
    cam2cam = read_calib_file(
        os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(
        os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    imu2velo = read_calib_file(
        os.path.join(calib_dir, 'calib_imu_to_velo.txt'))

    T_imu2velo = combine_rigid(imu2velo['R'].reshape(3, 3), imu2velo['T'])
    T_velo2cam = combine_rigid(velo2cam['R'].reshape(3, 3), velo2cam['T'])
    T_cam2rect = combine_rigid(cam2cam['R_rect_00'].reshape(3, 3))
    T_imu2rect = T_cam2rect @ T_velo2cam @ T_imu2velo
    return T_imu2rect


def compute_camera_pose(
        T_imu2rect: np.ndarray,
        T_rect2imu: np.ndarray,
        T0_global: np.ndarray,
        T1_global: np.ndarray) -> np.ndarray:
    """
    The point cloud transformed from velo depends on the orientation
    of a vehicle. The rotation matrix obtained from IMU is transformed
    into the navigation coordinate system from its body coordinate system.
    The "origin" is then shifted using the navigation vector between the
    origin position and the current position.
    Finally, transforms from the navigation coordinate system to the current
    body coordinate system.
    Since the coordinate origin moves, the translation is a vector to the
    center of the global coordinate system.

    Documentation of oxts.
    https://www.oxts.com/wp-content/uploads/2021/06/OxTS-RT500-RT3000-Manual-210622.pdf
    """
    M = np.linalg.solve(T1_global, T0_global)
    T = T_imu2rect.dot(np.dot(M, T_rect2imu))
    return T


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    calib_dir = 'data/kitti/2011_09_26'
    T_imu2rect = load_imu2cam_pose(calib_dir)
    T_rect2imu = np.linalg.inv(T_imu2rect)

    base_dir = 'data/kitti/2011_09_26/2011_09_26_drive_0001_sync/oxts/data'

    oxts0 = os.path.join(base_dir, '0000000000.txt')
    oxts1 = os.path.join(base_dir, '0000000001.txt')

    T0_global, T1_global = load_imu2global_pose(oxts0, oxts1)

    T = compute_camera_pose(T_imu2rect, T_rect2imu, T0_global, T1_global)

    print(T)
