"""Low-level KITTI file I/O: calibration, LiDAR, images, ego poses.

Adapted from multi_object_tracking/dataset/kitti_data_base.py
(https://github.com/abhisheksreesaila/multi-object-tracking).
"""

import re

import cv2
import numpy as np

# KITTI camera 02 image dimensions, used to crop the point cloud to the FOV
IMG_HEIGHT = 374
IMG_WIDTH  = 1241


def read_calib(calib_path):
    """Parse a KITTI calibration file.

    Returns
    -------
    P2  : ndarray (3, 4)  camera-frame 3D → image-pixel projection matrix
    V2C : ndarray (4, 4)  LiDAR-frame → rectified-camera-frame transform
    """
    with open(calib_path) as f:
        for line in f.readlines():
            if line[:2] == "P2":
                P2 = re.split(" ", line.strip())
                P2 = np.array(P2[-12:], np.float32)
                P2 = P2.reshape((3, 4))
            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                vtc_mat = re.split(" ", line.strip())
                vtc_mat = np.array(vtc_mat[-12:], np.float32)
                vtc_mat = vtc_mat.reshape((3, 4))
                vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
            if line[:7] == "R0_rect" or line[:6] == "R_rect":
                R0 = re.split(" ", line.strip())
                R0 = np.array(R0[-9:], np.float32)
                R0 = R0.reshape((3, 3))
                R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
    V2C = np.matmul(R0, vtc_mat)
    return P2, V2C


def read_velodyne(path):
    """Load a raw LiDAR scan.

    Returns
    -------
    ndarray (N, 4)  [x, y, z, intensity] in the LiDAR frame
    """
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def reduce_to_fov(points, P2, V2C):
    """Crop a LiDAR scan to the points that project inside the camera image.

    Parameters
    ----------
    points : ndarray (N, 4)  [x, y, z, intensity] LiDAR points
    P2     : ndarray (3, 4)  from read_calib
    V2C    : ndarray (4, 4)  from read_calib

    Returns
    -------
    ndarray (M, 4)  the subset of points visible to the camera
    """
    points = points[points[:, 0] > 0]  # in front of the sensor
    hom = np.ones((len(points), 4), dtype=np.float32)
    hom[:, :3] = points[:, :3]
    img_pts = hom @ V2C.T @ P2.T
    x = img_pts[:, 0] / img_pts[:, 2]
    y = img_pts[:, 1] / img_pts[:, 2]
    mask = (x >= 0) & (x < IMG_WIDTH) & (y >= 0) & (y < IMG_HEIGHT)
    return points[mask]


def cam_to_velo(cloud, V2C):
    """Transform points from the rectified camera frame to the LiDAR frame.

    Parameters
    ----------
    cloud : ndarray (N, 3+)  camera-frame points (extra columns ignored)
    V2C   : ndarray (4, 4)   from read_calib

    Returns
    -------
    ndarray (N, 3)  LiDAR-frame points
    """
    hom = np.ones((cloud.shape[0], 4), dtype=np.float32)
    hom[:, :3] = cloud[:, :3]
    return (hom @ np.linalg.inv(V2C).T[:, :3]).astype(np.float32)


def velo_to_cam(cloud, V2C):
    """Transform points from the LiDAR frame to the rectified camera frame.

    Parameters
    ----------
    cloud : ndarray (N, 3+)  LiDAR-frame points (extra columns ignored)
    V2C   : ndarray (4, 4)   from read_calib

    Returns
    -------
    ndarray (N, 3)  camera-frame points
    """
    hom = np.ones((cloud.shape[0], 4), dtype=np.float32)
    hom[:, :3] = cloud[:, :3]
    return (hom @ V2C.T[:, :3]).astype(np.float32)


def read_image(path):
    """Load an image file as a BGR ndarray."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


def read_pose(path):
    """Parse a KITTI pose.txt into {frame_id: 4×4 ego pose matrix}."""
    poses = {}
    with open(path) as f:
        for frame_id, line in enumerate(f.readlines()):
            pose = np.array(line.split(" "), dtype=np.float32).reshape((-1, 4))
            poses[frame_id] = np.concatenate([pose, [[0, 0, 0, 1]]])
    return poses
