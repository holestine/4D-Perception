"""Canonical 3D bounding-box format and conversions.

Everything inside `perception/` uses one box format — a 7-vector:

    [x, y, z, l, w, h, yaw]

  - (x, y, z): box centre in the LiDAR frame (or world frame after register_bbs)
  - (l, w, h): full extents; l along the heading direction, z-up height h
  - yaw:       rotation about +z; 0 = facing +x

This matches OpenPCDet's native output layout. Dataset adapters and detector
wrappers convert to this format at the boundary; nothing downstream should
need to know the source convention.

register_bbs / get_registration_angle adapted from
multi_object_tracking/tracker/box_op.py
(https://github.com/abhisheksreesaila/multi-object-tracking).
"""

import numpy as np

from perception.datasets.kitti_io import cam_to_velo

# Corner order produced by box_corners_3d: 0-3 bottom ring, 4-7 top ring,
# corner k directly below corner k+4.
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def kitti_camera_to_lidar(boxes_kitti, V2C):
    """Convert KITTI camera-frame label boxes to canonical LiDAR-frame boxes.

    KITTI labels store [h, w, l, x, y, z, ry] with (x, y, z) at the centre of
    the *bottom* face in the rectified camera frame (y down) and ry rotating
    about the camera y-axis.

    Parameters
    ----------
    boxes_kitti : ndarray (N, 7)  [h, w, l, x, y, z, ry] camera frame
    V2C         : ndarray (4, 4)  LiDAR-to-camera transform from read_calib

    Returns
    -------
    ndarray (N, 7)  [x, y, z_center, l, w, h, yaw] LiDAR frame
    """
    boxes_kitti = np.asarray(boxes_kitti, dtype=np.float32).reshape(-1, 7)
    boxes = np.zeros_like(boxes_kitti)
    if len(boxes_kitti) == 0:
        return boxes

    h = boxes_kitti[:, 0]
    boxes[:, :3] = cam_to_velo(boxes_kitti[:, 3:6], V2C)
    boxes[:, 2] += h / 2                          # bottom face → centre
    boxes[:, 3]  = boxes_kitti[:, 2]              # l
    boxes[:, 4]  = boxes_kitti[:, 1]              # w
    boxes[:, 5]  = h
    boxes[:, 6]  = -boxes_kitti[:, 6] - np.pi / 2  # yaw = -ry - π/2
    return boxes


def box_corners_3d(box):
    """Eight corners of a canonical box, in the same frame as the box.

    Returns
    -------
    ndarray (8, 3)  ordered per BOX_EDGES: 0-3 bottom ring, 4-7 top ring
    """
    x, y, z, l, w, h, yaw = np.asarray(box, dtype=np.float64)[:7]

    ring = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    corners = np.zeros((8, 3))
    corners[:4, :2] = ring
    corners[4:, :2] = ring
    corners[:4, 2]  = -h / 2
    corners[4:, 2]  =  h / 2

    c, s = np.cos(yaw), np.sin(yaw)
    corners[:, :2] = corners[:, :2] @ np.array([[c, -s], [s, c]]).T
    return corners + [x, y, z]


def get_registration_angle(mat):
    """Extract the yaw rotation angle from a 4×4 pose matrix."""
    cos_theta = np.clip(mat[0, 0], -1, 1)
    theta_cos = np.arccos(cos_theta)

    if mat[1, 0] >= 0:  # sin(theta)
        return theta_cos
    return 2 * np.pi - theta_cos


def register_bbs(boxes, pose):
    """Transform boxes from the ego frame to the world frame (in place).

    Parameters
    ----------
    boxes : ndarray (N, 7k)  canonical boxes in the ego frame
    pose  : ndarray (4, 4) or None  ego-vehicle pose; None is a no-op

    Returns
    -------
    ndarray (N, 7k)  boxes in the world frame
    """
    if pose is None:
        return boxes

    ang = get_registration_angle(pose)

    t_id = boxes.shape[1] // 7

    ones = np.ones(shape=(boxes.shape[0], 1))
    for i in range(t_id):
        b_id = i * 7
        box_xyz = boxes[:, b_id:b_id + 3]
        box_xyz1 = np.concatenate([box_xyz, ones], -1)

        box_world = np.matmul(box_xyz1, pose.T)

        boxes[:, b_id:b_id + 3] = box_world[:, 0:3]
        boxes[:, b_id + 6] += ang
    return boxes
