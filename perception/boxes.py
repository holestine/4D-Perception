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

from perception.datasets.kitti_io import cam_to_velo, velo_to_cam

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


def lidar_to_kitti_camera(boxes, V2C):
    """Convert canonical LiDAR-frame boxes to KITTI camera-frame label boxes.

    Inverse of kitti_camera_to_lidar: box centres drop to the bottom-face
    convention and move to the rectified camera frame, yaw maps back to ry.

    Parameters
    ----------
    boxes : ndarray (N, 7)  [x, y, z_center, l, w, h, yaw] LiDAR frame
    V2C   : ndarray (4, 4)  LiDAR-to-camera transform from read_calib

    Returns
    -------
    ndarray (N, 7)  [h, w, l, x, y, z_bottom, ry] camera frame
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    kitti = np.zeros_like(boxes)
    if len(boxes) == 0:
        return kitti

    bottom = boxes[:, :3].copy()
    bottom[:, 2] -= boxes[:, 5] / 2               # centre → bottom face
    kitti[:, 0]   = boxes[:, 5]                   # h
    kitti[:, 1]   = boxes[:, 4]                   # w
    kitti[:, 2]   = boxes[:, 3]                   # l
    kitti[:, 3:6] = velo_to_cam(bottom, V2C)
    kitti[:, 6]   = -boxes[:, 6] - np.pi / 2      # ry = -yaw - π/2 (self-inverse)
    return kitti


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


def interpolate_boxes(boxes_a, ids_a, boxes_b, ids_b, alpha, pose_a=None, pose_b=None):
    """Interpolate per-track boxes between two frames, in frame a's ego frame.

    Boxes are matched by track ID: matched pairs lerp centre and extents and
    take the shortest arc in yaw. Tracks present only in frame a are held
    static; tracks present only in frame b are omitted (they appear when
    their own frame is rendered). Ego motion between the frames is
    compensated first, so a vehicle that is stationary in the world stays
    put on screen while the ego moves.

    Parameters
    ----------
    boxes_a : ndarray (Na, 7)  canonical boxes in frame a's ego frame
    ids_a   : ndarray (Na,)    track IDs (0 = unconfirmed, never matched)
    boxes_b : ndarray (Nb, 7)  canonical boxes in frame b's ego frame
    ids_b   : ndarray (Nb,)    track IDs
    alpha   : float            interpolation fraction, 0 = frame a, 1 = frame b
    pose_a, pose_b : ndarray (4, 4) or None
        ego poses; if either is None, ego-motion compensation is skipped

    Returns
    -------
    ndarray (Na, 7)  interpolated boxes, aligned with the boxes_a order
    """
    out = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 7).copy()
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 7)
    if len(out) == 0 or len(boxes_b) == 0:
        return out

    if pose_a is not None and pose_b is not None:
        boxes_b = register_bbs(boxes_b.copy(), pose_b)     # frame b ego → world
        inv_a = np.linalg.inv(pose_a)                      # world → frame a ego
        xyz1 = np.concatenate([boxes_b[:, :3], np.ones((len(boxes_b), 1))], -1)
        boxes_b[:, :3] = (xyz1 @ inv_a.T)[:, :3]
        boxes_b[:, 6] -= get_registration_angle(pose_a)

    box_by_id = {int(t): b for t, b in zip(ids_b, boxes_b) if t > 0}
    for i, track_id in enumerate(ids_a):
        box_b = box_by_id.get(int(track_id))
        if box_b is None:
            continue
        out[i, :6] += alpha * (box_b[:6] - out[i, :6])
        d_yaw = (box_b[6] - out[i, 6] + np.pi) % (2 * np.pi) - np.pi
        out[i, 6] += alpha * d_yaw
    return out


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
