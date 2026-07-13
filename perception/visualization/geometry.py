import numpy as np

from perception.boxes import box_corners_3d
from perception.datasets.kitti_io import velo_to_cam


def project_box_to_image(box, V2C, P2, image_shape=None):
    """Project a canonical LiDAR-frame box onto the image plane.

    Parameters
    ----------
    box : array-like  [x, y, z, l, w, h, yaw]  canonical LiDAR-frame box
    V2C : ndarray 4×4  LiDAR-to-camera transform
    P2  : ndarray 3×4  camera projection matrix
    image_shape : tuple (H, W[, C]), optional
        When provided, corners are clipped to image bounds so out-of-frame
        boxes do not cause the viewer to resize.

    Returns
    -------
    ndarray (8, 2)  pixel coordinates of the eight box corners, ordered per
    perception.boxes.BOX_EDGES, or None if any corner is behind the camera.
    """
    corners_cam = velo_to_cam(box_corners_3d(box), V2C)          # (8, 3)
    corners_hom = np.hstack([corners_cam, np.ones((8, 1))])
    corners_2d  = corners_hom @ P2.T                              # (8, 3)
    if np.any(corners_2d[:, 2] <= 0):
        return None
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]           # (8, 2)

    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        corners_2d[:, 0] = np.clip(corners_2d[:, 0], 0, img_w - 1)
        corners_2d[:, 1] = np.clip(corners_2d[:, 1], 0, img_h - 1)

    return corners_2d
