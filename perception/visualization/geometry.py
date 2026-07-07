import numpy as np


def project_3d_box_to_image(bbox_3d, P2, image_shape=None):
    """Project a 3D bounding box from camera coordinates onto the image plane.

    Parameters
    ----------
    bbox_3d : array-like  [h, w, l, x, y, z, ry]  camera coords; y at bottom face
    P2      : ndarray 3×4  camera projection matrix
    image_shape : tuple (H, W[, C]), optional
        When provided, corners are clipped to image bounds so out-of-frame
        boxes do not cause the viewer to resize.

    Returns
    -------
    ndarray (8, 2)  pixel coordinates of the eight box corners
    """
    h, w, l, x, y, z, ry = bbox_3d

    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [   0,    0,    0,    0,   -h,   -h,   -h,   -h ]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    R_y = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    corners_3d = R_y @ corners_3d + np.array([[x], [y], [z]])

    corners_3d_hom = np.vstack((corners_3d, np.ones((1, 8))))
    corners_2d     = P2 @ corners_3d_hom
    if np.any(corners_2d[2] <= 0):
        return None
    corners_2d     = (corners_2d[:2] / corners_2d[2]).T   # (8, 2)

    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        corners_2d[:, 0] = np.clip(corners_2d[:, 0], 0, img_w - 1)
        corners_2d[:, 1] = np.clip(corners_2d[:, 1], 0, img_h - 1)

    return corners_2d
