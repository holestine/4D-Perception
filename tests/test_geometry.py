import numpy as np

from perception.visualization.geometry import project_box_to_image

IMAGE_SHAPE = (375, 1242, 3)


def test_box_in_front_projects_eight_corners(V2C, P2):
    box = [20.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]
    corners = project_box_to_image(box, V2C, P2, IMAGE_SHAPE)
    assert corners.shape == (8, 2)
    assert np.all(np.isfinite(corners))
    # box straight ahead → corners near the principal point (600, 200)
    assert 400 < corners[:, 0].mean() < 800
    assert 100 < corners[:, 1].mean() < 300


def test_box_behind_camera_returns_none(V2C, P2):
    box = [-20.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    assert project_box_to_image(box, V2C, P2, IMAGE_SHAPE) is None


def test_box_straddling_image_plane_returns_none(V2C, P2):
    # centre 1 m ahead but 4 m long → corners on both sides of Z=0
    box = [1.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    assert project_box_to_image(box, V2C, P2, IMAGE_SHAPE) is None


def test_corners_clipped_to_image_bounds(V2C, P2):
    # close and far off-centre → raw projection exceeds image bounds
    box = [5.0, 4.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    h, w = IMAGE_SHAPE[:2]

    unclipped = project_box_to_image(box, V2C, P2)
    assert unclipped[:, 0].min() < 0

    clipped = project_box_to_image(box, V2C, P2, IMAGE_SHAPE)
    assert np.all(clipped[:, 0] >= 0) and np.all(clipped[:, 0] <= w - 1)
    assert np.all(clipped[:, 1] >= 0) and np.all(clipped[:, 1] <= h - 1)
