import numpy as np

from perception.visualization.rerun_vis import _mask_points_outside_boxes


def test_points_inside_box_are_masked():
    boxes = [(np.array([10.0, 0.0, 0.0]), 4.0, 2.0, 1.5, 0.0)]
    points = np.array([
        [10.0, 0.0, 0.0],    # centre → inside
        [50.0, 0.0, 0.0],    # far away → outside
    ])
    keep = _mask_points_outside_boxes(points, boxes)
    np.testing.assert_array_equal(keep, [False, True])


def test_margin_extends_the_box():
    boxes = [(np.array([0.0, 0.0, 0.0]), 4.0, 2.0, 1.5, 0.0)]
    just_outside = np.array([[2.1, 0.0, 0.0]])  # 0.1 m beyond the l/2 face
    assert not _mask_points_outside_boxes(just_outside, boxes, margin=0.3)[0]
    assert _mask_points_outside_boxes(just_outside, boxes, margin=0.0)[0]


def test_yawed_box():
    # box rotated 90° → its 4 m length lies along y
    boxes = [(np.array([0.0, 0.0, 0.0]), 4.0, 2.0, 1.5, np.pi / 2)]
    points = np.array([
        [0.0, 1.8, 0.0],   # inside (within l/2 along y)
        [1.8, 0.0, 0.0],   # outside (beyond w/2 + margin along x)
    ])
    keep = _mask_points_outside_boxes(points, boxes, margin=0.3)
    np.testing.assert_array_equal(keep, [False, True])


def test_no_boxes_keeps_everything():
    points = np.random.RandomState(0).rand(10, 3)
    assert _mask_points_outside_boxes(points, []).all()
