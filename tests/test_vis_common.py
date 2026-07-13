import os

import numpy as np

from perception.frame import Detections
from perception.visualization.common import (
    CAR_OBJ_NATIVE_SIZE,
    CAR_OBJ_PATH,
    select_visible,
    track_color,
)


def test_car_obj_asset_exists():
    assert os.path.exists(CAR_OBJ_PATH)
    assert CAR_OBJ_NATIVE_SIZE.shape == (3,)


def test_track_color_stable_and_cycled():
    np.testing.assert_array_equal(track_color(3), track_color(3))
    np.testing.assert_array_equal(track_color(3), track_color(23))  # tab20 cycles
    assert track_color(3).dtype == np.uint8
    assert not np.array_equal(track_color(3), track_color(4))


def _detections(scores):
    n = len(scores)
    return Detections(
        boxes=np.arange(n * 7, dtype=np.float32).reshape(n, 7),
        scores=np.asarray(scores, dtype=np.float32),
        names=[f"obj{k}" for k in range(n)],
    )


def test_select_visible_confirmed_or_high_score():
    dets = _detections([0.9, 5.0, 0.2])
    det_ids = np.array([7, 0, 0])          # only the first is confirmed
    boxes, ids, names = select_visible(dets, det_ids, show_unconfirmed_above=4.0)
    # confirmed (idx 0) and high-scoring unconfirmed (idx 1); idx 2 dropped
    assert list(ids) == [7, 0]
    assert list(names) == ["obj0", "obj1"]
    np.testing.assert_array_equal(boxes[0], dets.boxes[0])


def test_select_visible_empty_frame():
    boxes, ids, names = select_visible(_detections([]), np.zeros(0, dtype=int), 4.0)
    assert len(boxes) == len(ids) == len(names) == 0
