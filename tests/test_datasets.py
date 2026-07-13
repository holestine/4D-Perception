import cv2
import numpy as np
import pytest

from perception.datasets.kitti import KittiLabelSource, KittiSequence
from perception.detections import OpenPCDetSource
from perception.frame import Detections, Frame

CALIB_TEXT = """P0: 700 0 600 0 0 700 200 0 0 0 1 0
P2: 700 0 600 0 0 700 200 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0
"""

# camera-frame KITTI label [h,w,l,x,y,z,ry] whose canonical LiDAR-frame
# equivalent is centre (10, 2, -0.5), l=4, w=1.8, h=1.5, yaw=-0.4-π/2
CAR_LINE = "Car 0 0 -1.5 0 0 50 50 1.5 1.8 4.0 -2.0 1.25 10.0 0.4 0.9\n"


@pytest.fixture
def kitti_root(tmp_path):
    """Minimal synthetic KITTI tracking tree with one frame of sequence 0008."""
    root = tmp_path / "data"
    (root / "velodyne" / "0008").mkdir(parents=True)
    (root / "image_02" / "0008").mkdir(parents=True)
    (root / "calib").mkdir()
    (root / "pose" / "0008").mkdir(parents=True)

    points = np.array([
        [10.0,  0.0, 0.0, 0.5],   # inside the camera FOV
        [-5.0,  0.0, 0.0, 1.0],   # behind the camera
    ], dtype=np.float32)
    points.tofile(str(root / "velodyne" / "0008" / "000000.bin"))

    cv2.imwrite(str(root / "image_02" / "0008" / "000000.png"),
                np.zeros((8, 8, 3), dtype=np.uint8))
    (root / "calib" / "0008.txt").write_text(CALIB_TEXT)
    (root / "pose" / "0008" / "pose.txt").write_text("1 0 0 5 0 1 0 0 0 0 1 0\n")
    return str(root)


@pytest.fixture
def label_root(tmp_path):
    labels = tmp_path / "labels" / "pvrcnn" / "0008"
    labels.mkdir(parents=True)
    (labels / "000000.txt").write_text(
        CAR_LINE +
        "Misc 0 0 0 0 0 10 10 1 1 1 0 0 5 0 0.8\n"   # filtered out
    )
    return str(tmp_path / "labels" / "pvrcnn")


class TestKittiLabelSource:
    def test_converts_to_canonical(self, kitti_root, label_root):
        source = KittiLabelSource(label_root, seq_id=8, data_root=kitti_root)
        dets = source.get(0)
        assert dets.names == ["Car"]
        np.testing.assert_allclose(dets.scores, [0.9])
        np.testing.assert_allclose(dets.boxes[0, :6], [10.0, 2.0, -0.5, 4.0, 1.8, 1.5], atol=1e-5)
        assert dets.boxes[0, 6] == pytest.approx(-0.4 - np.pi / 2, abs=1e-6)

    def test_empty_label_file(self, kitti_root, tmp_path):
        labels = tmp_path / "empty" / "0008"
        labels.mkdir(parents=True)
        (labels / "000000.txt").write_text("")
        source = KittiLabelSource(str(tmp_path / "empty"), seq_id=8, data_root=kitti_root)
        dets = source.get(0)
        assert len(dets) == 0
        assert dets.boxes.shape == (0, 7)


class TestKittiSequence:
    def test_frame_contents(self, kitti_root, label_root):
        ds = KittiSequence(
            kitti_root, seq_id=8,
            detections=KittiLabelSource(label_root, seq_id=8, data_root=kitti_root),
        )
        assert len(ds) == 1

        frame = ds[0]
        assert isinstance(frame, Frame)
        assert frame.frame_id == 0
        assert frame.points.shape == (1, 4)          # FOV crop drops the rear point
        assert frame.image.shape == (8, 8, 3)
        assert frame.camera.projection.shape == (3, 4)
        assert frame.camera.lidar_to_cam.shape == (4, 4)
        np.testing.assert_allclose(frame.ego_pose[0, 3], 5.0)
        assert len(frame.detections) == 1
        assert frame.detections.names == ["Car"]

    def test_no_detection_source(self, kitti_root):
        frame = KittiSequence(kitti_root, seq_id=8)[0]
        assert len(frame.detections) == 0


class _StubDetector:
    """Stands in for OpenPCDetDetector; counts inference calls."""

    def __init__(self):
        self.calls = 0

    def detect_frame(self, points, frame_id=0):
        self.calls += 1
        boxes = np.array([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1]], dtype=np.float32)
        return boxes, np.array([0.9], dtype=np.float32), ["Car"]


class TestOpenPCDetSource:
    def test_wraps_detector_output(self):
        source = OpenPCDetSource(_StubDetector())
        dets = source.get(0, points=np.zeros((5, 4), dtype=np.float32))
        assert isinstance(dets, Detections)
        assert dets.names == ["Car"]
        assert dets.boxes.shape == (1, 7)

    def test_caches_per_frame(self):
        stub = _StubDetector()
        source = OpenPCDetSource(stub)
        pts = np.zeros((5, 4), dtype=np.float32)
        source.get(0, pts)
        source.get(0, pts)          # cache hit — no second inference
        assert stub.calls == 1
        source.get(1, pts)
        assert stub.calls == 2


def test_detections_empty():
    dets = Detections.empty()
    assert len(dets) == 0
    assert dets.boxes.shape == (0, 7)
    assert dets.scores.shape == (0,)
    assert dets.names == []
