import os

import numpy as np
import pytest

from perception.datasets.nuscenes import (
    CATEGORY_MAP,
    global_box_to_lidar,
    transform_matrix,
    yaw_of,
)

DATAROOT = "data/nuscenes"
HAVE_MINI = os.path.isdir(os.path.join(DATAROOT, "v1.0-mini"))


def quat_z(angle):
    """nuScenes-order [w, x, y, z] quaternion for a rotation about +z."""
    return [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]


class TestTransforms:
    def test_translation_only(self):
        T = transform_matrix([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])

    def test_yaw_of_z_rotation(self):
        T = transform_matrix([0, 0, 0], quat_z(np.pi / 2))
        assert yaw_of(T) == pytest.approx(np.pi / 2)


class TestGlobalBoxToLidar:
    def test_identity_transform_reorders_wlh(self):
        box = global_box_to_lidar(
            center=[1.0, 2.0, 3.0], size_wlh=[1.8, 4.2, 1.5],
            rotation_wxyz=quat_z(0.3), T_lidar_from_global=np.eye(4),
        )
        np.testing.assert_allclose(box[:3], [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(box[3:6], [4.2, 1.8, 1.5], atol=1e-6)  # l, w, h
        assert box[6] == pytest.approx(0.3, abs=1e-6)

    def test_rotated_translated_frame(self):
        # LiDAR frame sits at (10, 5, 0) in the world, rotated 90° about z
        T_lidar_to_global = transform_matrix([10.0, 5.0, 0.0], quat_z(np.pi / 2))
        box = global_box_to_lidar(
            center=[10.0, 5.0, 1.0], size_wlh=[2.0, 4.0, 1.5],
            rotation_wxyz=quat_z(np.pi / 2),
            T_lidar_from_global=np.linalg.inv(T_lidar_to_global),
        )
        np.testing.assert_allclose(box[:3], [0.0, 0.0, 1.0], atol=1e-6)
        assert box[6] == pytest.approx(0.0, abs=1e-6)  # world yaw − frame yaw


def test_category_map_covers_tracked_classes():
    assert CATEGORY_MAP["vehicle.car"] == "Car"
    assert CATEGORY_MAP["human.pedestrian.adult"] == "Pedestrian"
    assert CATEGORY_MAP["vehicle.bicycle"] == "Cyclist"
    assert "movable_object.barrier" not in CATEGORY_MAP


@pytest.fixture(scope="module")
def dataset():
    from perception.datasets.nuscenes import NuScenesGTDetections, NuScenesSequence
    ds = NuScenesSequence(DATAROOT, scene=0)
    ds.detections = NuScenesGTDetections(ds)
    return ds


@pytest.mark.skipif(not HAVE_MINI, reason="nuScenes v1.0-mini not downloaded")
class TestNuScenesIntegration:
    def test_frame_contents(self, dataset):
        assert len(dataset) > 30            # mini scenes have ~40 keyframes
        frame = dataset[0]
        assert frame.points.shape[1] == 4
        assert frame.points.shape[0] > 10000
        assert frame.image.shape == (900, 1600, 3)
        assert frame.camera.projection.shape == (3, 4)
        assert frame.camera.lidar_to_cam.shape == (4, 4)
        assert frame.ego_pose.shape == (4, 4)
        assert len(frame.detections) > 0
        assert set(frame.detections.names) <= {"Car", "Van", "Truck", "Pedestrian", "Cyclist"}

    def test_boxes_are_plausibly_local(self, dataset):
        # LiDAR-frame boxes should be within ~150 m of the sensor, not at
        # global map coordinates (hundreds/thousands of metres)
        boxes = dataset[0].detections.boxes
        assert np.all(np.linalg.norm(boxes[:, :2], axis=1) < 150.0)

    def test_tracking_confirms_tracks(self, dataset):
        from perception.tracker.mot import Tracker3D
        tracker = Tracker3D(config={"min_hits": 2, "max_missed": 3,
                                    "dist_threshold": 4.5, "dt": 0.5})
        for i in range(5):
            frame = dataset[i]
            ids, bbs, _, _ = tracker.update(
                frame.detections.boxes, frame.detections.scores,
                pose=frame.ego_pose, names=frame.detections.names,
            )
        assert len(ids) >= 5                 # GT-as-detections tracks confirm quickly

    def test_stitched_scenes_concatenate(self, dataset):
        from perception.datasets.nuscenes import NuScenesSequence
        stitched = NuScenesSequence(DATAROOT, scene=[0, 1])
        other    = NuScenesSequence(DATAROOT, scene=1)
        assert len(stitched) == len(dataset) + len(other)
        assert ".." in stitched.scene_name
        # boundary frames come from the right scenes
        assert stitched.sample_tokens[len(dataset) - 1] == dataset.sample_tokens[-1]
        assert stitched.sample_tokens[len(dataset)] == other.sample_tokens[0]
        frame = stitched[len(dataset)]       # first frame past the boundary loads
        assert frame.frame_id == len(dataset)
        assert frame.points.shape[1] == 4
