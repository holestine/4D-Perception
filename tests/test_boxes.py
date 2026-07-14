import numpy as np
import pytest

from perception.boxes import (
    BOX_EDGES,
    box_corners_3d,
    interpolate_boxes,
    kitti_camera_to_lidar,
    lidar_to_kitti_camera,
    register_bbs,
)


class TestKittiCameraToLidar:
    def test_known_values(self, V2C):
        # LiDAR-frame ground truth: centre (10, 2, -0.5), l=4, w=1.8, h=1.5
        # Bottom face centre in LiDAR: (10, 2, -1.25)
        # → camera: x_c = -2, y_c = 1.25, z_c = 10
        ry = 0.4
        kitti = np.array([[1.5, 1.8, 4.0, -2.0, 1.25, 10.0, ry]])

        box = kitti_camera_to_lidar(kitti, V2C)[0]

        np.testing.assert_allclose(box[:3], [10.0, 2.0, -0.5], atol=1e-5)
        np.testing.assert_allclose(box[3:6], [4.0, 1.8, 1.5], atol=1e-6)
        assert box[6] == pytest.approx(-ry - np.pi / 2, abs=1e-6)

    def test_empty_input(self, V2C):
        out = kitti_camera_to_lidar(np.zeros((0, 7), dtype=np.float32), V2C)
        assert out.shape == (0, 7)

    def test_z_is_box_centre_not_bottom(self, V2C):
        h = 2.0
        # bottom face at LiDAR z = 0  →  centre must be at h/2
        kitti = np.array([[h, 1.0, 1.0, 0.0, 0.0, 5.0, 0.0]])
        box = kitti_camera_to_lidar(kitti, V2C)[0]
        assert box[2] == pytest.approx(h / 2, abs=1e-6)


class TestLidarToKittiCamera:
    def test_round_trip(self, V2C):
        lidar = np.array([[10.0, 2.0, -0.5, 4.0, 1.8, 1.5, 0.7],
                          [-3.0, 8.0,  1.2, 3.6, 1.6, 1.4, -2.1]], dtype=np.float32)
        back = kitti_camera_to_lidar(lidar_to_kitti_camera(lidar, V2C), V2C)
        np.testing.assert_allclose(back, lidar, atol=1e-4)

    def test_known_values(self, V2C):
        # Inverse of TestKittiCameraToLidar.test_known_values
        lidar = np.array([[10.0, 2.0, -0.5, 4.0, 1.8, 1.5, -0.4 - np.pi / 2]])
        kitti = lidar_to_kitti_camera(lidar, V2C)[0]
        np.testing.assert_allclose(kitti[:3], [1.5, 1.8, 4.0], atol=1e-6)
        np.testing.assert_allclose(kitti[3:6], [-2.0, 1.25, 10.0], atol=1e-5)
        assert kitti[6] == pytest.approx(0.4, abs=1e-6)

    def test_empty_input(self, V2C):
        out = lidar_to_kitti_camera(np.zeros((0, 7), dtype=np.float32), V2C)
        assert out.shape == (0, 7)


class TestBoxCorners3D:
    def test_axis_aligned_extents(self):
        box = [1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.0]
        c = box_corners_3d(box)
        assert c.shape == (8, 3)
        np.testing.assert_allclose(c.min(axis=0), [1 - 2, 2 - 1, 3 - 0.75])
        np.testing.assert_allclose(c.max(axis=0), [1 + 2, 2 + 1, 3 + 0.75])

    def test_yaw_rotation_swaps_footprint(self):
        box = [0, 0, 0, 4.0, 2.0, 1.0, np.pi / 2]
        c = box_corners_3d(box)
        # after 90° yaw, the length extent lies along y
        assert c[:, 0].max() == pytest.approx(1.0, abs=1e-9)
        assert c[:, 1].max() == pytest.approx(2.0, abs=1e-9)

    def test_edge_lengths_match_extents(self):
        l, w, h = 4.0, 2.0, 1.5
        c = box_corners_3d([5.0, -3.0, 1.0, l, w, h, 0.7])
        lengths = sorted(np.linalg.norm(c[a] - c[b]) for a, b in BOX_EDGES)
        expected = sorted([w] * 4 + [h] * 4 + [l] * 4)
        np.testing.assert_allclose(lengths, expected, atol=1e-9)

    def test_bottom_ring_below_top_ring(self):
        c = box_corners_3d([0, 0, 10.0, 4, 2, 1.5, 0.3])
        assert np.all(c[:4, 2] < c[4:, 2])
        # corner k is vertically aligned with corner k+4
        np.testing.assert_allclose(c[:4, :2], c[4:, :2], atol=1e-9)


class TestInterpolateBoxes:
    def test_midpoint_lerp(self):
        boxes_a = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        boxes_b = np.array([[2.0, 4.0, 1.0, 4.4, 2.2, 1.7, 0.4]])
        out = interpolate_boxes(boxes_a, [7], boxes_b, [7], 0.5)
        np.testing.assert_allclose(out[0], [1.0, 2.0, 0.5, 4.2, 2.1, 1.6, 0.2],
                                   atol=1e-9)

    def test_endpoints(self):
        boxes_a = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1]])
        boxes_b = np.array([[3.0, 1.0, 0.5, 4.0, 2.0, 1.5, 0.3]])
        np.testing.assert_allclose(
            interpolate_boxes(boxes_a, [1], boxes_b, [1], 0.0)[0], boxes_a[0])
        np.testing.assert_allclose(
            interpolate_boxes(boxes_a, [1], boxes_b, [1], 1.0)[0], boxes_b[0])

    def test_yaw_takes_shortest_arc_across_pi(self):
        boxes_a = np.array([[0, 0, 0, 4, 2, 1.5,  3.0]])
        boxes_b = np.array([[0, 0, 0, 4, 2, 1.5, -3.0]])
        out = interpolate_boxes(boxes_a, [1], boxes_b, [1], 0.5)
        # halfway from +3.0 to -3.0 through ±π, not through 0
        assert out[0, 6] == pytest.approx(np.pi, abs=1e-9)

    def test_unmatched_track_held_static(self):
        boxes_a = np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.5]])
        boxes_b = np.array([[9.0, 9.0, 9.0, 4.0, 2.0, 1.5, 0.5]])
        out = interpolate_boxes(boxes_a, [1], boxes_b, [2], 0.5)
        np.testing.assert_allclose(out[0], boxes_a[0])

    def test_unconfirmed_id_zero_never_matched(self):
        boxes_a = np.array([[1.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        boxes_b = np.array([[5.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        out = interpolate_boxes(boxes_a, [0], boxes_b, [0], 0.5)
        np.testing.assert_allclose(out[0], boxes_a[0])

    def test_ego_translation_compensated(self):
        # A world-stationary box seen from a translating ego must stay put
        # in frame a's coordinates for every alpha.
        pose_a = np.eye(4)
        pose_b = np.eye(4)
        pose_b[:3, 3] = [2.0, 0.0, 0.0]           # ego moved 2 m forward
        boxes_a = np.array([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
        boxes_b = np.array([[8.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
        for alpha in (0.25, 0.5, 0.75):
            out = interpolate_boxes(boxes_a, [1], boxes_b, [1], alpha,
                                    pose_a, pose_b)
            np.testing.assert_allclose(out[0], boxes_a[0], atol=1e-9)

    def test_ego_rotation_compensated(self):
        # Same world-stationary box, ego yawed 90°: frame b sees it rotated.
        ang = np.pi / 2
        pose_a = np.eye(4)
        pose_b = np.eye(4)
        pose_b[:2, :2] = [[np.cos(ang), -np.sin(ang)],
                          [np.sin(ang),  np.cos(ang)]]
        boxes_a = np.array([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
        # world (10, 0) in frame b coords: R⁻¹ @ (10, 0) = (0, -10); yaw − ang
        boxes_b = np.array([[0.0, -10.0, 0.0, 4.0, 2.0, 1.5, 0.3 - ang]])
        out = interpolate_boxes(boxes_a, [1], boxes_b, [1], 0.5,
                                pose_a, pose_b)
        np.testing.assert_allclose(out[0], boxes_a[0], atol=1e-9)

    def test_empty_inputs(self):
        empty = np.zeros((0, 7))
        boxes = np.array([[1.0, 0, 0, 4, 2, 1.5, 0.0]])
        assert interpolate_boxes(empty, [], boxes, [1], 0.5).shape == (0, 7)
        np.testing.assert_allclose(
            interpolate_boxes(boxes, [1], empty, [], 0.5), boxes)

    def test_output_does_not_alias_input(self):
        boxes_a = np.array([[1.0, 0, 0, 4, 2, 1.5, 0.0]])
        boxes_b = np.array([[2.0, 0, 0, 4, 2, 1.5, 0.0]])
        out = interpolate_boxes(boxes_a, [1], boxes_b, [1], 0.5)
        out[0, 0] = 99.0
        assert boxes_a[0, 0] == 1.0


class TestRegisterBbs:
    def test_none_pose_is_noop(self):
        boxes = np.random.RandomState(0).rand(3, 7)
        out = register_bbs(boxes, None)
        assert out is boxes

    def test_translation_only(self):
        boxes = np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.5]])
        pose = np.eye(4)
        pose[:3, 3] = [10.0, -5.0, 1.0]
        out = register_bbs(boxes.copy(), pose)
        np.testing.assert_allclose(out[0, :3], [11.0, -3.0, 4.0], atol=1e-9)
        np.testing.assert_allclose(out[0, 3:], boxes[0, 3:], atol=1e-9)

    def test_rotation_adds_yaw(self):
        boxes = np.array([[1.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.2]])
        ang = np.pi / 2
        pose = np.eye(4)
        pose[:2, :2] = [[np.cos(ang), -np.sin(ang)],
                        [np.sin(ang),  np.cos(ang)]]
        out = register_bbs(boxes.copy(), pose)
        np.testing.assert_allclose(out[0, :3], [0.0, 1.0, 0.0], atol=1e-9)
        assert out[0, 6] == pytest.approx(0.2 + ang, abs=1e-9)
