import numpy as np
import pytest

from perception.datasets import kitti_io

CALIB_TEXT = """P0: 700 0 600 0 0 700 200 0 0 0 1 0
P2: 700 0 600 0 0 700 200 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0
"""


@pytest.fixture
def calib_file(tmp_path):
    p = tmp_path / "calib.txt"
    p.write_text(CALIB_TEXT)
    return str(p)


class TestReadCalib:
    def test_shapes_and_values(self, calib_file):
        P2, V2C = kitti_io.read_calib(calib_file)
        assert P2.shape == (3, 4)
        assert V2C.shape == (4, 4)
        np.testing.assert_allclose(P2[0], [700, 0, 600, 0])
        # R0 is identity here, so V2C is Tr_velo_to_cam with a [0,0,0,1] row
        np.testing.assert_allclose(V2C[0], [0, -1, 0, 0])
        np.testing.assert_allclose(V2C[3], [0, 0, 0, 1])


class TestReadPose:
    def test_parses_frames(self, tmp_path):
        p = tmp_path / "pose.txt"
        p.write_text("1 0 0 5 0 1 0 -2 0 0 1 0\n1 0 0 6 0 1 0 -2 0 0 1 0\n")
        poses = kitti_io.read_pose(str(p))
        assert set(poses.keys()) == {0, 1}
        assert poses[0].shape == (4, 4)
        np.testing.assert_allclose(poses[0][:, 3], [5, -2, 0, 1])
        np.testing.assert_allclose(poses[1][0, 3], 6)


class TestReadVelodyne:
    @pytest.fixture
    def points(self):
        return np.array([
            [10.0,  0.0, 0.0, 0.5],   # projects to principal point → kept
            [-5.0,  0.0, 0.0, 1.0],   # behind the sensor → dropped by FOV crop
            [10.0, 30.0, 0.0, 1.0],   # projects far outside the image → dropped
        ], dtype=np.float32)

    def test_read_returns_raw_cloud(self, points, tmp_path):
        p = tmp_path / "scan.bin"
        points.tofile(str(p))
        loaded = kitti_io.read_velodyne(str(p))
        np.testing.assert_array_equal(loaded, points)

    def test_reduce_to_fov(self, points, P2, V2C):
        pts = kitti_io.reduce_to_fov(points, P2, V2C)
        assert pts.shape == (1, 4)
        np.testing.assert_allclose(pts[0], [10.0, 0.0, 0.0, 0.5])


class TestFrameTransforms:
    def test_round_trip(self, V2C):
        V2C = V2C.copy()
        V2C[:3, 3] = [0.1, -0.2, 0.3]  # add a translation like real calib
        cloud = np.random.RandomState(0).rand(50, 3).astype(np.float32) * 40
        back = kitti_io.cam_to_velo(kitti_io.velo_to_cam(cloud, V2C), V2C)
        np.testing.assert_allclose(back, cloud, atol=1e-3)

    def test_velo_to_cam_known_point(self, V2C):
        out = kitti_io.velo_to_cam(np.array([[10.0, 2.0, -0.5]]), V2C)
        np.testing.assert_allclose(out[0], [-2.0, 0.5, 10.0], atol=1e-6)


