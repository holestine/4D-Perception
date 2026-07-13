import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def V2C():
    """Synthetic LiDAR→camera transform: pure axis permutation.

    LiDAR (x fwd, y left, z up) → camera (x right, y down, z fwd):
    x_c = -y_l,  y_c = -z_l,  z_c = x_l
    """
    return np.array([
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [1,  0,  0, 0],
        [0,  0,  0, 1],
    ], dtype=np.float32)


@pytest.fixture
def P2():
    """Synthetic pinhole projection: f=700, principal point (600, 200)."""
    return np.array([
        [700,   0, 600, 0],
        [  0, 700, 200, 0],
        [  0,   0,   1, 0],
    ], dtype=np.float32)
