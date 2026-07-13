"""Dataset-agnostic data model: one sensor frame and its detections.

Dataset adapters (perception/datasets/) produce Frame objects; the tracker
and visualizers consume them. Nothing downstream of an adapter should need
to know dataset-specific names like KITTI's P2 or Tr_velo_to_cam.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Camera:
    """A calibrated camera.

    Attributes
    ----------
    projection   : ndarray (3, 4)  camera-frame points → homogeneous pixels
    lidar_to_cam : ndarray (4, 4)  LiDAR-frame → camera-frame transform
    """
    projection:   np.ndarray
    lidar_to_cam: np.ndarray


@dataclass
class Detections:
    """3D detections for one frame, in the canonical box format.

    Attributes
    ----------
    boxes  : ndarray (M, 7)  [x, y, z_center, l, w, h, yaw] LiDAR frame
    scores : ndarray (M,)    detector confidence scores
    names  : list[str]       class label per detection
    """
    boxes:  np.ndarray
    scores: np.ndarray
    names:  list

    @classmethod
    def empty(cls):
        return cls(
            boxes=np.zeros((0, 7), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            names=[],
        )

    def __len__(self):
        return len(self.boxes)


@dataclass
class Frame:
    """All sensor data and detections for one timestep of a sequence.

    Attributes
    ----------
    frame_id   : int
    points     : ndarray (N, 4)  [x, y, z, intensity] LiDAR points
    image      : ndarray (H, W, 3)  BGR camera image
    camera     : Camera
    ego_pose   : ndarray (4, 4) or None  ego-vehicle pose in the world frame
    detections : Detections
    """
    frame_id:   int
    points:     np.ndarray
    image:      np.ndarray
    camera:     Camera
    ego_pose:   np.ndarray = None
    detections: Detections = field(default_factory=Detections.empty)
