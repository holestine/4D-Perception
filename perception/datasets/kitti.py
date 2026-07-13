import os
import re

import numpy as np

from perception.boxes import kitti_camera_to_lidar
from perception.datasets import kitti_io
from perception.datasets.base import SequenceDataset
from perception.detections import DetectionSource
from perception.frame import Camera, Detections, Frame


class KittiSequence(SequenceDataset):
    """Adapter for a single KITTI tracking sequence.

    Loads LiDAR scans, camera images, calibration, and ego-vehicle poses on
    demand, one frame at a time. Detections come from the given
    DetectionSource (e.g. KittiLabelSource or OpenPCDetSource).
    """

    def __init__(self, root_path, seq_id, detections=None):
        """
        Parameters
        ----------
        root_path : str
            Path to the KITTI data directory containing velodyne/, image_02/,
            calib/, and pose/ subdirectories.
        seq_id : int
            Sequence number (e.g. 8 for sequence "0008").
        detections : DetectionSource, optional
            Where per-frame detections come from; omit for sensor data only.
        """
        self.seq_name   = str(seq_id).zfill(4)
        self.root_path  = root_path
        self.velo_path  = os.path.join(root_path, "velodyne",  self.seq_name)
        self.image_path = os.path.join(root_path, "image_02",  self.seq_name)
        self.calib_path = os.path.join(root_path, "calib", self.seq_name + ".txt")
        self.detections = detections

        pose_path  = os.path.join(root_path, "pose", self.seq_name, "pose.txt")
        self.poses = kitti_io.read_pose(pose_path)

    def __len__(self):
        return len(os.listdir(self.velo_path))

    def __getitem__(self, idx) -> Frame:
        name       = str(idx).zfill(6)
        velo_path  = os.path.join(self.velo_path,  name + ".bin")
        image_path = os.path.join(self.image_path, name + ".png")

        P2, V2C = kitti_io.read_calib(self.calib_path)
        raw_points = kitti_io.read_velodyne(velo_path)

        detections = (
            self.detections.get(idx, raw_points)
            if self.detections is not None else Detections.empty()
        )

        return Frame(
            frame_id=idx,
            points=kitti_io.reduce_to_fov(raw_points, P2, V2C),
            image=kitti_io.read_image(image_path),
            camera=Camera(projection=P2, lidar_to_cam=V2C),
            ego_pose=self.poses.get(idx),
            detections=detections,
        )


class KittiLabelSource(DetectionSource):
    """Pre-computed KITTI-format detection .txt files as a DetectionSource.

    Expects one file per frame at `label_root/<seq>/<frame>.txt` (the layout
    of multi_object_tracking/detectors/). Camera-frame label boxes are
    converted to the canonical format using the sequence calibration.
    """

    _CLASSES = ("Car", "Truck", "Van", "Cyclist", "Pedestrian")

    def __init__(self, label_root, seq_id, data_root):
        """
        Parameters
        ----------
        label_root : str  detector output directory, e.g. ".../detectors/pvrcnn"
        seq_id     : int  sequence number
        data_root  : str  KITTI data directory (for the calibration file)
        """
        self.seq_name   = str(seq_id).zfill(4)
        self.label_path = os.path.join(label_root, self.seq_name)
        calib_file      = os.path.join(data_root, "calib", self.seq_name + ".txt")
        _, self._V2C    = kitti_io.read_calib(calib_file)

    def get(self, frame_id, points=None):
        label_file = os.path.join(self.label_path, str(frame_id).zfill(6) + ".txt")
        boxes_kitti, scores, names = self._read_detection_label(label_file)
        return Detections(
            boxes=kitti_camera_to_lidar(boxes_kitti, self._V2C),
            scores=scores,
            names=names,
        )

    @classmethod
    def _read_detection_label(cls, label_path):
        """Parse a per-frame KITTI-format detection file.

        Line format:
            class trunc occ alpha x1 y1 x2 y2 h w l x y z ry score

        Returns
        -------
        boxes  (ndarray M×7)  [h, w, l, x, y, z, ry] in camera coords
        scores (ndarray M,)   raw detector confidence scores
        names  (list[str])    class label strings
        """
        boxes, scores, names = [], [], []
        with open(label_path) as f:
            for line in f:
                parts = re.split(" ", line)
                if parts[0] in cls._CLASSES:
                    boxes.append(parts[8:15])
                    scores.append(parts[15])
                    names.append(parts[0])
        return (
            np.array(boxes,  dtype=np.float32).reshape(-1, 7),
            np.array(scores, dtype=np.float32),
            names,
        )
