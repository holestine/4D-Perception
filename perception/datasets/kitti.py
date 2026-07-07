import os
import re

import numpy as np

from multi_object_tracking.dataset import kitti_data_base


class KittiDetectionDataset:
    """Dataset wrapper for a single KITTI tracking sequence.

    Loads LiDAR scans, camera images, calibration matrices, ego-vehicle poses,
    and 3D object detections on demand, one frame at a time.

    Detections can come from either pre-computed .txt files (label_path) or a
    live OpenPCDet model (detector).  Pass exactly one; if both are supplied,
    the live detector takes priority.
    """

    def __init__(self, root_path, seq_id, label_path=None, detector=None):
        """
        Parameters
        ----------
        root_path : str
            Path to the KITTI data directory containing velodyne/, image_02/,
            calib/, and pose/ subdirectories.
        seq_id : int
            Sequence number (e.g. 8 for sequence "0008").
        label_path : str, optional
            Path to the directory of pre-computed detection .txt files.
        detector : OpenPCDetDetector, optional
            Live detector from detector.py.  Results are cached per frame.
        """
        self.seq_name   = str(seq_id).zfill(4)
        self.root_path  = root_path
        self.velo_path  = os.path.join(root_path, "velodyne",  self.seq_name)
        self.image_path = os.path.join(root_path, "image_02",  self.seq_name)
        self.calib_path = os.path.join(root_path, "calib")
        self.label_path = label_path
        self.detector   = detector
        self._det_cache = {}

        pose_path  = os.path.join(root_path, "pose", self.seq_name, "pose.txt")
        self.poses = kitti_data_base.read_pose(pose_path)

    def __len__(self):
        return len(os.listdir(self.velo_path))

    def __getitem__(self, item):
        """Load all sensor data and detections for frame `item`.

        Returns
        -------
        dict with keys:
            frame_id    (int)
            pose        (ndarray 4×4 or None)
            P2          (ndarray 3×4)  camera projection matrix
            V2C         (ndarray 4×4)  LiDAR-to-camera transform
            points      (ndarray N×4)  LiDAR points in sensor frame
            image       (ndarray H×W×3)  BGR camera image
            objects     (ndarray M×7)  [h,w,l,x,y,z,ry] in LiDAR frame
            objects_cam (ndarray M×7)  same box with xyz in camera frame
            scores      (ndarray M,)   detector confidence scores
            names       (list[str])    class label per detection
        """
        name       = str(item).zfill(6)
        velo_path  = os.path.join(self.velo_path,  name + ".bin")
        image_path = os.path.join(self.image_path, name + ".png")
        calib_path = os.path.join(self.calib_path, self.seq_name + ".txt")

        P2, V2C = kitti_data_base.read_calib(calib_path)

        frame = {
            "frame_id":   item,
            "pose":       self.poses.get(item),
            "P2":         P2,
            "V2C":        V2C,
            "points":     kitti_data_base.read_velodyne(velo_path, P2, V2C),
            "image":      kitti_data_base.read_image(image_path),
        }

        if self.detector is not None:
            if item not in self._det_cache:
                self._det_cache[item] = self.detector.detect_frame(
                    velo_path, P2, V2C, frame_id=item
                )
            objects, objects_cam, det_scores, det_names = self._det_cache[item]
        elif self.label_path is not None:
            label_path = os.path.join(self.label_path, self.seq_name, name + ".txt")
            objects, det_scores, det_names = self._read_detection_label(label_path)
            objects_cam = np.array([], dtype=np.float32)
            if len(objects) > 0:
                objects_cam     = np.copy(objects)
                objects[:, 3:6] = kitti_data_base.cam_to_velo(objects[:, 3:6], V2C)[:, :3]
        else:
            objects = objects_cam = np.zeros((0, 7), dtype=np.float32)
            det_scores = np.zeros(0, dtype=np.float32)
            det_names  = []

        frame["objects"]     = objects
        frame["objects_cam"] = objects_cam
        frame["scores"]      = det_scores
        frame["names"]       = det_names
        return frame

    @staticmethod
    def _read_detection_label(label_path):
        """Parse a per-frame KITTI-format detection file.

        Line format:
            class trunc occ alpha x1 y1 x2 y2 h w l x y z ry score

        Returns
        -------
        objects   (ndarray M×7)  [h, w, l, x, y, z, ry] in camera coords
        scores    (ndarray M,)   raw detector confidence scores
        det_names (list[str])    class label strings
        """
        objects_list, det_scores, det_names = [], [], []
        with open(label_path) as f:
            for line in f:
                parts = re.split(" ", line)
                if parts[0] in ("Car", "Truck", "Van", "Cyclist", "Pedestrian"):
                    objects_list.append(parts[8:15])
                    det_scores.append(parts[15])
                    det_names.append(parts[0])
        return (
            np.array(objects_list, dtype=np.float32),
            np.array(det_scores,   dtype=np.float32),
            det_names,
        )
