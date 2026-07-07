"""
Model-agnostic OpenPCDet 3D detector for KITTI sequences.

Supports any OpenPCDet model (PointRCNN, PV-RCNN, CenterPoint, etc.) —
swap cfg_file and checkpoint to change model.

Requires OpenPCDet to be installed:
    pip install spconv-cu118          # already installed
    git clone https://github.com/open-mmlab/OpenPCDet
    cd OpenPCDet && pip install -e .

Model configs and pretrained weights:
    https://github.com/open-mmlab/OpenPCDet/blob/master/docs/MODEL_ZOO.md

Example:
    from detector import OpenPCDetDetector
    det = OpenPCDetDetector(
        cfg_file="tools/cfgs/kitti_models/pointrcnn.yaml",
        checkpoint="pointrcnn_7870.pth",
        data_root="multi_object_tracking/data",
    )
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Use the local OpenPCDet source so all Python wrappers around compiled ops are present
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "OpenPCDet"))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


_LABEL_MAP = {1: "Car", 2: "Pedestrian", 3: "Cyclist", 4: "Van"}


def _limit_period(val, offset=0.5, period=np.pi):
    """Wrap val into [-offset*period, (1-offset)*period]."""
    return val - np.floor(val / period + offset) * period


class _InferenceDataset(DatasetTemplate):
    """Minimal DatasetTemplate — exists only to provide collate_batch preprocessing."""

    def __init__(self, cfg, root_path):
        super().__init__(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=Path(root_path),
        )

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return {}


class OpenPCDetDetector:
    """
    Wraps any OpenPCDet model for per-frame KITTI 3D detection.

    Output format matches the pre-computed .txt files loaded by KittiDetectionDataset,
    making it a drop-in replacement for file-based loading.

    Args:
        cfg_file        (str):   Path to the model YAML config.
        checkpoint      (str):   Path to the model weights (.pth).
        data_root       (str):   KITTI data root (needed by DatasetTemplate init).
        score_threshold (float): Minimum confidence to return (default 0.3).
    """

    def __init__(self, cfg_file, checkpoint, data_root, score_threshold=0.3):
        cfg_from_yaml_file(cfg_file, cfg)
        self._score_threshold = score_threshold

        logger = common_utils.create_logger()
        self._dataset = _InferenceDataset(cfg, data_root)

        self._model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=self._dataset,
        )
        self._model.load_params_from_file(filename=checkpoint, logger=logger, to_cpu=True)
        self._model.cuda()
        self._model.eval()

    def detect_frame(self, velo_bin_path, P2, V2C, frame_id=0):
        """
        Run detection on a single LiDAR frame.

        Args:
            velo_bin_path (str):     Path to the raw .bin point cloud file.
            P2  (ndarray, 3×4):      Camera projection matrix from read_calib.
            V2C (ndarray, 4×4):      LiDAR-to-camera transform from read_calib.
            frame_id (int):          Frame index passed to the model (default 0).

        Returns:
            objects_lidar (ndarray): (M, 7) [h,w,l, x,y,z_bottom, ry] LiDAR frame.
            objects_cam   (ndarray): (M, 7) [h,w,l, x,y_bottom,z, ry] camera frame.
            scores        (ndarray): (M,) detection confidence scores.
            names         (list):    (M,) class name strings.
        """
        points = np.fromfile(velo_bin_path, dtype=np.float32).reshape(-1, 4)
        data_dict = self._dataset.prepare_data({"points": points, "frame_id": frame_id})
        batch = self._dataset.collate_batch([data_dict])
        load_data_to_gpu(batch)

        with torch.no_grad():
            pred_dicts, _ = self._model.forward(batch)

        bboxes = pred_dicts[0]["pred_boxes"].cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].cpu().numpy()
        labels = pred_dicts[0]["pred_labels"].cpu().numpy()

        mask = scores >= self._score_threshold
        bboxes, scores, labels = bboxes[mask], scores[mask], labels[mask]

        if len(bboxes) == 0:
            empty = np.zeros((0, 7), dtype=np.float32)
            return empty, empty, np.zeros(0, dtype=np.float32), []

        objects_lidar, objects_cam = self._to_kitti_format(bboxes, V2C)
        names = [_LABEL_MAP.get(int(l), "Car") for l in labels]
        return objects_lidar, objects_cam, scores, names

    def _to_kitti_format(self, bboxes, V2C):
        """
        Convert OpenPCDet [x,y,z,dx,dy,dz,heading] → KITTI [h,w,l, x,y,z_bottom, ry].

        OpenPCDet: LiDAR frame (x=fwd, y=left, z=up), box centres.
        KITTI:     camera frame (x=right, y=down, z=fwd), y at bottom face.
        """
        n = len(bboxes)
        objects_lidar = np.zeros((n, 7), dtype=np.float32)
        objects_cam   = np.zeros((n, 7), dtype=np.float32)

        for i, (x_l, y_l, z_l, dx, dy, dz, heading) in enumerate(bboxes):
            h, w, l = dz, dy, dx
            ry = _limit_period(-heading - np.pi / 2, period=2 * np.pi)

            x_c, y_c, z_c = (V2C @ np.array([x_l, y_l, z_l, 1.0], dtype=np.float64))[:3]

            # Camera y increases downward; KITTI stores y at the bottom face
            y_c_bottom = y_c + h / 2
            # LiDAR z increases upward; bottom face is centre minus half-height
            z_l_bottom = z_l - h / 2

            objects_lidar[i] = [h, w, l, x_l, y_l, z_l_bottom, ry]
            objects_cam[i]   = [h, w, l, x_c, y_c_bottom, z_c, ry]

        return objects_lidar, objects_cam
