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
from pathlib import Path

import numpy as np
import torch

# Use the local OpenPCDet source so all Python wrappers around compiled ops are present
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "OpenPCDet"))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

_LABEL_MAP = {1: "Car", 2: "Pedestrian", 3: "Cyclist", 4: "Van"}


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

    def detect_frame(self, points, frame_id=0):
        """
        Run detection on a single LiDAR frame.

        OpenPCDet's [x, y, z, dx, dy, dz, heading] output is already the
        canonical box format (see perception/boxes.py), so it is returned
        as-is.

        Args:
            points (ndarray): (N, 4) raw [x, y, z, intensity] LiDAR points.
            frame_id (int):   Frame index passed to the model (default 0).

        Returns:
            boxes  (ndarray): (M, 7) [x, y, z_center, l, w, h, yaw] LiDAR frame.
            scores (ndarray): (M,) detection confidence scores (sigmoid, 0-1).
            names  (list):    (M,) class name strings.
        """
        data_dict = self._dataset.prepare_data({"points": points, "frame_id": frame_id})
        batch = self._dataset.collate_batch([data_dict])
        load_data_to_gpu(batch)

        with torch.no_grad():
            pred_dicts, _ = self._model.forward(batch)

        boxes  = pred_dicts[0]["pred_boxes"].cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].cpu().numpy()
        labels = pred_dicts[0]["pred_labels"].cpu().numpy()

        mask = scores >= self._score_threshold
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        names = [_LABEL_MAP.get(int(l), "Car") for l in labels]
        return boxes.astype(np.float32), scores, names
