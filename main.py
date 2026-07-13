"""
3D Multi-Object Tracking pipeline for KITTI sequences.

Pipeline:
  1. Load a KITTI sequence (LiDAR, camera, calibration, pose) with either
     pre-computed detections or a live OpenPCDet model.
  2. Run a SORT-style 3D Kalman filter tracker across every frame.
  3. Visualize confirmed tracks with Rerun and/or export an MP4 video.
"""

import time

import numpy as np

from perception.datasets.kitti import (  # noqa: F401 — KittiLabelSource is Option A
    KittiLabelSource,
    KittiSequence,
)
from perception.detections import OpenPCDetSource
from perception.tracker.mot import Tracker3D
from perception.visualization.rerun_vis import visualize_tracking
from perception.visualization.video import create_tracking_video

# ── Dataset ────────────────────────────────────────────────────────────────────
DATA_ROOT  = "multi_object_tracking/data"
SEQ_ID     = 8

# Option A — pre-computed detections (default):
#detections = KittiLabelSource("multi_object_tracking/detectors/pvrcnn", SEQ_ID, DATA_ROOT)

# Option B — live PV-RCNN inference (requires OpenPCDet + model weights):
from detector import OpenPCDetDetector

detections = OpenPCDetSource(OpenPCDetDetector(
    cfg_file   = "OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml",
    checkpoint = "models/PVRCNN/pv_rcnn_8369.pth",
    data_root  = DATA_ROOT,
))

dataset = KittiSequence(DATA_ROOT, seq_id=SEQ_ID, detections=detections)


# ── Tracker ────────────────────────────────────────────────────────────────────
# Defaults tuned on seq 0008 with evaluate.py (MOTA 0.553 / IDF1 0.731 — see README);
# single-sequence tuning, revisit when more sequences are available
tracker = Tracker3D(config={
    "score_threshold":        0.5,
    "min_hits":               2,
    "max_missed":             3,
    "dist_threshold":         4.5,
    "velocity_process_noise": 1.0,
})


# ── Tracking loop ──────────────────────────────────────────────────────────────
frame_indices = range(len(dataset))

final_bbs     = []
final_ids     = []
final_det_ids = []

elapsed = 0.0
for i in frame_indices:
    frame = dataset[i]

    t0 = time.perf_counter()
    ids, bbs, _, det_ids = tracker.update(
        frame.detections.boxes,
        frame.detections.scores,
        pose=frame.ego_pose,
        names=frame.detections.names,
    )
    elapsed += time.perf_counter() - t0

    final_bbs.append(np.array(bbs) if bbs else np.zeros((0, 7)))
    final_ids.append(ids)
    final_det_ids.append(det_ids)

n = len(frame_indices)
print(f"Tracked {n} frames in {elapsed:.2f}s  ({n / elapsed:.1f} fps)")


# ── Visualization ──────────────────────────────────────────────────────────────
visualize_tracking(
    dataset,
    frame_indices,
    final_det_ids,
    threshold=4,
    out_file="tracking.rrd",
)

create_tracking_video(
    dataset,
    frame_indices,
    final_det_ids,
    threshold=4,
    out_file="tracking.mp4",
)

create_tracking_video(
    dataset,
    range(205, 265),
    final_det_ids,
    threshold=4,
    fps=5,
    out_file="showcase.mp4",
)
