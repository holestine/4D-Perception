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

from perception.datasets.kitti import KittiDetectionDataset
from perception.tracker.mot import Tracker3D
from perception.visualization.rerun_vis import visualize_tracking
from perception.visualization.video import create_tracking_video


# ── Dataset ────────────────────────────────────────────────────────────────────
DATA_ROOT  = "multi_object_tracking/data"
SEQ_ID     = 8

# Option A — pre-computed detections (default):
#LABEL_PATH = "multi_object_tracking/detectors/pvrcnn"
#dataset = KittiDetectionDataset(DATA_ROOT, seq_id=SEQ_ID, label_path=LABEL_PATH)

# Option B — live PV-RCNN inference (requires OpenPCDet + model weights):
from detector import OpenPCDetDetector
det = OpenPCDetDetector(
    cfg_file   = "OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml",
    checkpoint = "models/PVRCNN/pv_rcnn_8369.pth",
    data_root  = DATA_ROOT,
)
dataset = KittiDetectionDataset(DATA_ROOT, seq_id=SEQ_ID, detector=det)


# ── Tracker ────────────────────────────────────────────────────────────────────
tracker = Tracker3D(config={
    "score_threshold":        0.5,
    "min_hits":               3,
    "max_missed":             5,
    "dist_threshold":         6.0,
    "velocity_process_noise": 1.0,
})


# ── Tracking loop ──────────────────────────────────────────────────────────────
all_frames = range(len(dataset))

final_bbs     = []
final_ids     = []
final_det_ids = []

elapsed = 0.0
for i in all_frames:
    data = dataset[i]

    scores  = np.array(data["scores"], dtype=float)
    objects = data["objects"]

    full_n           = len(scores)
    mask             = scores > tracker.score_threshold
    filtered_objects = objects[mask, :7]
    filtered_scores  = scores[mask]

    t0 = time.perf_counter()
    ids, bbs, _, filtered_det_ids = tracker.update(
        filtered_objects, filtered_scores, pose=data["pose"]
    )
    elapsed += time.perf_counter() - t0

    det_ids       = np.zeros(full_n, dtype=int)
    det_ids[mask] = filtered_det_ids

    final_bbs.append(np.array(bbs) if bbs else np.zeros((0, 7)))
    final_ids.append(ids)
    final_det_ids.append(det_ids)

n = len(all_frames)
print(f"Tracked {n} frames in {elapsed:.2f}s  ({n / elapsed:.1f} fps)")


# ── Visualization ──────────────────────────────────────────────────────────────
visualize_tracking(
    dataset,
    all_frames,
    final_det_ids,
    threshold=4,
    out_file="tracking.rrd",
)

create_tracking_video(
    dataset,
    all_frames,
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
