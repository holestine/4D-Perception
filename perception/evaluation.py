"""Tracking evaluation against KITTI tracking ground truth (CLEAR-MOT metrics).

Ground truth comes from the KITTI tracking benchmark's label_02 files
(one file per sequence, camera-frame boxes with persistent track IDs).
Matching is done on bird's-eye-view centre distance in the world frame,
nuScenes-style, so it is independent of box-size estimation quality.

Metrics are computed with the `motmetrics` package: MOTA, MOTP (mean
matched distance in metres), IDF1, ID switches, fragmentations, FP/FN,
and mostly-tracked / mostly-lost trajectory counts.
"""

import numpy as np

# KITTI tracking label columns:
# 0 frame  1 track_id  2 type  3 truncated  4 occluded  5 alpha
# 6-9 bbox2d  10-12 h w l  13-15 x y z (camera frame)  16 rotation_y
_VEHICLE_CLASSES = ("Car", "Van", "Truck")


def read_tracking_labels(path, classes=_VEHICLE_CLASSES):
    """Parse a KITTI tracking label_02 file into per-frame ground truth.

    Parameters
    ----------
    path    : str  label file, e.g. label_02/0008.txt
    classes : tuple[str]  class names to keep (DontCare is always dropped)

    Returns
    -------
    dict  frame_id -> (boxes_kitti (N, 7) [h,w,l,x,y,z,ry] camera frame,
                       track_ids   (N,) int)
    """
    per_frame = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts[2] not in classes:
                continue
            frame_id = int(parts[0])
            box = np.array(parts[10:17], dtype=np.float32)
            boxes, ids = per_frame.setdefault(frame_id, ([], []))
            boxes.append(box)
            ids.append(int(parts[1]))

    return {
        fid: (np.array(boxes, dtype=np.float32).reshape(-1, 7),
              np.array(ids, dtype=int))
        for fid, (boxes, ids) in per_frame.items()
    }


def evaluate_tracking(frames, dist_threshold=2.0):
    """Accumulate CLEAR-MOT metrics over a sequence.

    Parameters
    ----------
    frames : iterable of (gt_ids, gt_xy, pred_ids, pred_xy)
        Per-frame ground-truth / prediction IDs and (N, 2) BEV centre
        positions, all in one common frame (ego or world — matching uses
        distances only, so any rigid frame works as long as it's shared).
    dist_threshold : float
        Maximum centre distance in metres for a GT-prediction match.

    Returns
    -------
    dict of metric name -> value (see _METRICS)
    """
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for gt_ids, gt_xy, pred_ids, pred_xy in frames:
        gt_xy   = np.asarray(gt_xy,   dtype=float).reshape(-1, 2)
        pred_xy = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
        dist = np.linalg.norm(gt_xy[:, None, :] - pred_xy[None, :, :], axis=2)
        dist[dist > dist_threshold] = np.nan   # nan = impossible match
        acc.update(list(gt_ids), list(pred_ids), dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=list(_METRICS), name="seq")
    return {name: summary.iloc[0][name] for name in _METRICS}


_METRICS = (
    "mota", "motp", "idf1", "num_switches", "num_fragmentations",
    "num_false_positives", "num_misses", "num_objects",
    "mostly_tracked", "partially_tracked", "mostly_lost", "num_unique_objects",
)


def format_summary(metrics):
    """Render an evaluate_tracking result as an aligned text table."""
    labels = {
        "mota":                "MOTA",
        "motp":                "MOTP (m)",
        "idf1":                "IDF1",
        "num_switches":        "ID switches",
        "num_fragmentations":  "Fragmentations",
        "num_false_positives": "False positives",
        "num_misses":          "Misses (FN)",
        "num_objects":         "GT boxes",
        "mostly_tracked":      "Mostly tracked",
        "partially_tracked":   "Partially tracked",
        "mostly_lost":         "Mostly lost",
        "num_unique_objects":  "GT trajectories",
    }
    lines = []
    for key, label in labels.items():
        value = metrics[key]
        text = f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(f"{label:<18} {text:>8}")
    return "\n".join(lines)
