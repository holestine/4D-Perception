"""Tracking evaluation against KITTI tracking ground truth.

Ground truth comes from the KITTI tracking benchmark's label_02 files
(one file per sequence, camera-frame boxes with persistent track IDs).
Matching is done on bird's-eye-view centre distance in the world frame,
nuScenes-style, so it is independent of box-size estimation quality.

Two metric families:
- CLEAR-MOT (via the `motmetrics` package): MOTA, MOTP, IDF1, ID switches,
  fragmentations, FP/FN, mostly-tracked / mostly-lost trajectory counts.
- HOTA (implemented here, following Luiten et al., IJCV 2021): decomposes
  into detection accuracy (DetA) and association accuracy (AssA), averaged
  over a sweep of localization thresholds — unlike MOTA, association
  quality carries equal weight to detection quality.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

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


def evaluate_hota(frames, max_dist=4.0, alphas=None):
    """HOTA with BEV-centre-distance similarity (Luiten et al., IJCV 2021).

    Similarity between a ground-truth object and a prediction is
    ``max(0, 1 - distance / max_dist)``, so the alpha sweep corresponds to
    distance gates: alpha 0.5 with the default max_dist of 4 m matches the
    2 m CLEAR-MOT gate.

    Parameters
    ----------
    frames : iterable of (gt_ids, gt_xy, pred_ids, pred_xy)
        Same per-frame tuples as evaluate_tracking.
    max_dist : float
        Distance in metres at which similarity reaches zero.
    alphas : ndarray, optional
        Similarity thresholds to average over (default 0.05 … 0.95).

    Returns
    -------
    dict with hota / det_a / ass_a / loc_a, each averaged over alphas.
    """
    if alphas is None:
        alphas = np.arange(0.05, 0.96, 0.05)

    # Collect the sequence and relabel IDs to contiguous indices
    gt_map, pred_map = {}, {}
    timesteps = []
    for gt_ids, gt_xy, pred_ids, pred_xy in frames:
        gt_idx   = np.array([gt_map.setdefault(g, len(gt_map)) for g in gt_ids], dtype=int)
        pred_idx = np.array([pred_map.setdefault(p, len(pred_map)) for p in pred_ids], dtype=int)
        gt_xy    = np.asarray(gt_xy,   dtype=float).reshape(-1, 2)
        pred_xy  = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
        dist = np.linalg.norm(gt_xy[:, None, :] - pred_xy[None, :, :], axis=2)
        sim  = np.maximum(0.0, 1.0 - dist / max_dist)
        timesteps.append((gt_idx, pred_idx, sim))

    n_gt_ids, n_pred_ids = len(gt_map), len(pred_map)
    num_gt_dets   = sum(len(g) for g, _, _ in timesteps)
    num_pred_dets = sum(len(p) for _, p, _ in timesteps)
    if n_gt_ids == 0 and n_pred_ids == 0:      # vacuously perfect
        return {"hota": 1.0, "det_a": 1.0, "ass_a": 1.0, "loc_a": 1.0}
    if n_gt_ids == 0 or n_pred_ids == 0:       # all misses or all false positives
        return {"hota": 0.0, "det_a": 0.0, "ass_a": 0.0, "loc_a": 0.0}

    # Pass 1 — global alignment: how consistently each GT/prediction ID pair
    # co-occurs, used to bias per-frame matching toward stable identities
    potential = np.zeros((n_gt_ids, n_pred_ids))
    gt_count   = np.zeros(n_gt_ids)
    pred_count = np.zeros(n_pred_ids)
    for gt_idx, pred_idx, sim in timesteps:
        if len(gt_idx) and len(pred_idx):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            ratio = np.divide(sim, denom, out=np.zeros_like(sim), where=denom > 1e-12)
            potential[gt_idx[:, None], pred_idx[None, :]] += ratio
        gt_count[gt_idx]     += 1
        pred_count[pred_idx] += 1

    alignment = potential / (gt_count[:, None] + pred_count[None, :] - potential)

    # Pass 2 — per-frame Hungarian matching, then per-alpha TP counting
    n_a = len(alphas)
    tp        = np.zeros(n_a)
    loc_sum   = np.zeros(n_a)
    matches   = np.zeros((n_a, n_gt_ids, n_pred_ids))
    for gt_idx, pred_idx, sim in timesteps:
        if not (len(gt_idx) and len(pred_idx)):
            continue
        score = alignment[gt_idx[:, None], pred_idx[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(alphas):
            ok = sim[rows, cols] >= alpha - 1e-12
            tp[a]      += ok.sum()
            loc_sum[a] += sim[rows[ok], cols[ok]].sum()
            matches[a, gt_idx[rows[ok]], pred_idx[cols[ok]]] += 1

    fn = num_gt_dets - tp
    fp = num_pred_dets - tp
    det_a = tp / np.maximum(1.0, tp + fn + fp)

    # per-pair association Jaccard, weighted by how often the pair matched
    pair_union = gt_count[:, None] + pred_count[None, :] - matches
    ass_scores = matches / np.maximum(1.0, pair_union)
    ass_a = (matches * ass_scores).sum(axis=(1, 2)) / np.maximum(1.0, tp)

    hota = np.sqrt(det_a * ass_a)

    return {
        "hota":  hota.mean(),
        "det_a": det_a.mean(),
        "ass_a": ass_a.mean(),
        # mean localization similarity over every TP in the alpha sweep
        "loc_a": loc_sum.sum() / max(1.0, tp.sum()),
    }


def read_nuscenes_gt(sequence, classes=_VEHICLE_CLASSES):
    """Extract per-frame ground truth from a NuScenesSequence.

    Parameters
    ----------
    sequence : NuScenesSequence
    classes  : tuple[str]  canonical class names to keep (matched via CATEGORY_MAP)

    Returns
    -------
    dict  frame_id -> (global_xy (N, 2) float64, instance_ids (N,) int)
        Positions are in the nuScenes global frame — the same frame the tracker
        registers boxes into via ego_pose — so distances are directly comparable.
    """
    from perception.datasets.nuscenes import CATEGORY_MAP

    classes_set = set(classes)
    instance_map = {}   # instance_token (str) → stable int ID

    per_frame = {}
    for frame_id, sample_token in enumerate(sequence.sample_tokens):
        anns = sequence.tables.annotations.get(sample_token, [])
        xys, ids = [], []
        for ann in anns:
            category = sequence.tables.category[
                sequence.tables.instance[ann["instance_token"]]["category_token"]
            ]
            if CATEGORY_MAP.get(category["name"]) not in classes_set:
                continue
            xys.append(ann["translation"][:2])
            ids.append(instance_map.setdefault(ann["instance_token"], len(instance_map)))
        per_frame[frame_id] = (
            np.array(xys, dtype=float).reshape(-1, 2),
            np.array(ids, dtype=int),
        )
    return per_frame


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
