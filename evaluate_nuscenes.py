"""
Evaluate the 3D MOT tracker against nuScenes ground-truth annotations.

Uses GT annotations as detections (no nuScenes-trained detector yet), so the
numbers reflect association quality rather than end-to-end detection + tracking.

    python evaluate_nuscenes.py                  # scene 0, v1.0-mini
    python evaluate_nuscenes.py --scene 1
    python evaluate_nuscenes.py --dist-threshold 2.0
"""

import argparse

import numpy as np

from perception.cli import add_tracker_args, build_tracker
from perception.datasets.nuscenes import NuScenesGTDetections, NuScenesSequence
from perception.evaluation import (
    _VEHICLE_CLASSES,
    evaluate_hota,
    evaluate_tracking,
    format_summary,
    read_nuscenes_gt,
)
from perception.frame import Detections

_DT = 0.5   # nuScenes keyframes at 2 Hz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/nuscenes",
                   help="nuScenes root containing v1.0-mini/, samples/, sweeps/")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default="0",
                   help="scene index (int) or name, e.g. scene-0061")
    p.add_argument("--dist-threshold", type=float, default=2.0,
                   help="max BEV centre distance in metres for a GT match")
    add_tracker_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        scene = int(args.scene)
    except ValueError:
        scene = args.scene

    dataset = NuScenesSequence(args.data_root, scene=scene, version=args.version)
    dataset.detections = NuScenesGTDetections(dataset)
    tracker = build_tracker(args, dt=_DT)
    gt = read_nuscenes_gt(dataset)

    vehicle_set = set(_VEHICLE_CLASSES)

    def frames():
        for i in range(len(dataset)):
            frame = dataset[i]
            # Keep only vehicle-class detections so non-vehicle confirmed tracks
            # don't inflate FP when evaluated against vehicle-only GT.
            mask = [n in vehicle_set for n in frame.detections.names]
            mask = np.array(mask, dtype=bool)
            dets = Detections(
                boxes=frame.detections.boxes[mask],
                scores=frame.detections.scores[mask],
                names=[n for n, m in zip(frame.detections.names, mask) if m],
            )
            pred_ids, bbs, _, _ = tracker.update(
                dets.boxes, dets.scores,
                pose=frame.ego_pose, names=dets.names,
            )
            pred_xy = np.array(bbs)[:, :2] if bbs else np.zeros((0, 2))
            gt_xy, gt_ids = gt.get(i, (np.zeros((0, 2)), np.zeros(0, dtype=int)))
            yield gt_ids, gt_xy, pred_ids, pred_xy

    per_frame = list(frames())
    metrics = evaluate_tracking(per_frame, dist_threshold=args.dist_threshold)
    hota    = evaluate_hota(per_frame, max_dist=2 * args.dist_threshold)

    print(f"\nScene {dataset.scene_name} | GT detections | match dist {args.dist_threshold} m\n")
    print(f"HOTA               {hota['hota']:8.3f}   "
          f"(DetA {hota['det_a']:.3f} · AssA {hota['ass_a']:.3f} · LocA {hota['loc_a']:.3f})")
    print(format_summary(metrics))


if __name__ == "__main__":
    main()
