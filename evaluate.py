"""
Evaluate the 3D MOT tracker against KITTI tracking ground truth.

Runs the tracker over a sequence, matches confirmed tracks to ground-truth
vehicle trajectories on BEV centre distance, and reports CLEAR-MOT metrics.

    python evaluate.py                       # pvrcnn pre-computed, seq 0008
    python evaluate.py --detector casa --score-threshold -1.0
    python evaluate.py --dist-threshold 3.0
"""

import argparse
import os

import numpy as np

from perception.boxes import kitti_camera_to_lidar, register_bbs
from perception.datasets.kitti import KittiLabelSource, KittiSequence
from perception.evaluation import evaluate_tracking, format_summary, read_tracking_labels
from perception.tracker.mot import Tracker3D


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",  default="multi_object_tracking/data")
    p.add_argument("--label-root", default="multi_object_tracking/detectors")
    p.add_argument("--detector",   default="pvrcnn",
                   help="pre-computed detector directory name (pvrcnn, casa, second_iou, point_rcnn)")
    p.add_argument("--seq", type=int, default=8)
    p.add_argument("--gt", default=None,
                   help="ground-truth label file (default: <data-root>/label_02/<seq>.txt)")
    p.add_argument("--score-threshold",  type=float, default=0.5)
    p.add_argument("--dist-threshold",   type=float, default=2.0,
                   help="max BEV centre distance in metres for a GT match")
    p.add_argument("--min-hits",   type=int,   default=3)
    p.add_argument("--max-missed", type=int,   default=5)
    p.add_argument("--gate",       type=float, default=6.0,
                   help="tracker Mahalanobis association gate")
    p.add_argument("--velocity-process-noise", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    seq_name = str(args.seq).zfill(4)
    gt_path = args.gt or os.path.join(args.data_root, "label_02", seq_name + ".txt")

    dataset = KittiSequence(
        args.data_root, seq_id=args.seq,
        detections=KittiLabelSource(
            os.path.join(args.label_root, args.detector), args.seq, args.data_root
        ),
    )
    tracker = Tracker3D(config={
        "score_threshold":        args.score_threshold,
        "min_hits":               args.min_hits,
        "max_missed":             args.max_missed,
        "dist_threshold":         args.gate,
        "velocity_process_noise": args.velocity_process_noise,
    })
    gt = read_tracking_labels(gt_path)

    def frames():
        for i in range(len(dataset)):
            frame = dataset[i]
            scores = np.asarray(frame.detections.scores, dtype=float)
            mask   = scores > tracker.score_threshold
            pred_ids, bbs, _, _ = tracker.update(
                frame.detections.boxes[mask, :7], scores[mask], pose=frame.ego_pose
            )
            pred_xy = np.array(bbs)[:, :2] if bbs else np.zeros((0, 2))

            # ground truth → canonical → same world frame as the tracker output
            gt_kitti, gt_ids = gt.get(i, (np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=int)))
            gt_boxes = kitti_camera_to_lidar(gt_kitti, frame.camera.lidar_to_cam)
            gt_boxes = register_bbs(gt_boxes.astype(np.float64), frame.ego_pose)
            yield gt_ids, gt_boxes[:, :2], pred_ids, pred_xy

    metrics = evaluate_tracking(frames(), dist_threshold=args.dist_threshold)

    print(f"\nSequence {seq_name} | detector: {args.detector} | "
          f"score>{args.score_threshold} | match dist {args.dist_threshold} m\n")
    print(format_summary(metrics))


if __name__ == "__main__":
    main()
