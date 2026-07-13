"""Shared command-line options for the pipeline entry points.

main.py and evaluate.py accept the same dataset and tracker options; this
module owns them so defaults live in exactly one place.
"""

import os

from perception.datasets.kitti import KittiLabelSource
from perception.tracker.mot import Tracker3D


def add_dataset_args(parser):
    parser.add_argument("--data-root",  default="multi_object_tracking/data",
                        help="KITTI data directory")
    parser.add_argument("--label-root", default="multi_object_tracking/detectors",
                        help="pre-computed detections directory")
    parser.add_argument("--detector",   default="pvrcnn",
                        help="pre-computed detector name (pvrcnn, casa, second_iou, point_rcnn)")
    parser.add_argument("--seq", type=int, default=8, help="KITTI sequence number")


def add_tracker_args(parser):
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="detections at or below this score are ignored "
                             "(scale differs per detector — see README)")
    parser.add_argument("--min-hits",   type=int,   default=2,
                        help="consecutive detections to confirm a track")
    parser.add_argument("--max-missed", type=int,   default=3,
                        help="missed frames before a track is pruned")
    parser.add_argument("--gate",       type=float, default=4.5,
                        help="Mahalanobis association gate")
    parser.add_argument("--velocity-process-noise", type=float, default=1.0,
                        help="Q-scale for the velocity states")


def build_tracker(args):
    return Tracker3D(config={
        "score_threshold":        args.score_threshold,
        "min_hits":               args.min_hits,
        "max_missed":             args.max_missed,
        "dist_threshold":         args.gate,
        "velocity_process_noise": args.velocity_process_noise,
    })


def build_label_source(args):
    return KittiLabelSource(
        os.path.join(args.label_root, args.detector), args.seq, args.data_root
    )
