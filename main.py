"""
3D Multi-Object Tracking pipeline for KITTI sequences.

Pipeline:
  1. Load a KITTI sequence (LiDAR, camera, calibration, pose) with either
     pre-computed detections (default) or live OpenPCDet inference (--live).
  2. Run a SORT-style 3D Kalman filter tracker across every frame.
  3. Visualize confirmed tracks with Rerun and export MP4 videos.

    python main.py                          # pre-computed pvrcnn detections
    python main.py --live                   # live PV-RCNN inference (GPU)
    python main.py --detector casa --score-threshold -1.0
    python main.py --frames 50 --no-video   # quick look at the first 50 frames
"""

import argparse
import time

import numpy as np

from perception.cli import add_dataset_args, add_tracker_args, build_label_source, build_tracker
from perception.datasets.kitti import KittiSequence
from perception.visualization.rerun_vis import visualize_tracking
from perception.visualization.video import create_tracking_video


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(p)
    add_tracker_args(p)

    live = p.add_argument_group("live inference (--live)")
    live.add_argument("--live", action="store_true",
                      help="run PV-RCNN live instead of loading pre-computed detections")
    live.add_argument("--cfg-file",   default="OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml")
    live.add_argument("--checkpoint", default="models/PVRCNN/pv_rcnn_8369.pth")

    vis = p.add_argument_group("visualization")
    vis.add_argument("--frames", type=int, default=None,
                     help="only process the first N frames (default: all)")
    vis.add_argument("--show-unconfirmed-above", type=float, default=4.0,
                     help="also draw unconfirmed detections scoring above this "
                          "(raw-logit scale; sigmoid scores never exceed it)")
    vis.add_argument("--no-rrd",   action="store_true", help="skip the Rerun .rrd export")
    vis.add_argument("--no-video", action="store_true", help="skip the MP4 exports")
    vis.add_argument("--showcase-frames", type=int, nargs=2, default=(205, 265),
                     metavar=("START", "END"), help="frame range for showcase.mp4")
    vis.add_argument("--showcase-fps", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    if args.live:
        from detector import OpenPCDetDetector
        from perception.detections import OpenPCDetSource
        detections = OpenPCDetSource(OpenPCDetDetector(
            cfg_file=args.cfg_file, checkpoint=args.checkpoint, data_root=args.data_root,
        ))
    else:
        detections = build_label_source(args)

    dataset = KittiSequence(args.data_root, seq_id=args.seq, detections=detections)
    tracker = build_tracker(args)

    n_frames      = len(dataset) if args.frames is None else min(args.frames, len(dataset))
    frame_indices = range(n_frames)

    # ── Tracking loop ──────────────────────────────────────────────────────────
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

    print(f"Tracked {n_frames} frames in {elapsed:.2f}s  ({n_frames / elapsed:.1f} fps)")

    # ── Visualization ──────────────────────────────────────────────────────────
    if not args.no_rrd:
        visualize_tracking(
            dataset,
            frame_indices,
            final_det_ids,
            show_unconfirmed_above=args.show_unconfirmed_above,
            out_file="tracking.rrd",
        )

    if not args.no_video:
        create_tracking_video(
            dataset,
            frame_indices,
            final_det_ids,
            show_unconfirmed_above=args.show_unconfirmed_above,
            out_file="tracking.mp4",
        )

        start, end = args.showcase_frames
        if start < n_frames:
            create_tracking_video(
                dataset,
                range(start, min(end, n_frames)),
                final_det_ids,
                show_unconfirmed_above=args.show_unconfirmed_above,
                fps=args.showcase_fps,
                out_file="showcase.mp4",
            )


if __name__ == "__main__":
    main()
