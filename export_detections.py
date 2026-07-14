"""
Regenerate pre-computed detection files by running live OpenPCDet inference.

Writes one KITTI-format .txt per frame — the layout KittiLabelSource reads —
so the output is a drop-in replacement for multi_object_tracking/detectors/.

    python export_detections.py                          # PV-RCNN, seq 0008
    python export_detections.py --name pvrcnn_live --frames 50

Note: live inference scores are sigmoid probabilities (0-1), while the
detector files shipped in this repo store raw logits (scale varies per
model). Regenerated files are therefore equivalent in format and coverage
but not bit-identical, and want a different --score-threshold downstream.

Requires a CUDA GPU, the OpenPCDet source tree, and model weights (see
README Setup).
"""

import argparse
import os

import numpy as np

from perception.boxes import lidar_to_kitti_camera
from perception.datasets.kitti import KittiSequence
from perception.detections import OpenPCDetSource
from perception.visualization.geometry import project_box_to_image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root",  default="multi_object_tracking/data")
    p.add_argument("--out-root",   default="multi_object_tracking/detectors")
    p.add_argument("--name",       default="pvrcnn_live",
                   help="detector name — output goes to <out-root>/<name>/<seq>/")
    p.add_argument("--seq", type=int, default=8, help="KITTI sequence number")
    p.add_argument("--frames", type=int, default=None,
                   help="only export the first N frames (default: all)")
    p.add_argument("--cfg-file",   default="OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml")
    p.add_argument("--checkpoint", default="models/PVRCNN/pv_rcnn_8369.pth")
    p.add_argument("--score-threshold", type=float, default=0.1,
                   help="minimum sigmoid score to export (downstream consumers "
                        "apply their own threshold on top)")
    return p.parse_args()


def detection_lines(frame):
    """Format one frame's detections as KITTI label lines.

    Line format (KittiLabelSource._read_detection_label):
        class trunc occ alpha x1 y1 x2 y2 h w l x y z ry score

    The 2D bbox comes from projecting the 3D corners; detections behind the
    camera (PV-RCNN detects 360°) get a -1 -1 -1 -1 placeholder, which the
    parser ignores.
    """
    V2C = frame.camera.lidar_to_cam
    P2  = frame.camera.projection
    kitti_boxes = lidar_to_kitti_camera(frame.detections.boxes, V2C)

    lines = []
    for box, kitti, score, name in zip(frame.detections.boxes, kitti_boxes,
                                       frame.detections.scores,
                                       frame.detections.names):
        h, w, l, x, y, z, ry = kitti.tolist()

        corners_2d = project_box_to_image(box, V2C, P2, frame.image.shape)
        if corners_2d is None:
            bbox = (-1.0, -1.0, -1.0, -1.0)
        else:
            bbox = (corners_2d[:, 0].min(), corners_2d[:, 1].min(),
                    corners_2d[:, 0].max(), corners_2d[:, 1].max())

        # Observation angle: heading relative to the ray from the camera.
        alpha = ry - np.arctan2(x, z)

        lines.append(
            f"{name} -1 -1 {alpha:.4f} "
            f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} "
            f"{h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {ry:.4f} "
            f"{score:.4f}"
        )
    return lines


def main():
    args = parse_args()

    from detector import OpenPCDetDetector
    detections = OpenPCDetSource(OpenPCDetDetector(
        cfg_file=args.cfg_file, checkpoint=args.checkpoint,
        data_root=args.data_root, score_threshold=args.score_threshold,
    ))
    dataset = KittiSequence(args.data_root, seq_id=args.seq, detections=detections)

    seq_name = str(args.seq).zfill(4)
    out_dir  = os.path.join(args.out_root, args.name, seq_name)
    os.makedirs(out_dir, exist_ok=True)

    n_frames = len(dataset) if args.frames is None else min(args.frames, len(dataset))
    print(f"Exporting {n_frames} frames to '{out_dir}' …")
    for i in range(n_frames):
        frame = dataset[i]
        out_file = os.path.join(out_dir, str(i).zfill(6) + ".txt")
        with open(out_file, "w") as f:
            f.write("\n".join(detection_lines(frame)) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_frames} frames exported")

    print(f"Done → '{out_dir}'")


if __name__ == "__main__":
    main()
