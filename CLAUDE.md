# 4D Perception — Project Notes for Claude

## Project Overview

3D Multi-Object Tracking (3D MOT) pipeline for autonomous vehicle perception using the KITTI dataset.
Built in Python as a portfolio project demonstrating sensor fusion, state estimation, and 3D visualization.

## What It Does

- Ingests synchronized LiDAR point clouds and camera images from KITTI driving sequences
- Runs 3D object detections (pre-computed or live PV-RCNN) through a SORT-style 3D Kalman filter tracker
- Produces confirmed vehicle tracks with unique IDs, persisted across frames
- Renders an interactive dual-view (camera + LiDAR) visualization saved as a Rerun `.rrd` file
- Exports an MP4 video with stacked camera and LiDAR depth panels for sharing

## Repo Layout

```
perception/                   Our source code — all new code goes here
  boxes.py                    Canonical box format + conversions (kitti_camera_to_lidar,
                               box_corners_3d, register_bbs, BOX_EDGES)
  frame.py                    Frame / Camera / Detections dataclasses — the dataset-agnostic
                               data model every adapter produces and all consumers use
  detections.py               DetectionSource ABC + OpenPCDetSource (live inference, cached)
  assets/car.obj              3D car mesh used by both rerun_vis.py and video.py
  datasets/
    base.py                   SequenceDataset ABC (__len__, __getitem__ → Frame)
    kitti.py                  KittiSequence adapter + KittiLabelSource (pre-computed .txt files)
    kitti_io.py               Low-level KITTI I/O: read_calib, read_velodyne, reduce_to_fov,
                               read_image, read_pose, cam_to_velo, velo_to_cam
  tracker/
    track.py                  Obstacle3D (Kalman filter per track)
    mot.py                    Tracker3D (Hungarian assignment + lifecycle)
  visualization/
    common.py                 shared vis constants/helpers: CAR_OBJ_PATH, MESH_CLASSES,
                               track_color, select_visible
    geometry.py               project_box_to_image (canonical box → 2D corners)
    rerun_vis.py              visualize_tracking + _mask_points_outside_boxes
    video.py                  create_tracking_video + OBJ crease-edge loader
  evaluation.py               read_tracking_labels + evaluate_tracking (CLEAR-MOT via motmetrics)
  cli.py                      shared argparse options (dataset + tracker) for main.py/evaluate.py

tests/                        Unit tests — run with `python -m pytest tests/`
main.py                       Entry point: argparse CLI, tracking loop, visualizer calls
evaluate.py                   Evaluation entry point (argparse; detector/threshold/gate options)
detector.py                   OpenPCDet live detector wrapper (model-agnostic)

multi_object_tracking/        Data only, gitignored (code was ported into perception/ in July 2026)
  detectors/                  Pre-computed detection .txt files: pvrcnn/, casa/, second_iou/, point_rcnn/
  data/                       Raw KITTI data: velodyne/, image_02/, calib/, pose/, label_02/ (GT)

models/
  PointRCNN/pointrcnn_7870.pth
  PVRCNN/pv_rcnn_8369.pth

OpenPCDet/                    Cloned OpenPCDet source (modified — do not upgrade without care)
```

## Running the Pipeline

```bash
python main.py                  # full pipeline → tracking.rrd + showcase.mp4 (pre-computed dets)
python main.py --live           # same but with live PV-RCNN inference
python evaluate.py              # CLEAR-MOT metrics vs KITTI ground truth
python -m pytest tests/         # unit tests (no GPU or dataset needed)
ruff check .                    # lint (configured in pyproject.toml)
```

All options are CLI flags (--detector, --score-threshold, --frames, --no-video, …);
shared dataset/tracker options live in perception/cli.py so defaults exist in one place.

## Coding Guidelines

- Keep the math modules (boxes.py, geometry.py, kitti_io.py) pure functions —
  no state, no I/O mixed with computation. Stateful classes only where state is
  real (tracker, detection sources, dataset adapters).
- Inject dependencies rather than importing them: sources take detector
  instances (duck-typed — keeps torch out of perception/ imports), datasets
  take DetectionSources. This is what makes stub-based testing work.
- Descriptive names; refactor nested conditionals into early-return guards.
- Comments explain *why*, not *what*. NumPy-style docstrings (Parameters /
  Returns) on public functions, matching the existing modules.

## Testing

- Write or update tests alongside any implementation change.
- Unit tests must run without GPU, CUDA, OpenPCDet, or the KITTI dataset:
  synthetic files in tmp_path, stub detectors (see tests/test_datasets.py).
- Test files: tests/test_<module>.py. Run with `python -m pytest tests/`
  (or the "Python: Unit Tests" launch config).

## Tech Stack

| Area | Tools |
|---|---|
| Core language | Python 3.10 |
| Deep learning / detection | PyTorch 2.7.0+cu128, PV-RCNN via OpenPCDet |
| CUDA | 12.8 (RTX 5080 / sm_120 — requires cu128 build of PyTorch) |
| State estimation | FilterPy (Kalman filter) |
| Data association | SciPy `linear_sum_assignment` (Hungarian algorithm) |
| Sensor math | NumPy ≥ 2, SciPy Rotation |
| 3D visualization | Rerun SDK 0.33.1 |
| Image processing | OpenCV |
| Dataset | KITTI (LiDAR, camera, calibration, ego-vehicle pose) — sequence 0008 |

## Detectors

### Pre-computed (Option A)
Four detector outputs ship in `multi_object_tracking/detectors/`:

| Detector | KITTI mAP (moderate) | Notes |
|---|---|---|
| `casa/` | ~86% | Best accuracy; scores use a very negative scale (median ≈ −5). Use `score_threshold ≈ −1.0` |
| `pvrcnn/` | ~84% | Current default; balanced density at `score_threshold = 0.5` |
| `second_iou/` | ~80% | Lowest frame coverage at any threshold |
| `point_rcnn/` | ~76% | Most detections above 0.5 (densest coverage) but weakest accuracy |

Score scales differ per detector — they are raw logits, not probabilities. When switching detectors,
adjust `score_threshold` in `main.py` accordingly.

### Live Inference (Option B)
`detector.py` wraps any OpenPCDet model. Currently configured for PV-RCNN:
- Config: `OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml`
- Weights: `models/PVRCNN/pv_rcnn_8369.pth`
- Live scores are sigmoid probabilities (0–1); `score_threshold = 0.5` works well

**Why PointRCNN was dropped for live inference:** It generates proposals from foreground-point
segmentation — if segmentation is uncertain (sparse/occluded returns), no proposals are generated.
PV-RCNN voxelizes first, giving spatially uniform coverage regardless of local point density.
Also, live inference returns sigmoid probabilities (0–1) while pre-computed files store raw logits,
so the same `score_threshold` value means different things.

## Key Technical Details

### Architecture: datasets and detection sources
- `SequenceDataset` adapters (perception/datasets/) own all dataset-specific I/O and produce
  dataset-agnostic `Frame` objects (perception/frame.py) with canonical-format detections
- `DetectionSource.get(frame_id, raw_points)` supplies per-frame detections; the dataset calls it
  while assembling each Frame. `OpenPCDetSource` (live, cached) works for any dataset that
  provides raw LiDAR points; `KittiLabelSource` parses pre-computed KITTI .txt files
- The live detector consumes the **raw** point cloud (PV-RCNN detects 360°); `Frame.points` is
  the FOV-cropped cloud used for visualization — don't feed the cropped one to a detector
- **Adding Waymo/nuScenes** = one new adapter producing Frames + (optionally) a file-based
  DetectionSource for that dataset's pre-computed detections; tracker and visualizers are untouched

### Evaluation
- `evaluate.py` runs the tracker and scores confirmed tracks against KITTI tracking ground truth
  (`data/label_02/<seq>.txt`) with CLEAR-MOT metrics (motmetrics package)
- Matching: BEV centre distance in the world frame, 2 m gate (nuScenes-style) — GT boxes go
  through the same `kitti_camera_to_lidar` + `register_bbs` path as detections
- GT classes evaluated: Car, Van, Truck (the tracker is class-agnostic; DontCare dropped)
- Current numbers, seq 0008, tuned defaults (min_hits=2, max_missed=3, gate=4.5):
  pvrcnn@0.5 → MOTA 0.553, IDF1 0.731, 1 switch, FP 82, FN 529 / 1369, MT 15/27.
  casa@−1.0 → MOTA 0.533, IDF1 0.724. Pre-tuning (min_hits=3, max_missed=5, gate=6.0)
  scored 0.520 / 0.468. `--gate 3.0` reaches 0.560 but is single-sequence tuning.
  FN-dominated: most lost GT is distant/occluded vehicles the detector misses at these thresholds.

### Tracker (3D SORT)
- 10-dimensional Kalman filter state: `[x, y, z, l, w, h, yaw, vx, vy, vz]`
- 7-dimensional measurement vector: `[x, y, z, l, w, h, yaw]`
- Mahalanobis distance cost matrix using per-track innovation covariance `S = H P Hᵀ + R`
- Hungarian algorithm association with configurable distance threshold
- **Class-gated association**: detections only match tracks from the same class group
  (`DEFAULT_CLASS_GROUPS`: {Car, Van, Truck}, {Pedestrian}, {Cyclist}); vehicles share a group
  because detectors flip labels on the same object. `names=None` disables gating.
- `update(boxes, scores, pose, names)` takes the **full** detection set and filters by
  `score_threshold` internally; returned `det_ids` covers every input detection (0 = unconfirmed)
- Track IDs are assigned by the tracker instance (`_next_id`), not a class-level counter
- `dt` config parameter for the motion model (default 0.1 s = 10 Hz KITTI; nuScenes keyframes are 2 Hz)
- Tunable `velocity_process_noise` parameter to control Q-matrix aggressiveness
- SORT-style lifecycle: `min_hits` to confirm a track, `max_missed` to prune stale tracks
- **Persistent confirmation** via `_confirmed_ids` set: once a track reaches `min_hits`, it stays
  confirmed until evicted by `max_missed` — prevents large vehicles from flickering out when the
  detector misses a frame
- Defaults `min_hits=2, max_missed=3, gate=4.5` tuned by sweep on seq 0008 (see Evaluation);
  larger `max_missed` hurts MOTA here because coasting tracks emit predicted boxes counted as FP

### Coordinate Conventions
- **Canonical box format** (everything inside `perception/`): `[x, y, z_center, l, w, h, yaw]` in
  the LiDAR frame (world frame after `register_bbs`) — defined in `perception/boxes.py`. Matches
  OpenPCDet's native output, so the live detector path needs no conversion at all.
- Sources convert at the boundary: KITTI label files `[h, w, l, x, y, z_bottom, ry]` (camera frame)
  go through `kitti_camera_to_lidar` in the dataset (`yaw = -ry - π/2`, `z_center = z_bottom + h/2`)
- 2D overlays are derived on demand: `project_box_to_image(box, V2C, P2)` builds corners in the
  LiDAR frame (`box_corners_3d`) and projects them through the camera
- Car OBJ native size: 8.95 m (length) × 3.71 m (width) × 2.97 m (height); each axis scaled independently

### Visualization (Rerun SDK)
- Dual-view blueprint: camera image (with 2D box overlays) alongside LiDAR point cloud
- Per-track 3D car mesh scaled per-axis to match detector box dimensions
- Ego vehicle rendered at 0.5× native OBJ scale
- Bounding-box–filtered point cloud: points inside tracked vehicle volumes excluded so meshes are visible
- Asset3D geometry logged once per track; only Transform3D updated each frame

### MP4 Video Export
- Stacked layout: camera panel (top) + LiDAR depth panel (bottom)
- LiDAR projected through `V2C → P2` so detections align pixel-for-pixel with camera panel
- Depth coloring: plasma colormap, far-to-near sort, 0–40 m range
- Car mesh wireframes drawn from OBJ crease edges (dihedral > 50°, keeps ~1236 of 8174 edges)

### OpenPCDet Integration
- Source tree at `OpenPCDet/` added to `sys.path` in `detector.py` (not installed package)
- Modified files in OpenPCDet (do not overwrite):
  - `pcdet/config.py` — `_BASE_CONFIG_` resolution walks up parent dirs to find YAML
  - `pcdet/models/backbones_3d/__init__.py` — spconv imports made optional
  - `pcdet/models/roi_heads/__init__.py` — PartA2 import made optional
  - `pcdet/datasets/__init__.py` — Argo2Dataset import removed (no `av2` module)
  - `pcdet/ops/pointnet2/{pointnet2_batch,pointnet2_stack}/pointnet2_utils.py`,
    `pcdet/ops/pointnet2/pointnet2_stack/voxel_query_utils.py`,
    `pcdet/ops/iou3d_nms/iou3d_nms_utils.py` — legacy `torch.cuda.*Tensor(size)`
    constructors replaced with `torch.empty(..., dtype=..., device="cuda")`
    (deprecated in PyTorch 2.x; silences the DtypeTensor UserWarning)
  - `pcdet/utils/loss_utils.py` — `torch.cuda.amp.custom_fwd` → `torch.amp.custom_fwd(device_type='cuda')`
  - six files under `pcdet/models/` and `pcdet/ops/` — `torch.meshgrid` calls given explicit
    `indexing='ij'` (the legacy default) to silence the FutureWarning
  - `pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py` — `\s` escape removed from docstrings
  - Net effect: PV-RCNN inference emits **zero** torch warnings even with `warnings.simplefilter("always")`
- Built for sm_120 (RTX 5080 Blackwell); requires nvcc 12.8 + cicc 12.8 symlinked into conda env

### Known CUDA Setup (RTX 5080)
- PyTorch: `torch==2.7.0+cu128`
- nvcc 12.8 symlinked from base conda env; cicc 12.8 and libdevice 12.8 also symlinked
- spconv-cu124 works with PyTorch cu128 despite version mismatch

## Bug Fixes Worth Remembering
- `logged_models` must be cleared on track eviction or re-appearing cars become invisible
- `rr.set_time_sequence` deprecated → use `rr.set_time("frame", sequence=i)` (Rerun 0.23+)
- 2D bounding box corners must be clipped to image bounds to prevent Rerun UI rescaling
- Video loop: index into `final_det_ids` with the actual frame number `i`, not enumerate index `idx`
- `hit_streak` resets on any missed frame — large vehicles never accumulate enough consecutive hits
  to stay confirmed; solved with `_confirmed_ids` set that survives missed frames
- **Behind-camera detections span the entire image**: PV-RCNN detects in 360° LiDAR space, so it
  produces valid detections behind the camera (negative Z in camera coords). Dividing by negative Z
  in `project_box_to_image` flips coordinates; when corners straddle Z=0 the results are
  arbitrary and clipping them to image bounds makes the box appear to fill the frame.
  Fix: `geometry.py` returns `None` when any corner has Z ≤ 0; callers in `video.py` and
  `rerun_vis.py` guard with `if corners_2d is not None: continue/skip`.
