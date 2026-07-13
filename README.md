# 4D Perception — 3D Multi-Object Tracking

A full-stack autonomous vehicle perception pipeline: LiDAR + camera sensor fusion, 3D Kalman filter tracking, and interactive dual-modality visualization — built from first principles on the KITTI tracking benchmark.

---

## Pipeline Overview

```
KITTI Sequence
      │
      ├─ LiDAR point cloud (.bin)  ──►  3D Object Detector  ──►  Detection boxes [h,w,l,x,y,z,ry]
      │                                  (PV-RCNN / pre-computed)          │
      └─ Camera image (.png)                                               │
      └─ Calibration + Ego pose                                            ▼
                                                              3D SORT Tracker
                                                         (Kalman filter + Hungarian)
                                                                           │
                                          ┌────────────────────────────────┤
                                          │                                │
                                          ▼                                ▼
                                 Rerun Visualization               MP4 Video Export
                              (camera + LiDAR 3D view)         (camera + LiDAR depth)
```

**Key numbers on KITTI sequence 0008 (390 frames):**
- Tracker runs at **~1,500 fps** (Kalman + Hungarian step only)
- PV-RCNN live inference: **~8 fps** on RTX 5080
- Pre-computed detections: full sequence processes in **< 1 second**

---

## Demo

The pipeline produces two outputs:

**`showcase.mp4`** — Stacked dual-panel video: camera RGB with 3D bounding box overlays (top) and LiDAR returns projected through the identical camera matrix, coloured by depth (bottom).

[![4D Perception — 3D Multi-Object Tracking on KITTI](https://img.youtube.com/vi/6A5poCpgLGk/maxresdefault.jpg)](https://www.youtube.com/watch?v=6A5poCpgLGk)

**`tracking.rrd`** — Interactive Rerun viewer: scrub through frames, orbit the 3D scene, inspect individual tracks.

Both outputs colour tracks by ID (tab20 colourmap) and keep IDs stable across frame gaps.

---

## Technical Depth

### 3D SORT Tracker

A 3D extension of the [SORT](https://arxiv.org/abs/1602.00763) algorithm operating in LiDAR space.

**Kalman filter state (10-D):**
```
[x, y, z, l, w, h, yaw, vx, vy, vz]
```
Constant-velocity motion model; box dimensions and yaw modelled as constant between frames.

**Association:**
- Cost matrix uses **Mahalanobis distance** (not 3D IoU) — naturally gates uncertain tracks more loosely and confident tracks more tightly via the per-track innovation covariance `S = H P Hᵀ + R`
- **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) solves the assignment problem optimally in O(n³)
- Yaw differences wrapped to `[−π, π]` before computing distance to handle the 180° ambiguity

**Track lifecycle:**
- `min_hits = 3` consecutive detections to confirm a track
- `max_missed = 5` consecutive misses before pruning
- **Persistent confirmation** (`_confirmed_ids` set): once confirmed, a track stays visible through missed frames — critical for large vehicles that intermittently fall below detection threshold

### Sensor Fusion & Coordinate Handling

```
Camera frame:  x = right,   y = down,  z = forward  (RIGHT_HAND_Y_DOWN)
LiDAR frame:   x = forward, y = left,  z = up        (RIGHT_HAND_Z_UP)
World frame:   ego-vehicle pose applied to LiDAR frame
```

- KITTI calibration chain: `Velodyne → R0_rect → P2` applied consistently in all views
- Box centre lifted from KITTI bottom-face convention: `z_centre = z_bottom + h/2`
- Yaw convention: `yaw_lidar = −ry − π/2`
- Ego-vehicle pose integrated every frame so box centres are tracked in a consistent world frame, eliminating drift from vehicle motion

### Visualization

**Rerun:** Per-track 3D car mesh (`.obj`) scaled per-axis to match detector output — `car.obj` native size 8.95 × 3.71 × 2.97 m mapped to each detected box. Points inside confirmed track volumes masked out so meshes are not buried by their own returns.

**MP4:** LiDAR depth panel computed by projecting every point through `V2C → P2` (the same matrices used for the camera), so detections align pixel-for-pixel across both panels. OBJ wireframes rendered using crease-edge filtering (dihedral > 50°, retaining ~1,236 of 8,174 edges) for clean car outlines without triangle noise.

---

## Detector Comparison

Four pre-computed detector outputs are included for sequence 0008. **Scores are raw logits** — scale varies per model; adjust `score_threshold` when switching.

| Detector | KITTI Car AP (moderate) | Avg dets/frame at threshold | Empty frames | Notes |
|---|---|---|---|---|
| CasA (`casa/`) | **~86%** | 3.1 @ threshold −1.0 | 33 | Best accuracy; requires lower threshold due to score scale |
| PV-RCNN (`pvrcnn/`) | ~84% | 2.5 @ threshold 0.5 | 46 | **Current default** — strong balance of precision and recall |
| SECOND-IoU (`second_iou/`) | ~80% | 1.8 @ threshold 0.5 | 96 | Fewest detections; worst frame coverage |
| PointRCNN (`point_rcnn/`) | ~76% | 3.4 @ threshold 0.5 | 21 | Densest pre-computed coverage but lowest accuracy |

**Why PointRCNN is not used for live inference:** PointRCNN generates proposals from foreground-point segmentation (PointNet++). When segmentation is uncertain — sparse returns from distant vehicles, partial occlusions — no foreground points means no proposals. PV-RCNN voxelizes first, giving spatially uniform feature coverage regardless of local point density.

---

## Setup

### Requirements

- Python 3.10
- CUDA 12.8+ (tested on RTX 5080 / Blackwell sm_120; any CUDA 11.8+ GPU should work with adjusted PyTorch build)
- Conda

### 1. Create the environment

```bash
conda create -n 4D python=3.10 -y
conda activate 4D
```

### 2. Install PyTorch

**RTX 5080 / Ada / Hopper (CUDA 12.8):**
```bash
pip install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**RTX 3090 / 4090 / A100 (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install OpenPCDet

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
pip install -e .
cd ..
```

> **Note for RTX 5080 users:** sm_120 requires PyTorch cu128 and nvcc 12.8. See [OpenPCDet docs](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) for build details.

### 5. Download model weights (for live inference)

PV-RCNN checkpoint from the [OpenPCDet model zoo](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/MODEL_ZOO.md):

```bash
mkdir -p models/PVRCNN
# Download pv_rcnn_8369.pth to models/PVRCNN/
```

### 6. Download KITTI Tracking data

From the [KITTI tracking benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php), download:
- Left color images
- Velodyne point clouds
- Camera calibration files
- Ego vehicle poses
- Training labels (ground truth, for evaluation)

Place under `multi_object_tracking/data/` with the structure:
```
multi_object_tracking/data/
  velodyne/0008/000000.bin ...
  image_02/0008/000000.png ...
  calib/0008.txt
  pose/0008/pose.txt
  label_02/0008.txt
```

### 7. Run

```bash
python main.py
```

Produces `tracking.rrd` (open with `rerun tracking.rrd`) and `showcase.mp4`.

---

## Evaluation

Measure tracking quality against KITTI ground truth (CLEAR-MOT metrics via
[motmetrics](https://github.com/cheind/py-motmetrics)):

```bash
python evaluate.py                                    # pvrcnn, seq 0008
python evaluate.py --detector casa --score-threshold -1.0
```

Confirmed tracks are matched to ground-truth vehicles (Car/Van/Truck) on
bird's-eye-view centre distance (2 m gate, nuScenes-style) in the world frame.

Results on sequence 0008 (pre-computed detections, tuned tracker defaults —
`min_hits=2, max_missed=3, gate=4.5`, selected by parameter sweep with this harness):

| Detector | MOTA | MOTP | IDF1 | ID sw. | FP | FN | MT |
|---|---|---|---|---|---|---|---|
| `pvrcnn` @ 0.5 | **0.553** | 0.284 m | **0.731** | 1 | 82 | 529 | 15/27 |
| `casa` @ −1.0 | 0.533 | 0.251 m | 0.724 | 4 | 141 | 495 | 16/27 |

(Pre-tuning defaults `min_hits=3, max_missed=5, gate=6.0` scored MOTA 0.520 / 0.468.
A tighter `--gate 3.0` reaches MOTA 0.560 on this sequence but was left off the
defaults — tuned on a single sequence, the χ²-principled 4.5 gate is safer.)

---

## Configuration

Edit the top of `main.py` to switch between pre-computed and live detection:

```python
# Option A — pre-computed (fast, no GPU required):
#   label root: multi_object_tracking/detectors/pvrcnn (or casa, second_iou, point_rcnn)
detections = KittiLabelSource("multi_object_tracking/detectors/pvrcnn", SEQ_ID, DATA_ROOT)

# Option B — live PV-RCNN inference:
from detector import OpenPCDetDetector
detections = OpenPCDetSource(OpenPCDetDetector(
    cfg_file   = "OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml",
    checkpoint = "models/PVRCNN/pv_rcnn_8369.pth",
    data_root  = DATA_ROOT,
))

dataset = KittiSequence(DATA_ROOT, seq_id=SEQ_ID, detections=detections)
```

Tracker hyperparameters in `main.py`:

| Parameter | Default | Effect |
|---|---|---|
| `score_threshold` | 0.5 | Detections below this are ignored (adjust per detector — see table above) |
| `min_hits` | 2 | Consecutive detections to confirm a track |
| `max_missed` | 3 | Missed frames before a track is pruned |
| `dist_threshold` | 4.5 | Mahalanobis gate — increase for faster-moving or noisier scenes |
| `velocity_process_noise` | 1.0 | Higher = tracker adapts faster to acceleration |
| `dt` | 0.1 | Seconds between frames (10 Hz KITTI; set 0.5 for nuScenes keyframes) |
| `class_groups` | vehicles / peds / cyclists | Classes allowed to associate with each other |

---

## Project Structure

```
perception/                   Core library
  boxes.py                    Canonical box format [x,y,z,l,w,h,yaw] + conversions
  frame.py                    Frame / Camera / Detections — dataset-agnostic data model
  detections.py               DetectionSource interface + live OpenPCDetSource
  assets/car.obj              3D car mesh for visualization
  datasets/
    base.py                   SequenceDataset interface
    kitti.py                  KittiSequence adapter + KittiLabelSource (pre-computed files)
    kitti_io.py               Low-level KITTI I/O (calibration, LiDAR, images, poses)
  tracker/
    track.py                  Obstacle3D — per-track Kalman filter
    mot.py                    Tracker3D — Hungarian assignment + track lifecycle
  visualization/
    geometry.py               project_box_to_image
    rerun_vis.py              Rerun SDK visualization
    video.py                  MP4 export with dual camera/LiDAR panels
  evaluation.py               CLEAR-MOT metrics against KITTI tracking ground truth

tests/                        Unit tests (pytest)
main.py                       Entry point
evaluate.py                   Tracking evaluation entry point (CLEAR-MOT)
detector.py                   OpenPCDet live inference wrapper (model-agnostic)
requirements.txt

multi_object_tracking/        Data only (not tracked in git)
  detectors/                  Pre-computed detections: pvrcnn/, casa/, second_iou/, point_rcnn/
  data/                       Raw KITTI sequences

models/
  PointRCNN/pointrcnn_7870.pth
  PVRCNN/pv_rcnn_8369.pth

OpenPCDet/                    Detection backbone (cloned, locally modified for sm_120)
```

---

## Roadmap

### Datasets

| Dataset | Status | Notes |
|---|---|---|
| KITTI Tracking | ✅ Done | 21 sequences, 64-beam Velodyne HDL-64E |
| Waymo Open Dataset | Planned | 1,150 segments, 5-beam top LiDAR + 4 side LiDAR; different coordinate system; richer ego-motion |
| nuScenes | Planned | 1,000 scenes, 6-camera surround, 32-beam LiDAR, multi-sweep accumulation |

### Detectors

| Model | Status | Notes |
|---|---|---|
| PV-RCNN | ✅ Live inference | Current live detector |
| CasA | ✅ Pre-computed | Best KITTI accuracy (~86% mAP) |
| CenterPoint | Planned | Anchor-free, heatmap-based; dominant on Waymo and nuScenes leaderboards |
| BEVFusion | Planned | Camera + LiDAR fusion in BEV space; addresses LiDAR sparsity at range |
| DSVT | Planned | Dynamic sparse voxel transformer; strong across all three benchmarks |

### Tracker

| Feature | Status | Notes |
|---|---|---|
| 3D SORT (CV Kalman + Hungarian) | ✅ Done | |
| CTRA motion model | Planned | Constant turn-rate and acceleration — better for turning vehicles |
| Appearance features | Planned | Re-ID embedding to recover tracks after long occlusion |
| Multi-class tracking | Planned | Separate lifecycle params per class (pedestrian vs. vehicle) |
| HOTA / MOTA / MOTP evaluation | Planned | Quantitative benchmark against KITTI tracking ground truth |

### Infrastructure

| Feature | Status | Notes |
|---|---|---|
| Per-sequence config files | Planned | YAML-driven scene configuration rather than hardcoded constants |
| nuScenes devkit integration | Planned | Standardized evaluation via the official nuScenes tracking API |
| ROS 2 node | Planned | Wrap the tracker as a ROS 2 node for real-time sensor input |

---

## Tech Stack

| Area | Tools |
|---|---|
| Core language | Python 3.10 |
| Deep learning | PyTorch 2.7, OpenPCDet |
| Detection models | PV-RCNN, CasA, SECOND-IoU, PointRCNN |
| State estimation | FilterPy — Kalman filter |
| Data association | SciPy — Hungarian algorithm |
| Sensor math | NumPy ≥ 2, SciPy Rotation |
| 3D visualization | Rerun SDK 0.33.1 |
| Video export | OpenCV |
| Dataset | KITTI Tracking Benchmark |

---

## References

- [SORT: Simple, Online and Realtime Tracking](https://arxiv.org/abs/1602.00763) — Bewley et al., 2016
- [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](https://arxiv.org/abs/1912.13192) — Shi et al., CVPR 2020
- [CasA: A Cascade Attention Network for 3D Object Detection](https://arxiv.org/abs/2208.09723) — Wu et al., 2022
- [CenterPoint: Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) — Yin et al., CVPR 2021
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) — Open-source toolbox for 3D object detection
- [KITTI Tracking Benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) — Geiger et al., CVPR 2012
