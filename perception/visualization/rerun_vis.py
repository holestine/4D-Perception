from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation as R

from perception.boxes import BOX_EDGES
from perception.visualization.geometry import project_box_to_image

_CAR_OBJ        = str(Path(__file__).resolve().parents[1] / "assets" / "car.obj")
_CAR_SCALE      = [0.5, 0.5, 0.5]   # ego vehicle: 0.5× native OBJ size
_OBJ_NATIVE     = np.array([8.95, 3.71, 2.97], dtype=np.float32)
_VEHICLE_CLASSES = {"Car", "Van", "Truck"}


def _mask_points_outside_boxes(positions, boxes, margin=0.3):
    """Boolean mask selecting points that fall outside all bounding boxes.

    Parameters
    ----------
    positions : ndarray (N, 3)
    boxes     : list of (center, l, w, h, yaw)
    margin    : float  extra clearance in metres per half-extent

    Returns
    -------
    ndarray bool (N,)  True = point is outside every box
    """
    keep = np.ones(len(positions), dtype=bool)
    for center, l, w, h, yaw in boxes:
        pts     = positions - center
        c, s    = np.cos(yaw), np.sin(yaw)
        local_x =  pts[:, 0] * c + pts[:, 1] * s
        local_y = -pts[:, 0] * s + pts[:, 1] * c
        local_z =  pts[:, 2]
        inside  = (
            (np.abs(local_x) <= l / 2 + margin) &
            (np.abs(local_y) <= w / 2 + margin) &
            (np.abs(local_z) <= h / 2 + margin)
        )
        keep &= ~inside
    return keep


def visualize_tracking(dataset, frames, final_det_ids, show_unconfirmed_above=4.0, out_file=None):
    """Render tracked detections with the Rerun SDK.

    Two side-by-side views:
      - Camera: raw image with 3D bounding boxes projected onto it.
      - LiDAR:  coloured point cloud with a 3D car mesh per confirmed track.

    Parameters
    ----------
    dataset        : SequenceDataset
    frames         : iterable of int  frame indices to visualize
    final_det_ids  : list[ndarray]    per-frame confirmed track IDs from the tracking loop
    show_unconfirmed_above : float
        Also draw unconfirmed detections scoring above this (default 4 —
        raw-logit scale, so sigmoid-scored detections are never drawn).
    out_file       : str or None      save to .rrd file instead of displaying inline
    """
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/ego_vehicle/camera", name="Camera"),
            rrb.Spatial3DView(origin="world/ego_vehicle/lidar",  name="LiDAR"),
            column_shares=[1, 1],
        )
    )
    rr.init("KITTI Visualizer Tracking", spawn=False, default_blueprint=blueprint)
    rr.log("world/ego_vehicle/camera/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
    rr.log("world/ego_vehicle/lidar/",  rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    frames = list(frames)
    cmap   = plt.get_cmap("tab20")

    logged_models: set[int] = set()
    prev_active:   set[int] = set()
    ego_car_logged           = False

    for i in frames:
        rr.set_time("frame", sequence=i)

        rr.log("world/ego_vehicle/camera/image/detections", rr.Clear(recursive=True))
        rr.log("world/ego_vehicle/lidar/points",            rr.Clear(recursive=False))

        frame = dataset[i]
        pose       = frame.ego_pose
        P2         = frame.camera.projection
        V2C        = frame.camera.lidar_to_cam
        points     = frame.points
        image      = frame.image
        boxes      = frame.detections.boxes
        det_scores = frame.detections.scores

        # ── Ego vehicle ───────────────────────────────────────────────────────
        if pose is not None:
            translation = pose[:3, 3]
            quat_xyzw   = R.from_matrix(pose[:3, :3]).as_quat()
            rr.log("world/ego_vehicle", rr.Transform3D(
                translation=translation,
                quaternion=rr.Quaternion(xyzw=quat_xyzw),
                relation=rr.TransformRelation.ParentFromChild,
            ))
            if not ego_car_logged:
                rr.log("world/ego_vehicle/lidar/car_model",
                       rr.Transform3D(scale=_CAR_SCALE))
                rr.log("world/ego_vehicle/lidar/car_model/mesh",
                       rr.Asset3D(path=_CAR_OBJ))
                ego_car_logged = True

        # ── Camera image ──────────────────────────────────────────────────────
        rr.log("world/ego_vehicle/camera/image",
               rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        # ── Point cloud (deferred until box masking is ready) ─────────────────
        positions = points[:, :3]
        distances = np.linalg.norm(positions, axis=1)
        norm      = (255.0 * (distances - distances.min()) /
                     (np.ptp(distances) + 1e-5)).astype(np.uint8)
        colors    = (plt.cm.cividis(norm / 255.0)[:, :3] * 255).astype(np.uint8)

        # ── Per-detection visualisation ───────────────────────────────────────
        confirmed_mask = final_det_ids[i] > 0
        score_mask     = det_scores > show_unconfirmed_above
        vis_mask       = confirmed_mask | score_mask
        boxes_v        = np.array(boxes)[vis_mask]
        det_ids        = final_det_ids[i][vis_mask]
        names_v        = np.array(frame.detections.names)[vis_mask]

        detected_boxes: list[tuple] = []
        curr_active:    set[int]    = set()

        for box, track_id, name in zip(boxes_v, det_ids, names_v):
            if track_id == 0:
                continue

            color = (np.array(cmap(track_id % cmap.N)) * 255).astype(np.uint8)

            corners_2d = project_box_to_image(box, V2C, P2, image.shape)
            if corners_2d is not None:
                lines = [np.array([corners_2d[s], corners_2d[e]]) for s, e in BOX_EDGES]
                rr.log(
                    f"world/ego_vehicle/camera/image/detections/box_{track_id}",
                    rr.LineStrips2D(lines, labels=[f"{track_id}"], colors=np.array(color)),
                )

            x, y, z, l, w, h, yaw = box.tolist()
            if l * w * h == 0:
                continue

            center = np.array([x, y, z])
            quat   = R.from_euler("z", yaw, degrees=False).as_quat()
            detected_boxes.append((center, l, w, h, yaw))

            entity = f"world/ego_vehicle/lidar/models/track_{track_id}"

            if name in _VEHICLE_CLASSES:
                rr.log(entity, rr.Transform3D(
                    translation=center,
                    quaternion=rr.Quaternion(xyzw=quat),
                    scale=[l / _OBJ_NATIVE[0], w / _OBJ_NATIVE[1], h / _OBJ_NATIVE[2]],
                    relation=rr.TransformRelation.ParentFromChild,
                ))
                if track_id not in logged_models:
                    rr.log(f"{entity}/model", rr.Asset3D(
                        path=_CAR_OBJ, albedo_factor=color[:3] / 255.0,
                    ))
                    logged_models.add(track_id)
            else:
                # Cyclist / Pedestrian — draw a box primitive instead of a car mesh
                rr.log(entity, rr.Boxes3D(
                    half_sizes=[[l / 2, w / 2, h / 2]],
                    centers=[center],
                    rotations=[rr.Quaternion(xyzw=quat)],
                    colors=[color[:3]],
                    labels=[f"{name} {track_id}"],
                ))

            curr_active.add(track_id)

        for dead_id in prev_active - curr_active:
            rr.log(f"world/ego_vehicle/lidar/models/track_{dead_id}", rr.Clear(recursive=True))
            logged_models.discard(dead_id)
        prev_active = curr_active

        keep = _mask_points_outside_boxes(positions, detected_boxes)
        rr.log("world/ego_vehicle/lidar/points",
               rr.Points3D(positions=positions[keep], colors=colors[keep]))

    if out_file is not None:
        rr.save(out_file, default_blueprint=blueprint)
    else:
        rr.notebook_show(height=500, width=1000)
