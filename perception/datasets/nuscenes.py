"""nuScenes adapter: SequenceDataset over one scene of a nuScenes split.

Reads the nuScenes JSON tables directly (no devkit dependency — the devkit
pins numpy<2, which conflicts with rerun-sdk). Only keyframes (2 Hz samples)
are used, matching the annotation rate.

Frames follow the same conventions as the KITTI adapter: canonical boxes in
the LiDAR frame, `ego_pose` transforming LiDAR-frame points to a fixed world
frame, raw 360° point cloud in `Frame.points`.
"""

import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from perception.datasets.base import SequenceDataset
from perception.detections import DetectionSource
from perception.frame import Camera, Detections, Frame

# nuScenes category → the class names used for association gating and meshes
CATEGORY_MAP = {
    "vehicle.car":                "Car",
    "vehicle.emergency.police":   "Car",
    "vehicle.truck":              "Truck",
    "vehicle.bus.bendy":          "Truck",
    "vehicle.bus.rigid":          "Truck",
    "vehicle.construction":       "Truck",
    "vehicle.trailer":            "Truck",
    "vehicle.emergency.ambulance": "Van",
    "human.pedestrian.adult":         "Pedestrian",
    "human.pedestrian.child":         "Pedestrian",
    "human.pedestrian.police_officer": "Pedestrian",
    "human.pedestrian.construction_worker": "Pedestrian",
    "vehicle.bicycle":            "Cyclist",
    "vehicle.motorcycle":         "Cyclist",
}


def transform_matrix(translation, rotation_wxyz):
    """4×4 rigid transform from a nuScenes translation + [w, x, y, z] quaternion."""
    w, x, y, z = rotation_wxyz
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3]  = translation
    return T


def yaw_of(matrix):
    """Rotation about +z of a 4×4 (or 3×3) transform."""
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def global_box_to_lidar(center, size_wlh, rotation_wxyz, T_lidar_from_global):
    """Convert one nuScenes global-frame annotation to a canonical box.

    Parameters
    ----------
    center              : (3,) box centre in the global frame
    size_wlh            : (3,) nuScenes size convention [w, l, h]
    rotation_wxyz       : (4,) box orientation quaternion, global frame
    T_lidar_from_global : ndarray (4, 4)

    Returns
    -------
    ndarray (7,) canonical [x, y, z_center, l, w, h, yaw] in the LiDAR frame
    """
    center_lidar = (T_lidar_from_global @ [*center, 1.0])[:3]
    yaw = yaw_of(T_lidar_from_global[:3, :3] @ transform_matrix([0, 0, 0], rotation_wxyz)[:3, :3])
    w, l, h = size_wlh
    return np.array([*center_lidar, l, w, h, yaw], dtype=np.float32)


class _Tables:
    """The nuScenes relational tables for one split, indexed by token."""

    def __init__(self, dataroot, version):
        self.dataroot = dataroot
        table_dir = os.path.join(dataroot, version)
        for name in ("scene", "sample", "sample_data", "calibrated_sensor",
                     "sensor", "ego_pose", "sample_annotation", "instance", "category"):
            with open(os.path.join(table_dir, name + ".json")) as f:
                rows = json.load(f)
            setattr(self, name, {r["token"]: r for r in rows})

        # sample_token → {channel: keyframe sample_data row}
        self.keyframes = {}
        for sd in self.sample_data.values():
            if not sd["is_key_frame"]:
                continue
            sensor = self.sensor[self.calibrated_sensor[sd["calibrated_sensor_token"]]["sensor_token"]]
            self.keyframes.setdefault(sd["sample_token"], {})[sensor["channel"]] = sd

        # sample_token → [annotation rows]
        self.annotations = {}
        for ann in self.sample_annotation.values():
            self.annotations.setdefault(ann["sample_token"], []).append(ann)

    def scene_samples(self, scene):
        """Ordered sample tokens of a scene given by index or name."""
        scenes = sorted(self.scene.values(), key=lambda s: s["name"])
        if isinstance(scene, int):
            row = scenes[scene]
        else:
            row = next(s for s in scenes if s["name"] == scene)
        tokens, token = [], row["first_sample_token"]
        while token:
            tokens.append(token)
            token = self.sample[token]["next"]
        return row["name"], tokens

    def sensor_to_ego(self, sd):
        cs = self.calibrated_sensor[sd["calibrated_sensor_token"]]
        return transform_matrix(cs["translation"], cs["rotation"])

    def ego_to_global(self, sd):
        pose = self.ego_pose[sd["ego_pose_token"]]
        return transform_matrix(pose["translation"], pose["rotation"])


class NuScenesSequence(SequenceDataset):
    """Adapter for one scene of a nuScenes split (keyframes only, 2 Hz).

    Remember to set the tracker's dt to 0.5 for nuScenes keyframes.
    """

    def __init__(self, dataroot, scene=0, version="v1.0-mini",
                 camera="CAM_FRONT", detections=None):
        """
        Parameters
        ----------
        dataroot   : str  directory containing v1.0-mini/, samples/, sweeps/
        scene      : int or str  scene index (name-sorted) or name, e.g. "scene-0061"
        version    : str  table directory name
        camera     : str  camera channel used for Frame.image
        detections : DetectionSource, optional (see NuScenesGTDetections)
        """
        self.tables = _Tables(dataroot, version)
        self.scene_name, self.sample_tokens = self.tables.scene_samples(scene)
        self.camera = camera
        self.detections = detections

    def __len__(self):
        return len(self.sample_tokens)

    def lidar_to_global(self, idx):
        """4×4 LiDAR-frame → world-frame transform for frame `idx`."""
        sd = self.tables.keyframes[self.sample_tokens[idx]]["LIDAR_TOP"]
        return self.tables.ego_to_global(sd) @ self.tables.sensor_to_ego(sd)

    def __getitem__(self, idx) -> Frame:
        tables = self.tables
        frames = tables.keyframes[self.sample_tokens[idx]]
        lidar_sd, cam_sd = frames["LIDAR_TOP"], frames[self.camera]

        raw = np.fromfile(os.path.join(tables.dataroot, lidar_sd["filename"]),
                          dtype=np.float32).reshape(-1, 5)
        points = raw[:, :4]  # x, y, z, intensity (drop ring index)

        # lidar → ego(t_lidar) → global → ego(t_cam) → camera
        lidar_to_global = tables.ego_to_global(lidar_sd) @ tables.sensor_to_ego(lidar_sd)
        global_to_cam   = np.linalg.inv(tables.ego_to_global(cam_sd) @ tables.sensor_to_ego(cam_sd))
        lidar_to_cam    = global_to_cam @ lidar_to_global

        K = np.array(tables.calibrated_sensor[cam_sd["calibrated_sensor_token"]]["camera_intrinsic"])
        projection = np.hstack([K, np.zeros((3, 1))]).astype(np.float32)

        detections = (
            self.detections.get(idx, points)
            if self.detections is not None else Detections.empty()
        )

        return Frame(
            frame_id=idx,
            points=points,
            image=cv2.imread(os.path.join(tables.dataroot, cam_sd["filename"])),
            camera=Camera(projection=projection, lidar_to_cam=lidar_to_cam.astype(np.float32)),
            ego_pose=lidar_to_global,
            detections=detections,
        )


class NuScenesGTDetections(DetectionSource):
    """Ground-truth annotations served as detections (score 1.0).

    Lets the full tracking + visualization pipeline run on nuScenes without a
    nuScenes-trained detector; swap in an OpenPCDetSource when one is available.
    Categories without a CATEGORY_MAP entry (barriers, cones, …) are dropped.
    """

    def __init__(self, sequence):
        self._seq = sequence

    def get(self, frame_id, points=None):
        tables = self._seq.tables
        T_lidar_from_global = np.linalg.inv(self._seq.lidar_to_global(frame_id))

        boxes, names = [], []
        for ann in tables.annotations.get(self._seq.sample_tokens[frame_id], []):
            category = tables.category[tables.instance[ann["instance_token"]]["category_token"]]
            name = CATEGORY_MAP.get(category["name"])
            if name is None:
                continue
            boxes.append(global_box_to_lidar(
                ann["translation"], ann["size"], ann["rotation"], T_lidar_from_global
            ))
            names.append(name)

        return Detections(
            boxes=np.array(boxes, dtype=np.float32).reshape(-1, 7),
            scores=np.ones(len(boxes), dtype=np.float32),
            names=names,
        )
