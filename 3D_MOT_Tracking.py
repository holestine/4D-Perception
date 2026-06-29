# Standard library
import os
import random
import re
import time

# Third-party libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation as R

from multi_object_tracking.dataset.kitti_data_base import *

class KittiDetectionDataset:
    def __init__(self,root_path,seq_id, label_path = None):
        self.seq_name = str(seq_id).zfill(4)
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne", self.seq_name)
        self.image_path = os.path.join(self.root_path,"image_02", self.seq_name)
        self.calib_path = os.path.join(self.root_path,"calib")
        self.label_path = label_path
        pose_path = os.path.join(self.root_path, "pose", self.seq_name,'pose.txt')
        self.poses = read_pose(pose_path)

        first_5_poses = dict(list(self.poses.items())[:5])

        """"
        for key, mat in first_5_poses.items():
            print(f"Pose {key}:")
            print("[")
            for row in mat:
                formatted_row = ", ".join(f"{v: .6e}" for v in row)
                print(f" [{formatted_row}],")
            print("]\n")
        """

    def __len__(self):
        return len(os.listdir(self.velo_path))

    def __getitem__(self, item):
        input_dict = {}
        name = str(item).zfill(6)
        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')
        calib_path = os.path.join(self.calib_path, self.seq_name+'.txt')

        input_dict["frame_id"] = item

        pose = self.poses[item] if item in self.poses.keys() else None;
        input_dict["pose"] = pose

        P2,V2C = read_calib(calib_path)
        input_dict["P2"] = P2
        input_dict["V2C"] = V2C

        points = read_velodyne(velo_path,P2,V2C)
        input_dict["points"] = points

        image = read_image(image_path)
        input_dict["image"] = image

        objects, objects_cam, det_scores, det_names = [], [], [], []

        if self.label_path is not None:
            # If we have a label path, we'll read the labels instead of predicting them
            label_path = os.path.join(self.label_path, self.seq_name, name+'.txt')
            objects, det_scores, det_names = self.read_detection_label(label_path)
        if len(objects)>0:
            objects_cam = np.copy(objects)
            objects[:,3:6] = cam_to_velo(objects[:,3:6],V2C)[:,:3]
        input_dict["objects"] = objects
        input_dict["objects_cam"] = objects_cam
        input_dict["scores"] = det_scores
        input_dict["names"] = det_names

        return input_dict

    def read_detection_label(self,label_path):
        objects_list = []
        det_scores = []
        det_names = []
        with open(label_path) as f:
            for each_ob in f.readlines():
                infos = re.split(' ', each_ob)
                if infos[0] in ['Car', 'Truck','Van', 'Cyclist']:
                    objects_list.append(infos[8:15])
                    det_scores.append(infos[15])
                    det_names.append(infos[0])
        return np.array(objects_list, np.float32), np.array(det_scores, np.float32), det_names


root="multi_object_tracking/data"
label_path = "multi_object_tracking/detectors/point_rcnn"
dataset = KittiDetectionDataset(root,seq_id=8,label_path=label_path)

data = dataset[0]
objects = data['objects']
print(objects.shape)

def view_images(images_to_be_shown):
  _, axs = plt.subplots(1, len(images_to_be_shown), figsize=(30, 30))

  if len(images_to_be_shown)> 1:
    axs = axs.flatten()
    for img, ax in zip(images_to_be_shown, axs):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  else:
        axs.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  plt.show()


hand_picked_frames = [0,1,2,3]
images_2d = [dataset[i]["image"] for i in hand_picked_frames]

#view_images(images_2d)

def project_3d_box_to_image(bbox_3d, P2):
    h, w, l, x, y, z, ry = bbox_3d
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ]
    y_corners = [  0,    0,    0,    0,   -h,   -h,   -h,   -h ]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2 ]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    R_y = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [         0, 1,         0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = R_y @ corners_3d + np.array([[x], [y], [z]])
    corners_3d_hom = np.vstack((corners_3d, np.ones((1, 8))))
    corners_2d = P2 @ corners_3d_hom
    corners_2d = corners_2d[:2] / corners_2d[2]
    return corners_2d.T

def visualize(dataset, frames=None, score_threshold=3, viz_labels=False, out_file=None):
    rr.init("KITTI Visualizer", recording_id="new_run", spawn=False)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/ego_vehicle/camera", name="Camera"),
            rrb.Spatial3DView(origin="world/ego_vehicle/lidar", name="LiDAR"),
            column_shares=[3, 1]
        )
    )

    rr.log("world/ego_vehicle/camera/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
    rr.log("world/ego_vehicle/lidar/", rr.ViewCoordinates.RIGHT_HAND_Z_UP)


    class_colors = {
        'Car': [255, 0, 0, 128],
        'Pedestrian': [0, 255, 0, 128],
        'Cyclist': [0, 0, 255, 128]
    }

    if frames is None:
        frames = list(range(len(dataset)))

    for i in frames:
        rr.set_time_sequence("frame", i)
        rr.log("world/ego_vehicle/camera/image/detections", rr.Clear(recursive=True))
        rr.log("world/ego_vehicle/lidar/points", rr.Clear(recursive=False))
        rr.log("world/ego_vehicle/lidar/objects", rr.Clear(recursive=True))

        data = dataset[i]
        # print(f"the keys are {dataset[i].keys()}")
        image = np.array(data['image'])
        points = data['points']
        P2, V2C = data['P2'], data['V2C']
        pose = data['pose']

        if pose is not None:
            # Extract translation and rotation from pose matrix
            translation = pose[:3, 3]  # Extract translation from 4x4 pose matrix
            rotation_matrix = pose[:3, :3]  # Extract rotation matrix
            # Convert rotation matrix to quaternion
            r = R.from_matrix(rotation_matrix)
            quat_xyzw = r.as_quat()  # scipy returns [x, y, z, w]

            rr.log("world/ego_vehicle", rr.Transform3D(
                translation=translation,
                quaternion=rr.Quaternion(xyzw=quat_xyzw),
                relation=rr.TransformRelation.ParentFromChild,
            ))

        rr.log("world/ego_vehicle/camera/image", rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        positions = points[:, :3]
        distances = np.linalg.norm(positions, axis=1)
        norm = (255.0 * (distances - distances.min()) / (np.ptp(distances) + 1e-5)).astype(np.uint8)
        colors = (plt.cm.cividis(norm / 255.0)[:, :3] * 255).astype(np.uint8)
        rr.log("world/ego_vehicle/lidar/points", rr.Points3D(positions=positions, colors=colors))

        if viz_labels:
            centers, sizes, rotations, labels, box_colors = [], [], [], [], []

            for j, (box, box_cam, score, name) in enumerate(zip(data['objects'], data['objects_cam'], data['scores'], data['names'])):
                if score < score_threshold:
                    continue

                # color = class_colors.get(name, [128, 128, 128, 128])  # RGBA

                color = [
                    random.randint(0, 255),  # R
                    random.randint(0, 255),  # G
                    random.randint(0, 255),  # B
                    180  # Alpha (semi-transparent)
                    ]

                # # Map score to a color (e.g., low score = blue, high score = red)
                # norm_score = (score - score_threshold) / (5.0 - score_threshold)  # Scale to [0,1]
                # color_rgb = cm.viridis(norm_score)[:3]  # Get RGB (ignore alpha)
                # color = [int(c * 255) for c in color_rgb] + [128]

                # 3D Camera Boxes
                corners_2d = project_3d_box_to_image(box_cam,P2)
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],
                    [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]
                ]
                lines = [np.array([corners_2d[start], corners_2d[end]]) for start, end in edges]
                # box_colors.append(class_colors.get(name, [128, 128, 128, 128]))

                rr.log(f"world/ego_vehicle/camera/image/detections/box_{j}", rr.LineStrips2D(lines, labels=[f"{name} {score:.2f}"], colors=np.array([color])))
                #rr.log(f"world/ego_vehicle/camera/image/detections/box_{j}/label", rr.Points2D([np.mean(corners_2d, axis=0)], colors=np.array([color])))


                # 3D Lidar boxes
                h, w, l, x, y, z, ry = box
                # Convert KITTI rotation to Rerun coordinate system
                yaw = -ry - np.pi / 2
                center = np.array([x, y, z + h / 2])
                size = np.array([l, w, h])

                # Create quaternion for rotation around Z-axis in Rerun's coordinate system
                quat = R.from_euler('z', yaw, degrees=False).as_quat()
                # Rerun expects [w, x, y, z] format
                quat = np.array([quat[0], quat[1], quat[2], quat[3]])


                rr.log(f"world/ego_vehicle/lidar/objects/box_{j}", rr.Transform3D(
                    translation=center,
                    quaternion=rr.Quaternion(xyzw=quat),
                    relation=rr.TransformRelation.ParentFromChild,
                ))

                rr.log(f"world/ego_vehicle/lidar/objects/box_{j}/model", rr.Asset3D(
                    path="multi_object_tracking/viewer/car.obj",
                    albedo_factor=np.array(color[:3]) / 255.0
                ))


    # Save and display
    if out_file is not None:
        rr.save(out_file)
    else:
        rr.notebook_show(height=500, width=1000)


visualize(dataset, frames=hand_picked_frames, viz_labels=False, out_file="rerun_output_1.rrd")

# Visualize the LiDAR point cloud and 2d Image in a video sequence without labels
visualize(dataset, frames=hand_picked_frames, viz_labels=True, out_file="rerun_output_2.rrd")

from multi_object_tracking.tracker.box_op import *
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

LIDAR_SCANNING_FREQUENCY = 10  # Hz
DT = 1.0 / LIDAR_SCANNING_FREQUENCY


class Obstacle3D:
    """Individual 3D track using a constant-velocity Kalman filter.
    Tracks the full 7D box [x,y,z,l,w,h,yaw] with position velocity.
    Mirrors the trajectory class in tracking/tracker.py."""

    current_id = 1

    def __init__(self, box, score):
        self.id = Obstacle3D.current_id
        Obstacle3D.current_id += 1

        self.time_since_update = 0
        self.hit_streak = 0
        self.score = score

        self._init_kalman(box)

    def _init_kalman(self, box):
        # 10D state: [x, y, z, l, w, h, yaw, vx, vy, vz]
        # 7D measurement: [x, y, z, l, w, h, yaw]
        self.kf = KalmanFilter(dim_x=10, dim_z=7)

        # H observes the first 7 state components
        self.kf.H = np.eye(10)[0:7]

        # Constant-velocity for position; shape/yaw/velocity modeled as constant
        self.kf.F = np.eye(10)
        self.kf.F[0:3, 7:10] = DT * np.eye(3)  # pos += vel * dt

        # Seed state from first detection
        self.kf.x[:7] = box[:7].reshape(7, 1)

        # High uncertainty on initial velocity; confident on initial box
        self.kf.P = np.eye(10)
        self.kf.P[7:10, 7:10] *= 100

        # Process noise: shape/yaw change slowly, velocity can drift
        self.kf.Q = np.eye(10) * 0.1
        self.kf.Q[3:7, 3:7] *= 0.05   # l,w,h,yaw are nearly constant
        self.kf.Q[7:10, 7:10] *= 1.0  # velocity allowed more variation

        # Measurement noise: position noisier than shape
        self.kf.R = np.eye(7) * 1.0
        self.kf.R[3:7, 3:7] *= 0.5    # detector is consistent on shape/yaw

    def predict(self):
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x[:7].reshape(-1)

    def update(self, box, score):
        self.time_since_update = 0
        self.hit_streak += 1
        self.score = score
        self.kf.update(box[:7].reshape(7, 1))

    def get_state(self):
        return self.kf.x[:7].reshape(-1)


class Tracker3D:
    """Multi-object 3D tracker using Hungarian assignment.
    Mirrors the bb_tracker class in tracking/tracker.py."""

    def __init__(self, config=None):
        config = config or {}
        self.dist_threshold     = config.get('dist_threshold', 3.0)
        self.max_missed         = config.get('max_missed', 3)
        self.min_hits           = config.get('min_hits', 3)
        self.score_threshold    = config.get('score_threshold', 0.5)
        self.box_type           = config.get('box_type', 'Kitti')

        self.trajectories = []
        self.frame_count  = 0

    def _cost_matrix(self, predictions, detections):
        """Combined cost: position distance + shape mismatch + yaw difference."""
        cost = np.zeros((len(predictions), len(detections)))
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                pos_dist = np.linalg.norm(pred[:3] - det[:3])
                # Normalized shape distance (l, w, h) — penalizes matching a car to a cyclist
                avg_dims = np.maximum(pred[3:6], det[3:6]) + 1e-3
                shape_dist = float(np.sum(np.abs(pred[3:6] - det[3:6]) / avg_dims))
                # Circular yaw distance in [0, 2]; 0 = same heading, 2 = opposite
                yaw_dist = 1.0 - np.cos(pred[6] - det[6])
                cost[i, j] = pos_dist + 0.5 * shape_dist + 0.3 * yaw_dist
        return cost

    def associate(self, detections, scores):
        """Predict all tracks, run Hungarian assignment, update matched tracks,
        spawn new tracks for unmatched detections, cull stale tracks."""

        predictions = [t.predict() for t in self.trajectories]

        matched_tracks = set()
        matched_dets   = set()

        if self.trajectories and len(detections) > 0:
            cost = self._cost_matrix(predictions, detections)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.dist_threshold:
                    self.trajectories[r].update(detections[c], scores[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

        # New tracks for unmatched detections
        for j, (box, score) in enumerate(zip(detections, scores)):
            if j not in matched_dets:
                self.trajectories.append(Obstacle3D(box, score))

        # Remove tracks that haven't been seen for too long
        self.trajectories = [t for t in self.trajectories
                             if t.time_since_update < self.max_missed]

    def update(self, boxes, scores, pose=None):
        """Main per-frame entry point. Returns (ids, boxes, scores) for
        confirmed tracks (hit_streak >= min_hits)."""
        self.frame_count += 1

        if len(boxes) > 0:
            boxes = convert_bbs_type(boxes, self.box_type)
            boxes = register_bbs(boxes, pose)

        self.associate(boxes, scores)
        return self.get_matches()

    def get_matches(self):
        """Return confirmed tracks, mirroring bb_tracker.get_matches()."""
        ids, boxes, scores = [], [], []
        for t in self.trajectories:
            if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                ids.append(t.id)
                boxes.append(t.get_state())
                scores.append(t.score)
        return ids, boxes, scores


hand_picked_frames = range(0, 100)

tracker = Tracker3D()

all_time  = 0
frame_num = 0
final_bbs = []
final_ids = []

for i in hand_picked_frames:
    data = dataset[i]

    pose        = data['pose']
    objects     = data['objects']
    scores      = np.array(data['scores'], dtype=float)

    mask    = scores > tracker.score_threshold
    objects = objects[mask, :7]
    scores  = scores[mask]

    start = time.time()
    ids, bbs, _ = tracker.update(objects, scores, pose=pose)
    end   = time.time()

    final_bbs.append(np.array(bbs) if bbs else np.zeros((0, 7)))
    final_ids.append(ids)

    all_time  += end - start
    frame_num += 1

print(f"Tracked {frame_num} frames in {all_time:.2f}s ({frame_num/all_time:.1f} fps)")
print(f"Final track ids: {final_ids}")

print(final_ids)

print(final_bbs[0])

print(hand_picked_frames)

def visualize_tracking(dataset, hand_picked_frames, final_ids, threshold=4, out_file=None):
    rr.init("KITTI Visualizer Tracking", spawn=False)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/ego_vehicle/camera", name="Camera"),
            rrb.Spatial3DView(origin="world/ego_vehicle/lidar", name="LiDAR"),
            column_shares=[1, 1]
        )
    )

    rr.log("world/ego_vehicle/camera/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
    rr.log("world/ego_vehicle/lidar/", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    if hand_picked_frames is None:
        frames = list(range(len(dataset)))
    else:
        frames = hand_picked_frames

    for idx, i in enumerate(frames):
        rr.set_time_sequence("frame", i)
        rr.log("world/ego_vehicle/camera/image/detections", rr.Clear(recursive=True))
        rr.log("world/ego_vehicle/lidar/points", rr.Clear(recursive=False))
        rr.log("world/ego_vehicle/lidar/boxes", rr.Clear(recursive=True))
        rr.log("world/ego_vehicle/lidar/models", rr.Clear(recursive=True))

        data = dataset[i]

        pose        = data['pose']
        P2          = data['P2']
        V2C         = data['V2C']
        points      = data['points']
        image       = data['image']
        objects     = data['objects']
        objects_cam = data['objects_cam']
        det_scores  = data['scores']
        det_names   = data['names']

        if pose is not None:
            # Extract translation and rotation from pose matrix
            translation = pose[:3, 3]  # Extract translation from 4x4 pose matrix
            rotation_matrix = pose[:3, :3]  # Extract rotation matrix
            # Convert rotation matrix to quaternion
            r = R.from_matrix(rotation_matrix)
            quat_xyzw = r.as_quat()  # scipy returns [x, y, z, w]

            rr.log("world/ego_vehicle", rr.Transform3D(
                translation=translation,
                quaternion=rr.Quaternion(xyzw=quat_xyzw),
                relation=rr.TransformRelation.ParentFromChild,
            ))

            # ===== ADD EGO VEHICLE MODEL =====
            rr.log(
                "world/ego_vehicle/lidar/car_model",
                rr.Asset3D(
                    path="multi_object_tracking/viewer/ego_car.3ds",  # .glb unavailable,
                    albedo_factor = [1., 1., 1.]
                )
            )

        rr.log("world/ego_vehicle/camera/image", rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        # Log point cloud with distance-based coloring
        positions = points[:, :3]
        distances = np.linalg.norm(positions, axis=1)
        norm = (255.0 * (distances - distances.min()) / (np.ptp(distances) + 1e-5)).astype(np.uint8)
        colors = (plt.cm.cividis(norm / 255.0)[:, :3] * 255).astype(np.uint8)
        rr.log("world/ego_vehicle/lidar/points", rr.Points3D(positions=positions, colors=colors))

        mask = det_scores > threshold
        objects = objects_cam[mask]
        scores = det_scores[mask]
        ids = final_ids[idx]
        names = np.array(det_names)[mask]

        centers, sizes, rotations, labels, color_ids = [], [], [], [], []

        for obj, id, name in zip(objects, ids, names):
            h, w, l, x, y, z, ry, track_id = obj.tolist() + [id]

            box_cam = np.array([h, w, l, x, y, z, ry])
            # 3D Camera Boxes
            corners_2d = project_3d_box_to_image(box_cam, P2)
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            lines = [np.array([corners_2d[start], corners_2d[end]]) for start, end in edges]

            if l * w * h == 0:  # Skip invalid boxes
                continue

            # KITTI coordinate system: X=right, Y=down, Z=forward
            # Rerun RIGHT_HAND_Z_UP: X=right, Y=forward, Z=up
            center = np.array([z, -x, -y + h / 2])  # Transform KITTI to Rerun coordinates
            size = np.array([l, w, h])

            # Fix rotation: Ensure boxes are horizontal and properly oriented
            # For horizontal boxes in the XY plane, we need to adjust the rotation
            yaw = ry + np.pi / 2  # Adjust rotation to keep boxes horizontal
            quat = R.from_euler('z', yaw, degrees=False).as_quat()
            quat_rerun = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w,x,y,z]

            # Generate unique color per track ID
            cmap = plt.get_cmap("tab20")  # or "tab10", "hsv", etc.
            color = (np.array(cmap(track_id % cmap.N)) * 255).astype(np.uint8)

            rr.log(f"world/ego_vehicle/camera/image/detections/box_{track_id}", rr.LineStrips2D(
                lines,
                labels=[f"{track_id}"],
                colors=np.array(color)
            ))

            centers.append(center)
            sizes.append(size)
            rotations.append(quat)
            labels.append(f"{int(track_id)}")
            color_ids.append(color)

            rr.log(f"world/ego_vehicle/lidar/models/car_{track_id}", rr.Transform3D(
                translation=center,
                quaternion=rr.Quaternion(xyzw=quat),
                scale=[0.4, 0.4, 0.4],
                relation=rr.TransformRelation.ParentFromChild,
            ))
            rr.log(f"world/ego_vehicle/lidar/models/car_{track_id}/model", rr.Asset3D(
                path="multi_object_tracking/viewer/car.obj",
                albedo_factor=color / 255.0
            ))

        # Log 3D boxes
        if centers:
            rr.log("world/ego_vehicle/lidar/boxes", rr.Boxes3D(
                centers=np.array(centers),
                half_sizes=np.array(sizes) / 2,
                quaternions=np.array(rotations),
                labels=labels,
                colors=np.array(color_ids),
            ))

    # Save and display

    if out_file is not None:
        rr.save(out_file)
    else:
        rr.notebook_show(height=500, width=1000)

visualize_tracking(dataset, hand_picked_frames, final_ids, threshold=4, out_file="rerun_output_3.rrd")
