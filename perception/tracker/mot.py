import numpy as np
from scipy.optimize import linear_sum_assignment

from multi_object_tracking.tracker.box_op import convert_bbs_type, register_bbs
from perception.tracker.track import Obstacle3D


class Tracker3D:
    """Multi-object 3D tracker: Kalman filtering + Hungarian assignment.

    A 3D SORT extension.  Each frame:
      1. Predict all active tracks forward.
      2. Build a Mahalanobis cost matrix and solve with the Hungarian algorithm.
      3. Update matched tracks, spawn tracks for unmatched detections,
         prune tracks that have been missed too long.

    Tracks are *confirmed* once they accumulate min_hits consecutive hits.
    Confirmed status is permanent until the track is pruned — this prevents
    intermittently-detected large vehicles from flickering in/out.
    """

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict, optional
            dist_threshold  (float)  Mahalanobis gate (default 4.5)
            max_missed      (int)    frames before a track is dropped (default 3)
            min_hits        (int)    consecutive detections to confirm (default 3)
            score_threshold (float)  minimum score fed to tracker (default 0.5)
            box_type        (str)    'Kitti', 'OpenPCDet', or 'Waymo' (default 'Kitti')
            velocity_process_noise (float)  Q-scale for vx,vy,vz (default 1.0)
        """
        config = config or {}
        self.dist_threshold         = config.get("dist_threshold",         4.5)
        self.max_missed             = config.get("max_missed",             3)
        self.min_hits               = config.get("min_hits",               3)
        self.score_threshold        = config.get("score_threshold",        0.5)
        self.box_type               = config.get("box_type",               "Kitti")
        self.velocity_process_noise = config.get("velocity_process_noise", 1.0)

        self.trajectories:   list[Obstacle3D] = []
        self.frame_count:    int              = 0
        self._confirmed_ids: set[int]         = set()

    def _cost_matrix(self, predictions, detections):
        """Mahalanobis distance matrix (N tracks × M detections)."""
        cost = np.zeros((len(predictions), len(detections)))
        for i, (pred, traj) in enumerate(zip(predictions, self.trajectories)):
            H     = traj.kf.H
            S     = H @ traj.kf.P @ H.T + traj.kf.R
            S_inv = np.linalg.inv(S)
            for j, det in enumerate(detections):
                diff    = det[:7] - pred
                diff[6] = (diff[6] + np.pi) % (2 * np.pi) - np.pi
                cost[i, j] = float(np.sqrt(diff @ S_inv @ diff))
        return cost

    def _associate(self, detections, scores):
        """Predict, match, update tracks; return det_index → track_id map."""
        predictions = [t.predict() for t in self.trajectories]

        matched_dets    = set()
        det_to_track_id = {}

        if self.trajectories and len(detections) > 0:
            cost             = self._cost_matrix(predictions, detections)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.dist_threshold:
                    self.trajectories[r].update(detections[c], scores[c])
                    matched_dets.add(c)
                    det_to_track_id[c] = self.trajectories[r].id

        for j, (box, score) in enumerate(zip(detections, scores)):
            if j not in matched_dets:
                self.trajectories.append(
                    Obstacle3D(box, score, self.velocity_process_noise)
                )

        dead_ids = {
            t.id for t in self.trajectories
            if t.time_since_update >= self.max_missed
        }
        self._confirmed_ids -= dead_ids
        self.trajectories = [
            t for t in self.trajectories if t.time_since_update < self.max_missed
        ]

        return det_to_track_id

    def update(self, boxes, scores, pose=None):
        """Per-frame entry point.

        Parameters
        ----------
        boxes  : ndarray (M, 7)   detection boxes in box_type format
        scores : ndarray (M,)     confidence scores
        pose   : ndarray (4, 4)   optional ego-vehicle pose for world-frame tracking

        Returns
        -------
        ids      : list[int]      confirmed track IDs
        bbs      : list[ndarray]  7-D Kalman-filtered box per confirmed track
        scores   : list[float]    latest score per confirmed track
        det_ids  : ndarray (M,)   confirmed track ID per input detection (0 = none)
        """
        self.frame_count += 1
        n = len(boxes)

        if n > 0:
            boxes = convert_bbs_type(boxes, self.box_type)
            boxes = register_bbs(boxes, pose)

        det_to_track_id = self._associate(boxes, scores)

        for t in self.trajectories:
            if t.hit_streak >= self.min_hits:
                self._confirmed_ids.add(t.id)

        confirmed_ids = {
            t.id for t in self.trajectories
            if t.id in self._confirmed_ids or self.frame_count <= self.min_hits
        }

        det_ids = np.zeros(n, dtype=int)
        for det_idx, track_id in det_to_track_id.items():
            if track_id in confirmed_ids:
                det_ids[det_idx] = track_id

        ids, bbs, scores_out = self._get_confirmed()
        return ids, bbs, scores_out, det_ids

    def _get_confirmed(self):
        ids, boxes, scores = [], [], []
        for t in self.trajectories:
            if t.id in self._confirmed_ids or self.frame_count <= self.min_hits:
                ids.append(t.id)
                boxes.append(t.get_state())
                scores.append(t.score)
        return ids, boxes, scores
