import numpy as np
from scipy.optimize import linear_sum_assignment

from perception.boxes import register_bbs
from perception.tracker.track import Obstacle3D

# Detections may only be associated with tracks from the same class group;
# vehicle classes are grouped because detectors flip between them (e.g. the
# same physical van labelled Car in one frame and Van in the next).
DEFAULT_CLASS_GROUPS = ({"Car", "Van", "Truck"}, {"Pedestrian"}, {"Cyclist"})

# Finite penalty for cross-group pairs: keeps the Hungarian problem feasible
# while guaranteeing rejection by any reasonable dist_threshold.
_GROUP_MISMATCH_COST = 1e6


class Tracker3D:
    """Multi-object 3D tracker: Kalman filtering + Hungarian assignment.

    A 3D SORT extension.  Each frame:
      1. Predict all active tracks forward.
      2. Build a Mahalanobis cost matrix (class-gated) and solve with the
         Hungarian algorithm.
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
            score_threshold (float)  detections below this are ignored (default 0.5)
            dt              (float)  seconds between frames (default 0.1)
            velocity_process_noise (float)  Q-scale for vx,vy,vz (default 1.0)
            class_groups    (tuple[set])  classes that may associate with each
                other (default DEFAULT_CLASS_GROUPS); classes not listed only
                associate with themselves
        """
        config = config or {}
        self.dist_threshold         = config.get("dist_threshold",         4.5)
        self.max_missed             = config.get("max_missed",             3)
        self.min_hits               = config.get("min_hits",               3)
        self.score_threshold        = config.get("score_threshold",        0.5)
        self.dt                     = config.get("dt",                     0.1)
        self.velocity_process_noise = config.get("velocity_process_noise", 1.0)
        self.class_groups           = config.get("class_groups", DEFAULT_CLASS_GROUPS)

        self.trajectories:   list[Obstacle3D] = []
        self.frame_count:    int              = 0
        self._next_id:       int              = 1
        self._confirmed_ids: set[int]         = set()

    def _group_of(self, name):
        """Class-group key for association gating; unknown classes gate on themselves."""
        for i, group in enumerate(self.class_groups):
            if name in group:
                return i
        return name

    def _cost_matrix(self, predictions, detections, groups):
        """Mahalanobis distance matrix (N tracks × M detections), class-gated."""
        cost = np.zeros((len(predictions), len(detections)))
        for i, (pred, traj) in enumerate(zip(predictions, self.trajectories)):
            # Innovation covariance: how much uncertainty exists in the predicted
            # measurement space. Combines state uncertainty (H P Hᵀ) with sensor
            # noise (R) — larger S means the filter expects more spread, so the
            # same physical offset yields a smaller Mahalanobis distance.
            H     = traj.kf.H
            S     = H @ traj.kf.P @ H.T + traj.kf.R
            S_inv = np.linalg.inv(S)

            traj_group = self._group_of(traj.name)
            for j, det in enumerate(detections):
                if groups[j] != traj_group:
                    cost[i, j] = _GROUP_MISMATCH_COST
                    continue
                diff    = det[:7] - pred
                # Wrap yaw residual to (−π, π] so a 350°/10° pair costs the
                # same as a 10° difference, not a 340° one.
                diff[6] = (diff[6] + np.pi) % (2 * np.pi) - np.pi
                # √(diffᵀ S⁻¹ diff): the standard Mahalanobis distance,
                # dimensionless and comparable across all state dimensions.
                cost[i, j] = float(np.sqrt(diff @ S_inv @ diff))
        return cost

    def _associate(self, detections, scores, names):
        """Predict, match, update tracks; return det_index → track_id map."""
        # Step 1 — predict: advance every track's Kalman filter one step.
        predictions = [t.predict() for t in self.trajectories]
        groups      = [self._group_of(n) for n in names]

        matched_dets    = set()
        det_to_track_id = {}

        if self.trajectories and len(detections) > 0:
            # Step 2 — associate: find the globally optimal 1-to-1 assignment
            # between tracks and detections, then accept only pairs whose
            # Mahalanobis distance is within the gate.
            cost                   = self._cost_matrix(predictions, detections, groups)
            track_ind, det_ind     = linear_sum_assignment(cost)
            for track, det in zip(track_ind, det_ind):
                if cost[track, det] < self.dist_threshold:
                    self.trajectories[track].update(detections[det], scores[det])
                    matched_dets.add(det)
                    det_to_track_id[det] = self.trajectories[track].id

        # Step 3 — birth: every unmatched detection spawns a new tentative track.
        # Tracks are tentative until they accumulate min_hits consecutive hits.
        for j, (box, score) in enumerate(zip(detections, scores)):
            if j not in matched_dets:
                self.trajectories.append(Obstacle3D(
                    box, score, self._next_id,
                    velocity_process_noise=self.velocity_process_noise,
                    dt=self.dt, name=names[j],
                ))
                self._next_id += 1

        # Step 4 — death: evict tracks that have been unmatched for too long.
        # Remove from _confirmed_ids first so confirmed status doesn't outlive
        # the track object.
        dead_ids = {
            t.id for t in self.trajectories
            if t.time_since_update >= self.max_missed
        }
        self._confirmed_ids -= dead_ids
        self.trajectories = [
            t for t in self.trajectories if t.time_since_update < self.max_missed
        ]

        return det_to_track_id

    def update(self, boxes, scores, pose=None, names=None):
        """Per-frame entry point.

        Detections at or below score_threshold are ignored here — callers
        pass everything the detector produced.

        Parameters
        ----------
        boxes  : ndarray (M, 7)   canonical [x, y, z, l, w, h, yaw] detection boxes
        scores : ndarray (M,)     confidence scores
        pose   : ndarray (4, 4)   optional ego-vehicle pose for world-frame tracking
        names  : list[str] (M,)   optional class labels for class-gated association

        Returns
        -------
        ids      : list[int]      confirmed track IDs
        bbs      : list[ndarray]  7-D Kalman-filtered box per confirmed track
        scores   : list[float]    latest score per confirmed track
        det_ids  : ndarray (M,)   confirmed track ID per input detection (0 = none)
        """
        self.frame_count += 1

        # Drop low-confidence detections before tracking, but remember the full
        # count so det_ids can be indexed back into the caller's original array.
        scores = np.asarray(scores, dtype=float)
        n_full = len(scores)
        mask   = scores > self.score_threshold

        if names is None:
            names_kept = [None] * int(mask.sum())
        else:
            names_kept = [name for name, keep in zip(names, mask) if keep]

        # np.array() copies so register_bbs doesn't mutate the caller's boxes.
        boxes_kept = np.array(boxes, dtype=np.float64).reshape(-1, 7)[mask, :7]
        if len(boxes_kept) > 0:
            # Transform detections into the world frame so the Kalman filter
            # state and measurements share a fixed reference frame across frames.
            boxes_kept = register_bbs(boxes_kept, pose)

        det_to_track_id = self._associate(boxes_kept, scores[mask], names_kept)

        # Promote any track that has now accumulated enough consecutive hits.
        for t in self.trajectories:
            if t.hit_streak >= self.min_hits:
                self._confirmed_ids.add(t.id)

        # During the warm-up period (first min_hits frames) treat all tracks as
        # confirmed so the visualizer has something to show immediately.
        confirmed_ids = {
            t.id for t in self.trajectories
            if t.id in self._confirmed_ids or self.frame_count <= self.min_hits
        }

        # Map each input detection back to its confirmed track ID (0 = none).
        # det_ids_kept works over the filtered subset; it's then scattered back
        # into a full-length array aligned with the caller's detection list.
        det_ids_kept = np.zeros(int(mask.sum()), dtype=int)
        for det_idx, track_id in det_to_track_id.items():
            if track_id in confirmed_ids:
                det_ids_kept[det_idx] = track_id
        det_ids       = np.zeros(n_full, dtype=int)
        det_ids[mask] = det_ids_kept

        ids, bbs, scores_out = self._get_confirmed()
        return ids, bbs, scores_out, det_ids

    def _get_confirmed(self):
        """Current confirmed tracks as (ids, Kalman-state boxes, scores).

        Boxes come from get_state(), so a track missed this frame still
        reports its coasted (predicted) position rather than vanishing.
        """
        ids, boxes, scores = [], [], []
        for t in self.trajectories:
            # Same warm-up grace as the confirmed_ids set in update(): during
            # the first min_hits frames no track can be confirmed yet, so all
            # are reported to avoid an empty start of every sequence.
            if t.id in self._confirmed_ids or self.frame_count <= self.min_hits:
                ids.append(t.id)
                boxes.append(t.get_state())
                scores.append(t.score)
        return ids, boxes, scores
