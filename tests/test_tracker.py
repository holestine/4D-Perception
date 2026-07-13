import numpy as np
import pytest

from perception.tracker.mot import Tracker3D
from perception.tracker.track import Obstacle3D

EMPTY = np.zeros((0, 7))


def box(x, y=0.0):
    return np.array([x, y, 0.0, 4.0, 2.0, 1.5, 0.0])


def make_tracker(**overrides):
    config = {"min_hits": 3, "max_missed": 3, "dist_threshold": 6.0}
    config.update(overrides)
    return Tracker3D(config=config)


def step(tracker, *boxes):
    dets = np.array(boxes) if boxes else EMPTY
    scores = np.ones(len(dets))
    return tracker.update(dets, scores)


class TestObstacle3D:
    def test_learns_constant_velocity(self):
        # target moves +1 m per 0.1 s frame → 10 m/s
        obs = Obstacle3D(box(0.0), score=1.0, track_id=1)
        for k in range(1, 10):
            obs.predict()
            obs.update(box(float(k)), score=1.0)
        predicted_x = obs.predict()[0]
        assert predicted_x == pytest.approx(10.0, abs=1.0)

    def test_hit_streak_resets_on_miss(self):
        obs = Obstacle3D(box(0.0), score=1.0, track_id=1)
        for k in (1.0, 2.0):
            obs.predict()
            obs.update(box(k), 1.0)
        assert obs.hit_streak == 2
        obs.predict()          # missed frame
        obs.predict()          # hit_streak resets once a miss is observed
        assert obs.hit_streak == 0

    def test_dt_scales_the_motion_model(self):
        obs = Obstacle3D(box(0.0), score=1.0, track_id=1, dt=0.5)
        np.testing.assert_allclose(obs.kf.F[0:3, 7:10], 0.5 * np.eye(3))

    def test_tracker_assigns_unique_ids_per_instance(self):
        # two trackers each start their own ID sequence at 1
        for _ in range(2):
            tracker = make_tracker()
            _, _, _, _ = step(tracker, box(0.0, y=0.0), box(0.0, y=50.0))
            ids = sorted(t.id for t in tracker.trajectories)
            assert ids == [1, 2]


class TestTrackerLifecycle:
    def test_bootstrap_frames_output_immediately(self):
        # during the first min_hits frames every track is provisionally confirmed
        tracker = make_tracker()
        ids, _, _, det_ids = step(tracker, box(0.0))
        assert len(ids) == 1
        assert det_ids[0] == 0          # spawn frame: not yet matched to a track
        ids, _, _, det_ids = step(tracker, box(1.0))
        assert det_ids[0] == ids[0]     # second frame: matched

    def test_new_track_needs_min_hits_after_bootstrap(self):
        tracker = make_tracker(min_hits=3)
        for _ in range(4):           # move past the bootstrap window
            step(tracker)

        confirmed_at = None
        for k in range(5):           # detection appears and persists
            ids, _, _, _ = step(tracker, box(float(k)))
            if ids and confirmed_at is None:
                confirmed_at = k
        # spawn frame (streak 0) + 3 consecutive hits → confirmed on 4th frame
        assert confirmed_at == 3

    def test_track_pruned_after_max_missed(self):
        tracker = make_tracker(max_missed=3)
        for k in range(4):
            ids, _, _, _ = step(tracker, box(float(k)))
        track_id = ids[0]

        alive = []
        for _ in range(4):
            ids, _, _, _ = step(tracker)   # no detections
            alive.append(track_id in ids)
        # survives while time_since_update < max_missed, then evicted
        assert alive == [True, True, False, False]

    def test_confirmation_survives_missed_frame(self):
        tracker = make_tracker(min_hits=3, max_missed=4)
        for k in range(4):
            ids, _, _, _ = step(tracker, box(float(k)))
        track_id = ids[0]

        ids, _, _, _ = step(tracker)                    # miss one frame
        assert track_id in ids                          # still output (coasting)

        ids, _, _, det_ids = step(tracker, box(5.0))    # reappears
        assert track_id in ids
        assert det_ids[0] == track_id                   # matched, not re-spawned


class TestAssociation:
    def test_det_ids_map_to_consistent_tracks(self):
        tracker = make_tracker()
        step(tracker, box(0.0, y=0.0), box(0.0, y=20.0))   # spawn frame
        first = step(tracker, box(1.0, y=0.0), box(1.0, y=20.0))[3]
        assert first[0] != first[1] and 0 not in first

        for k in range(2, 5):
            _, _, _, det_ids = step(tracker, box(float(k), y=0.0), box(float(k), y=20.0))
        np.testing.assert_array_equal(det_ids, first)

    def test_distant_detection_spawns_new_track(self):
        tracker = make_tracker(dist_threshold=6.0)
        step(tracker, box(0.0))
        # far beyond the Mahalanobis gate → must spawn instead of matching
        step(tracker, box(80.0))
        assert len(tracker.trajectories) == 2

    def test_input_boxes_not_mutated(self):
        tracker = make_tracker()
        dets = np.array([box(1.0)])
        original = dets.copy()
        pose = np.eye(4)
        pose[:3, 3] = [100.0, 50.0, 2.0]
        tracker.update(dets, np.ones(1), pose=pose)
        np.testing.assert_array_equal(dets, original)

    def test_world_frame_registration(self):
        tracker = make_tracker()
        pose = np.eye(4)
        pose[:3, 3] = [100.0, 0.0, 0.0]
        _, bbs, _, _ = tracker.update(np.array([box(5.0)]), np.ones(1), pose=pose)
        assert bbs[0][0] == pytest.approx(105.0, abs=1e-6)

    def test_score_threshold_filters_inside_update(self):
        tracker = make_tracker(score_threshold=0.5)
        dets = np.array([box(0.0), box(0.0, y=20.0)])
        ids, _, _, det_ids = tracker.update(dets, np.array([0.9, 0.3]))
        assert len(tracker.trajectories) == 1        # low-score det never spawned
        assert det_ids.shape == (2,)                 # but det_ids covers all inputs


class TestClassGating:
    def test_different_group_cannot_match(self):
        tracker = make_tracker()
        # confirm a Car track
        for k in range(3):
            step_named(tracker, (box(float(k)), "Car"))
        n_tracks = len(tracker.trajectories)
        # a Pedestrian at the same location must spawn, not update the car
        step_named(tracker, (box(3.0), "Pedestrian"))
        assert len(tracker.trajectories) == n_tracks + 1

    def test_vehicle_classes_share_a_group(self):
        tracker = make_tracker()
        step_named(tracker, (box(0.0), "Car"))
        track_id = tracker.trajectories[0].id
        # detector relabels the same object as Van → must still match
        _, _, _, det_ids = step_named(tracker, (box(1.0), "Van"))
        assert det_ids[0] == track_id
        assert len(tracker.trajectories) == 1

    def test_no_names_means_no_gating(self):
        tracker = make_tracker()
        step(tracker, box(0.0))
        _, _, _, det_ids = step(tracker, box(1.0))
        assert det_ids[0] != 0


def step_named(tracker, *dets):
    boxes = np.array([b for b, _ in dets]) if dets else EMPTY
    names = [n for _, n in dets]
    return tracker.update(boxes, np.ones(len(boxes)), names=names)
