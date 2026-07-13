import numpy as np
import pytest

from perception.evaluation import evaluate_tracking, format_summary, read_tracking_labels

GT_LINE = ("{frame} {tid} {cls} 0 0 -1.5 0 0 50 50 "
           "1.5 1.8 4.0 -2.0 1.25 10.0 0.4\n")


class TestReadTrackingLabels:
    def test_parses_frames_ids_and_boxes(self, tmp_path):
        p = tmp_path / "0008.txt"
        p.write_text(
            GT_LINE.format(frame=0, tid=1, cls="Car") +
            GT_LINE.format(frame=0, tid=2, cls="Van") +
            GT_LINE.format(frame=1, tid=1, cls="Car") +
            "0 -1 DontCare -1 -1 -10 457 185 505 215 -1000 -1000 -1000 -10 -1 -1 -1\n" +
            GT_LINE.format(frame=2, tid=7, cls="Pedestrian")   # not a vehicle class
        )
        gt = read_tracking_labels(str(p))
        assert set(gt.keys()) == {0, 1}
        boxes0, ids0 = gt[0]
        assert boxes0.shape == (2, 7)
        np.testing.assert_array_equal(ids0, [1, 2])
        np.testing.assert_allclose(boxes0[0], [1.5, 1.8, 4.0, -2.0, 1.25, 10.0, 0.4])

    def test_class_filter_is_configurable(self, tmp_path):
        p = tmp_path / "labels.txt"
        p.write_text(GT_LINE.format(frame=0, tid=3, cls="Pedestrian"))
        assert read_tracking_labels(str(p)) == {}
        gt = read_tracking_labels(str(p), classes=("Pedestrian",))
        assert list(gt[0][1]) == [3]


def _xy(*pairs):
    return np.array(pairs, dtype=float).reshape(-1, 2)


class TestEvaluateTracking:
    def test_perfect_tracking(self):
        frames = [
            ([1, 2], _xy((0, 0), (10, 0)), [11, 12], _xy((0.1, 0), (10.1, 0))),
            ([1, 2], _xy((1, 0), (11, 0)), [11, 12], _xy((1.1, 0), (11.1, 0))),
        ]
        m = evaluate_tracking(frames, dist_threshold=2.0)
        assert m["mota"] == pytest.approx(1.0)
        assert m["idf1"] == pytest.approx(1.0)
        assert m["num_switches"] == 0
        assert m["motp"] == pytest.approx(0.1, abs=1e-6)
        assert m["num_objects"] == 4

    def test_missed_detection_counts_fn(self):
        frames = [
            ([1], _xy((0, 0)), [11], _xy((0, 0))),
            ([1], _xy((1, 0)), [],   _xy()),          # tracker lost it
            ([1], _xy((2, 0)), [11], _xy((2, 0))),
        ]
        m = evaluate_tracking(frames, dist_threshold=2.0)
        assert m["num_misses"] == 1
        assert m["mota"] == pytest.approx(2 / 3)

    def test_false_positive_counts_fp(self):
        frames = [([1], _xy((0, 0)), [11, 99], _xy((0, 0), (50, 50)))]
        m = evaluate_tracking(frames, dist_threshold=2.0)
        assert m["num_false_positives"] == 1

    def test_id_switch_detected(self):
        frames = [
            ([1], _xy((0, 0)), [11], _xy((0, 0))),
            ([1], _xy((1, 0)), [22], _xy((1, 0))),   # same GT, new track id
        ]
        m = evaluate_tracking(frames, dist_threshold=2.0)
        assert m["num_switches"] == 1

    def test_distance_gate(self):
        # prediction 3 m away must not match at a 2 m gate
        frames = [([1], _xy((0, 0)), [11], _xy((3, 0)))]
        m = evaluate_tracking(frames, dist_threshold=2.0)
        assert m["num_misses"] == 1
        assert m["num_false_positives"] == 1


def test_format_summary_renders_all_metrics():
    frames = [([1], _xy((0, 0)), [11], _xy((0, 0)))]
    text = format_summary(evaluate_tracking(frames))
    assert "MOTA" in text and "IDF1" in text and "ID switches" in text
