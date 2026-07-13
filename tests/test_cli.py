import argparse

from perception.cli import add_dataset_args, add_tracker_args, build_tracker


def _parse(argv):
    p = argparse.ArgumentParser()
    add_dataset_args(p)
    add_tracker_args(p)
    return p.parse_args(argv)


def test_defaults_match_tuned_tracker_settings():
    args = _parse([])
    tracker = build_tracker(args)
    assert tracker.score_threshold == 0.5
    assert tracker.min_hits == 2
    assert tracker.max_missed == 3
    assert tracker.dist_threshold == 4.5


def test_overrides_reach_the_tracker():
    args = _parse(["--score-threshold", "-1.0", "--gate", "3.0", "--min-hits", "1"])
    tracker = build_tracker(args)
    assert tracker.score_threshold == -1.0
    assert tracker.dist_threshold == 3.0
    assert tracker.min_hits == 1


def test_dataset_defaults():
    args = _parse([])
    assert args.detector == "pvrcnn"
    assert args.seq == 8
