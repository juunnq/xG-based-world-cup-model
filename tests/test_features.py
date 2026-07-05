"""Geometry unit tests against hand-computed values."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import freeze_frame_features, shot_angle, shot_distance


def test_distance_penalty_spot():
    # penalty spot is 12 yards from the goal centre
    assert shot_distance(108, 40) == pytest.approx(12.0)


def test_distance_corner():
    assert shot_distance(120, 0) == pytest.approx(40.0)


def test_angle_penalty_spot():
    # symmetric: 2 * atan(half goal width / distance) = 2 * atan(4/12)
    assert shot_angle(108, 40) == pytest.approx(2 * np.arctan(4 / 12))


def test_angle_on_goal_line_between_posts():
    # standing between the posts, the goal mouth spans ~pi radians
    assert shot_angle(119.99, 40) == pytest.approx(np.pi, abs=0.02)


def test_angle_on_extended_goal_line_is_zero():
    # from the extended goal line the goal mouth is invisible
    assert shot_angle(120, 50) == pytest.approx(0.0, abs=1e-6)


def test_angle_decreases_with_distance():
    a = shot_angle(np.array([108, 96, 84]), np.array([40, 40, 40]))
    assert a[0] > a[1] > a[2]


def _ff(players):
    return json.dumps(players)


def test_freeze_frame_counts_defenders_in_cone():
    # shot from the penalty spot; one opponent directly in the cone,
    # one far outside it, plus the GK on the line
    ff = _ff(
        [
            {"location": [114, 40], "teammate": False, "position": {"name": "Center Back"}},
            {"location": [90, 10], "teammate": False, "position": {"name": "Left Back"}},
            {"location": [119, 40], "teammate": False, "position": {"name": "Goalkeeper"}},
            {"location": [110, 38], "teammate": True, "position": {"name": "Center Forward"}},
        ]
    )
    f = freeze_frame_features(ff, 108, 40)
    assert f["defenders_in_cone"] == 1.0
    assert f["dist_nearest_opp"] == pytest.approx(6.0)
    assert f["gk_dist_to_line"] == pytest.approx(1.0)
    assert f["gk_in_cone"] == 1.0
    assert f["gk_lateral_offset"] == pytest.approx(0.0)
    assert f["ff_missing"] == 0.0


def test_freeze_frame_gk_lateral_offset():
    # straight-on shot from (100, 40); GK at (110, 43) is 3 yards off the line
    ff = _ff([{"location": [110, 43], "teammate": False, "position": {"name": "Goalkeeper"}}])
    f = freeze_frame_features(ff, 100, 40)
    assert f["gk_lateral_offset"] == pytest.approx(3.0)


def test_freeze_frame_missing():
    f = freeze_frame_features(None, 108, 40)
    assert f["ff_missing"] == 1.0
    assert np.isnan(f["defenders_in_cone"])
