"""Shot feature engineering: geometry and freeze-frame (defender/GK) features."""

import json

import numpy as np
import pandas as pd

import config

BASELINE_FEATURES = ["distance", "angle", "header"]

ALL_FEATURES = [
    "distance",
    "angle",
    "header",
    "body_other",
    "is_free_kick",
    "from_corner",
    "first_time",
    "under_pressure",
    "volley",
    "assist_cross",
    "assist_cutback",
    "assist_through_ball",
    "defenders_in_cone",
    "dist_nearest_opp",
    "gk_dist_to_line",
    "gk_lateral_offset",
    "gk_in_cone",
    "ff_missing",
]

_EPS = 1e-9


def shot_distance(x, y):
    """Euclidean distance to the goal centre (120, 40)."""
    gx, gy = config.GOAL_CENTER
    return np.hypot(gx - np.asarray(x, float), gy - np.asarray(y, float))


def shot_angle(x, y):
    """Visible goal-mouth angle (radians) between the two posts.

    Law of cosines with a = dist to left post, b = dist to right post,
    c = goal width. Zero when the goal mouth is not visible (on the
    extended goal line); approaches pi on the line between the posts.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    lx, ly = config.POST_LEFT
    rx, ry = config.POST_RIGHT
    a = np.hypot(lx - x, ly - y)
    b = np.hypot(rx - x, ry - y)
    c = ry - ly
    cos = (a**2 + b**2 - c**2) / np.maximum(2 * a * b, _EPS)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _in_cone(px, py, sx, sy):
    """Is point (px,py) inside the triangle (shot, left post, right post)?"""
    lx, ly = config.POST_LEFT
    rx, ry = config.POST_RIGHT

    def sign(ax, ay, bx, by, cx, cy):
        return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)

    d1 = sign(px, py, sx, sy, lx, ly)
    d2 = sign(px, py, lx, ly, rx, ry)
    d3 = sign(px, py, rx, ry, sx, sy)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def freeze_frame_features(ff_json, sx, sy):
    """Defender and goalkeeper features from one shot's freeze frame.

    Returns dict with NaNs when the freeze frame (or the GK) is absent;
    imputation happens in build_features.
    """
    out = {
        "defenders_in_cone": np.nan,
        "dist_nearest_opp": np.nan,
        "gk_dist_to_line": np.nan,
        "gk_lateral_offset": np.nan,
        "gk_in_cone": np.nan,
        "ff_missing": 1.0,
    }
    if not ff_json:
        return out
    players = json.loads(ff_json)
    opponents = [p for p in players if not p["teammate"]]
    gk = [p for p in opponents if p.get("position", {}).get("name") == "Goalkeeper"]
    outfield = [p for p in opponents if p not in gk]

    out["ff_missing"] = 0.0
    out["defenders_in_cone"] = float(
        sum(_in_cone(p["location"][0], p["location"][1], sx, sy) for p in outfield)
    )
    if outfield:
        out["dist_nearest_opp"] = min(
            np.hypot(p["location"][0] - sx, p["location"][1] - sy) for p in outfield
        )
    if gk:
        gx, gy = gk[0]["location"]
        out["gk_dist_to_line"] = config.PITCH_LENGTH - gx
        out["gk_in_cone"] = float(_in_cone(gx, gy, sx, sy))
        # perpendicular distance of GK from the shot -> goal-centre line
        cx, cy = config.GOAL_CENTER
        vx, vy = cx - sx, cy - sy
        norm = max(np.hypot(vx, vy), _EPS)
        out["gk_lateral_offset"] = abs(vx * (gy - sy) - vy * (gx - sx)) / norm
    return out


def build_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix + target + metadata, one row per shot."""
    df = pd.DataFrame(index=shots.index)
    df["distance"] = shot_distance(shots["x"], shots["y"])
    df["angle"] = shot_angle(shots["x"], shots["y"])
    df["header"] = (shots["body_part"] == "Head").astype(float)
    df["body_other"] = (~shots["body_part"].isin(["Left Foot", "Right Foot", "Head"])).astype(float)
    df["is_free_kick"] = (shots["shot_type"] == "Free Kick").astype(float)
    df["from_corner"] = (shots["play_pattern"] == "From Corner").astype(float)
    df["first_time"] = shots["first_time"].astype(float)
    df["under_pressure"] = shots["under_pressure"].astype(float)
    df["volley"] = shots["technique"].isin(["Volley", "Half Volley"]).astype(float)
    for c in ["assist_cross", "assist_cutback", "assist_through_ball"]:
        df[c] = shots[c].astype(float)

    ff = pd.DataFrame(
        [
            freeze_frame_features(f, x, y)
            for f, x, y in zip(shots["freeze_frame"], shots["x"], shots["y"])
        ],
        index=shots.index,
    )
    # median-impute freeze-frame gaps; ff_missing keeps the signal
    for c in ["defenders_in_cone", "dist_nearest_opp", "gk_dist_to_line",
              "gk_lateral_offset", "gk_in_cone"]:
        ff[c] = ff[c].fillna(ff[c].median())
    df = pd.concat([df, ff], axis=1)

    df["is_goal"] = shots["is_goal"].astype(int)
    for c in ["match_id", "competition", "x", "y", "statsbomb_xg"]:
        df[c] = shots[c]
    return df
