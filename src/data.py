"""Download StatsBomb open-data events and build a tidy one-row-per-shot table.

Raw per-match shot extracts are cached in data/raw/ so re-runs are offline.
"""

import json
import warnings

import pandas as pd

import config

warnings.filterwarnings("ignore", module="statsbombpy")


def _col(df, name, default=None):
    """Column accessor tolerant of statsbombpy's variable flattening."""
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def extract_match_shots(match_id: int, label: str) -> pd.DataFrame:
    """Pull one match's events and return its shots with assist context."""
    from statsbombpy import sb

    events = sb.events(match_id=match_id)
    shots = events[events["type"] == "Shot"].copy()
    if shots.empty:
        return pd.DataFrame()

    loc = shots["location"].apply(pd.Series)
    shots["x"], shots["y"] = loc[0].astype(float), loc[1].astype(float)

    # Assist context from the key pass event, if any
    passes = events[events["type"] == "Pass"].set_index("id")
    key_ids = _col(shots, "shot_key_pass_id")

    def pass_flag(col, technique=None):
        def flag(kid):
            if pd.isna(kid) or kid not in passes.index:
                return False
            p = passes.loc[kid]
            if technique is not None and p.get("pass_technique") == technique:
                return True
            return bool(p.get(col)) if col in passes.columns and pd.notna(p.get(col)) else False
        return key_ids.apply(flag)

    out = pd.DataFrame(
        {
            "event_id": shots["id"],
            "match_id": match_id,
            "competition": label,
            "period": shots["period"],
            "minute": shots["minute"],
            "x": shots["x"],
            "y": shots["y"],
            "is_goal": (shots["shot_outcome"] == "Goal").astype(int),
            "statsbomb_xg": shots["shot_statsbomb_xg"].astype(float),
            "body_part": _col(shots, "shot_body_part", "Unknown"),
            "shot_type": _col(shots, "shot_type", "Open Play"),
            "technique": _col(shots, "shot_technique", "Normal"),
            "first_time": _col(shots, "shot_first_time", False).eq(True),
            "under_pressure": _col(shots, "under_pressure", False).eq(True),
            "play_pattern": _col(shots, "play_pattern", "Regular Play"),
            "assist_cross": pass_flag("pass_cross"),
            "assist_cutback": pass_flag("pass_cut_back"),
            "assist_through_ball": pass_flag("pass_through_ball", technique="Through Ball"),
            "freeze_frame": _col(shots, "shot_freeze_frame").apply(
                lambda ff: json.dumps(ff) if isinstance(ff, list) else None
            ),
        }
    )
    return out.reset_index(drop=True)


def build_dataset(force: bool = False) -> pd.DataFrame:
    """Build (or load) the pooled shot table across all configured tournaments."""
    from statsbombpy import sb

    if config.SHOTS_PARQUET.exists() and not force:
        return pd.read_parquet(config.SHOTS_PARQUET)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for comp_id, season_id, label in config.TOURNAMENTS:
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        jobs += [(int(mid), label) for mid in matches["match_id"]]
        print(f"{label}: {len(matches)} matches")

    # Serial on purpose: threaded fetching crashed natively (0xC0000005) on
    # Windows / Python 3.13. One-time cost; per-match caching makes it resumable.
    frames = []
    for i, (mid, lab) in enumerate(jobs, 1):
        cache = config.RAW_DIR / f"{mid}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
        else:
            df = extract_match_shots(mid, lab)
            df.to_parquet(cache, index=False)
            frames.append(df)
        if i % 25 == 0:
            print(f"  {i}/{len(jobs)} matches processed")

    shots = pd.concat(frames, ignore_index=True)
    n_pens = int((shots["shot_type"] == "Penalty").sum())
    shots = shots[shots["shot_type"] != "Penalty"].reset_index(drop=True)

    assert shots[["x", "y"]].notna().all().all(), "missing shot coordinates"
    shots.to_parquet(config.SHOTS_PARQUET, index=False)

    print(f"\nExcluded {n_pens} penalties.")
    summary = shots.groupby("competition").agg(
        shots=("is_goal", "size"), goals=("is_goal", "sum"), goal_rate=("is_goal", "mean")
    )
    print(summary.round(4))
    print(f"Total: {len(shots)} shots, goal rate {shots['is_goal'].mean():.4f}")
    ff_missing = shots["freeze_frame"].isna().mean()
    print(f"Freeze-frame missing rate: {ff_missing:.4f}")
    return shots


if __name__ == "__main__":
    build_dataset()
