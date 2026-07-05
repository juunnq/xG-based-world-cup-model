"""2026 World Cup: match-level calibration audit of the provider xG feed.

No public shot-level data exists for 2026 yet, so this analysis works at the
granularity the data supports: per-team-per-match goals vs provider xG from
the CC0 dataset at github.com/mominullptr/FIFA-World-Cup-2026-Dataset.
Under the null that goals for a team-match are Poisson with mean equal to its
xG, sums and dispersion statistics below have known distributions.
"""

import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

import config


def fetch_wc2026(refresh=False) -> pd.DataFrame:
    """Download (or load cached) matches_detailed.csv; completed matches only,
    reshaped to one row per team-match."""
    config.WC2026_DIR.mkdir(parents=True, exist_ok=True)
    path = config.WC2026_DIR / "matches_detailed.csv"
    if refresh or not path.exists():
        urllib.request.urlretrieve(f"{config.WC2026_REPO_RAW}/matches_detailed.csv", path)
    m = pd.read_csv(path)
    m = m[m["status"] == "Completed"].copy()

    home = m[["match_id", "date", "stage_name", "home_team_name", "home_score", "home_xg"]]
    home.columns = ["match_id", "date", "stage", "team", "goals", "xg"]
    away = m[["match_id", "date", "stage_name", "away_team_name", "away_score", "away_xg"]]
    away.columns = ["match_id", "date", "stage", "team", "goals", "xg"]
    tm = pd.concat([home, away], ignore_index=True).dropna(subset=["goals", "xg"])
    tm["goals"] = tm["goals"].astype(int)
    tm["xg"] = tm["xg"].astype(float)
    return tm


def calibration_in_the_large(tm: pd.DataFrame) -> dict:
    """Total goals vs total xG, z-test with Poisson variance (= total xG)."""
    g, x = int(tm["goals"].sum()), float(tm["xg"].sum())
    z = (g - x) / np.sqrt(x)
    return {
        "team_matches": len(tm),
        "matches": tm["match_id"].nunique(),
        "total_goals": g,
        "total_xg": x,
        "goals_minus_xg": g - x,
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def overdispersion_test(tm: pd.DataFrame) -> dict:
    """Poisson dispersion test on team-match residuals:
    D = sum (g_i - x_i)^2 / x_i ~ chi2(n) under the Poisson null."""
    d = float((((tm["goals"] - tm["xg"]) ** 2) / tm["xg"]).sum())
    n = len(tm)
    return {
        "dispersion_stat": d,
        "df": n,
        "dispersion_ratio": d / n,
        "p_value": float(stats.chi2.sf(d, n)),
    }


def reliability_bins(tm: pd.DataFrame, n_bins=8) -> pd.DataFrame:
    """Binned team-match xG vs realized mean goals, Poisson CI on the mean."""
    d = tm.copy()
    d["bin"] = pd.qcut(d["xg"], n_bins, labels=False, duplicates="drop")
    g = d.groupby("bin").agg(n=("goals", "size"), xg_mean=("xg", "mean"),
                             goals_mean=("goals", "mean"), goals_sum=("goals", "sum"))
    # 95% CI for the mean via normal approx to the Poisson sum
    g["ci_half"] = 1.96 * np.sqrt(g["goals_sum"]) / g["n"]
    return g.reset_index()


def team_table(tm: pd.DataFrame) -> pd.DataFrame:
    """Cumulative goals - xG per team with two-sided Poisson z-test."""
    g = tm.groupby("team").agg(matches=("match_id", "nunique"),
                               goals=("goals", "sum"), xg=("xg", "sum"))
    g["goals_minus_xg"] = g["goals"] - g["xg"]
    g["z"] = g["goals_minus_xg"] / np.sqrt(g["xg"])
    g["p_value"] = 2 * stats.norm.sf(np.abs(g["z"]))
    g["flag"] = np.where(g["p_value"] < 0.05,
                         np.where(g["z"] > 0, "overperforming", "underperforming"), "")
    return g.sort_values("goals_minus_xg", ascending=False).reset_index()


def stage_table(tm: pd.DataFrame) -> pd.DataFrame:
    g = tm.groupby("stage").agg(team_matches=("goals", "size"),
                                goals_per_tm=("goals", "mean"), xg_per_tm=("xg", "mean"))
    g["goals_minus_xg_per_tm"] = g["goals_per_tm"] - g["xg_per_tm"]
    return g.reset_index()
