"""Spatial residual analysis: pitch binning, per-bin standardized residuals,
global spatial chi-square, Moran's I with permutation inference, and subgroup
calibration tests.

Under the null that predicted probabilities p_i are calibrated, the bin sums
obey E[G_b] = sum(p_i) and Var[G_b] = sum(p_i (1 - p_i)) (Poisson-binomial),
giving standardized residuals z_b that are approximately N(0,1) for bins with
enough shots.
"""

import numpy as np
import pandas as pd
from scipy import stats

import config


def bin_shots(df, pred_col):
    """Aggregate shots into GRID_CELL x GRID_CELL pitch bins (x >= GRID_X_MIN).

    Returns one row per non-empty bin with observed goals, expected goals,
    Poisson-binomial variance, standardized residual z, and a `qualifies`
    flag (n >= MIN_SHOTS_PER_BIN) used by the formal tests.
    """
    d = df[df["x"] >= config.GRID_X_MIN].copy()
    cell = config.GRID_CELL
    d["ix"] = np.minimum(((d["x"] - config.GRID_X_MIN) // cell).astype(int),
                         int((config.PITCH_LENGTH - config.GRID_X_MIN) / cell) - 1)
    d["iy"] = np.minimum((d["y"] // cell).astype(int),
                         int(config.PITCH_WIDTH / cell) - 1)
    p = d[pred_col].astype(float)
    d["_var"] = p * (1 - p)
    g = (
        d.groupby(["ix", "iy"])
        .agg(n=("is_goal", "size"), obs=("is_goal", "sum"),
             exp=(pred_col, "sum"), var=("_var", "sum"))
        .reset_index()
    )
    g["z"] = (g["obs"] - g["exp"]) / np.sqrt(g["var"])
    g["qualifies"] = g["n"] >= config.MIN_SHOTS_PER_BIN
    g["x_lo"] = config.GRID_X_MIN + g["ix"] * cell
    g["y_lo"] = g["iy"] * cell
    return g


def spatial_chi_square(bins):
    """Hosmer-Lemeshow-style global test over spatial bins (qualifying only).

    T = sum_b (obs_b - exp_b)^2 / var_b  ~  chi2(B) under calibration.
    Predictions are out-of-fold, so no parameter-count correction is applied.
    """
    q = bins[bins["qualifies"]]
    stat = float((((q["obs"] - q["exp"]) ** 2) / q["var"]).sum())
    dof = int(len(q))
    p = float(stats.chi2.sf(stat, dof))
    return {"chi2": stat, "df": dof, "p_value": p, "n_bins": dof,
            "shots_covered": int(q["n"].sum()), "shots_total": int(bins["n"].sum())}


def queen_weights(bins):
    """Binary queen-contiguity matrix over qualifying grid cells."""
    q = bins[bins["qualifies"]].reset_index(drop=True)
    ix, iy = q["ix"].to_numpy(), q["iy"].to_numpy()
    dx = np.abs(ix[:, None] - ix[None, :])
    dy = np.abs(iy[:, None] - iy[None, :])
    w = ((dx <= 1) & (dy <= 1)).astype(float)
    np.fill_diagonal(w, 0.0)
    return q, w


def morans_i(bins, n_perm=None, seed=None):
    """Moran's I on per-bin z residuals; one-sided permutation p-value for
    positive spatial autocorrelation (miscalibration that clusters)."""
    n_perm = n_perm or config.N_PERMUTATIONS
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    q, w = queen_weights(bins)
    v = q["z"].to_numpy(float)
    v = v - v.mean()
    s0 = w.sum()
    n = len(v)

    def moran(x):
        return (n / s0) * (x @ w @ x) / (x @ x)

    i_obs = float(moran(v))
    perm = np.empty(n_perm)
    for j in range(n_perm):
        perm[j] = moran(rng.permutation(v))
    p = float((1 + (perm >= i_obs).sum()) / (n_perm + 1))
    return {"morans_i": i_obs, "p_value": p, "n_bins": n,
            "expected_i": -1.0 / (n - 1), "n_permutations": n_perm}


def group_calibration(df, pred_col, mask, name):
    """Observed-vs-expected z-test for one shot subgroup."""
    d = df[mask]
    p = d[pred_col].astype(float)
    obs, exp = int(d["is_goal"].sum()), float(p.sum())
    var = float((p * (1 - p)).sum())
    z = (obs - exp) / np.sqrt(var)
    return {
        "group": name,
        "n_shots": len(d),
        "observed_goals": obs,
        "expected_goals": exp,
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "goals_per_100_shots_gap": float((obs - exp) / len(d) * 100) if len(d) else np.nan,
    }


def subgroup_tests(df, pred_col):
    """Calibration z-tests for the pre-registered subgroups."""
    tight = df["angle"] < config.TIGHT_ANGLE_RAD
    set_piece = (df["is_free_kick"] == 1) | (df["from_corner"] == 1)
    groups = [
        (df["header"] == 1, "headers"),
        (df["header"] == 0, "non-headers"),
        (tight, f"tight angle (<{np.degrees(config.TIGHT_ANGLE_RAD):.0f} deg)"),
        (~tight, "open angle"),
        (set_piece, "set piece (FK/corner phase)"),
        (~set_piece, "open play"),
    ]
    groups += [(df["competition"] == c, c) for c in sorted(df["competition"].unique())]
    return pd.DataFrame([group_calibration(df, pred_col, m, n) for m, n in groups])
