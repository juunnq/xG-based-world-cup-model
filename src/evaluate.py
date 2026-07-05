"""Metrics with cluster-bootstrap CIs, reliability curves, Brier decomposition,
and cross-validated isotonic recalibration."""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import config

METRICS = {
    "log_loss": log_loss,
    "brier": brier_score_loss,
    "auc": roc_auc_score,
}


def bootstrap_metric_ci(df, pred_col, metric_fn, n_boot=None, seed=None):
    """Point estimate + 95% CI, bootstrapping over matches (cluster bootstrap).

    Resampling whole matches respects the within-match correlation of shots.
    """
    n_boot = n_boot or config.N_BOOTSTRAP
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    y = df["is_goal"].to_numpy(int)
    p = df[pred_col].to_numpy(float)
    point = metric_fn(y, p)

    match_ids = df["match_id"].to_numpy()
    unique_ids = np.unique(match_ids)
    idx_by_match = {m: np.flatnonzero(match_ids == m) for m in unique_ids}
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        idx = np.concatenate([idx_by_match[m] for m in sample])
        if y[idx].min() == y[idx].max():  # degenerate resample (no goals)
            continue
        stats.append(metric_fn(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, lo, hi


def metrics_table(df, pred_cols):
    """One row per model, columns metric / ci_lo / ci_hi."""
    rows = []
    for col in pred_cols:
        row = {"model": col.removeprefix("p_")}
        for mname, fn in METRICS.items():
            point, lo, hi = bootstrap_metric_ci(df, col, fn)
            row[mname] = point
            row[f"{mname}_lo"], row[f"{mname}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_table(y, p, n_bins=10):
    """Quantile-binned reliability curve with Wilson 95% intervals."""
    df = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(n=("y", "size"), p_mean=("p", "mean"), y_rate=("y", "mean"))
    z = 1.96
    n, ph = g["n"], g["y_rate"]
    denom = 1 + z**2 / n
    centre = (ph + z**2 / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / denom
    g["y_lo"], g["y_hi"] = centre - half, centre + half
    return g.reset_index()


def brier_decomposition(y, p, n_bins=10):
    """Murphy decomposition over quantile bins: brier = REL - RES + UNC."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(n=("y", "size"), p_mean=("p", "mean"), y_rate=("y", "mean"))
    n_total = len(df)
    ybar = y.mean()
    rel = float((g["n"] * (g["p_mean"] - g["y_rate"]) ** 2).sum() / n_total)
    res = float((g["n"] * (g["y_rate"] - ybar) ** 2).sum() / n_total)
    unc = float(ybar * (1 - ybar))
    return {
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "brier_decomposed": rel - res + unc,
        "brier_raw": float(np.mean((p - y) ** 2)),
    }


def isotonic_cv(df, pred_col):
    """Leave-one-fold-out isotonic recalibration of OOF predictions.

    For each fold k the isotonic map is fitted on the other folds' OOF
    predictions, so the recalibrated values remain out-of-sample.
    """
    p = df[pred_col].to_numpy(float)
    y = df["is_goal"].to_numpy(int)
    folds = df["fold"].to_numpy(int)
    out = np.full(len(df), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(p[tr], y[tr])
        out[te] = iso.predict(p[te])
    return out
