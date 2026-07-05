"""Statistical machinery tests on synthetic data with known answers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.evaluate import brier_decomposition
from src.residuals import bin_shots, morans_i, spatial_chi_square


def _synthetic_bins(z_values, n=100):
    """Fabricate a qualifying bin table on a grid from a 2-D z array."""
    rows = []
    for ix in range(z_values.shape[0]):
        for iy in range(z_values.shape[1]):
            rows.append({"ix": ix, "iy": iy, "n": n, "obs": 10, "exp": 10.0,
                         "var": 9.0, "z": z_values[ix, iy], "qualifies": True})
    return pd.DataFrame(rows)


def test_morans_i_stripes_are_negative():
    # alternating column stripes: most queen neighbours have opposite sign
    # (a checkerboard would NOT work here — its diagonal neighbours match)
    z = np.tile((np.arange(6) % 2 * 2.0 - 1.0), (6, 1))
    res = morans_i(_synthetic_bins(z), n_perm=999, seed=0)
    assert res["morans_i"] < -0.3
    assert res["p_value"] > 0.5  # one-sided test for POSITIVE autocorrelation


def test_morans_i_smooth_gradient_is_positive():
    z = np.indices((6, 6)).sum(axis=0).astype(float)  # smooth ramp
    res = morans_i(_synthetic_bins(z), n_perm=999, seed=0)
    assert res["morans_i"] > 0.3
    assert res["p_value"] < 0.01


def test_spatial_chi_square_calibrated_data_not_rejected():
    # simulate perfectly calibrated shots -> p-value should not be tiny
    rng = np.random.default_rng(7)
    n = 20000
    df = pd.DataFrame({
        "x": rng.uniform(config.GRID_X_MIN, 120, n),
        "y": rng.uniform(0, 80, n),
        "p": rng.uniform(0.02, 0.4, n),
    })
    df["is_goal"] = (rng.uniform(size=n) < df["p"]).astype(int)
    bins = bin_shots(df, "p")
    res = spatial_chi_square(bins)
    assert res["p_value"] > 0.001


def test_spatial_chi_square_detects_miscalibration():
    # same shots but predictions halved -> should reject decisively
    rng = np.random.default_rng(7)
    n = 20000
    df = pd.DataFrame({
        "x": rng.uniform(config.GRID_X_MIN, 120, n),
        "y": rng.uniform(0, 80, n),
        "p": rng.uniform(0.02, 0.4, n),
    })
    df["is_goal"] = (rng.uniform(size=n) < df["p"]).astype(int)
    df["p"] = df["p"] / 2
    res = spatial_chi_square(bin_shots(df, "p"))
    assert res["p_value"] < 1e-6


def test_brier_decomposition_identity():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.5, 5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    d = brier_decomposition(y, p)
    # REL - RES + UNC equals the raw Brier score up to within-bin variance
    assert d["brier_decomposed"] == pytest.approx(d["brier_raw"], abs=0.01)
    assert d["reliability"] < 0.005  # calibrated by construction


def test_bin_shots_totals_preserved():
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({
        "x": rng.uniform(config.GRID_X_MIN, 120, n),
        "y": rng.uniform(0, 80, n),
        "p": rng.uniform(0.02, 0.4, n),
    })
    df["is_goal"] = (rng.uniform(size=n) < df["p"]).astype(int)
    bins = bin_shots(df, "p")
    assert bins["n"].sum() == n
    assert bins["obs"].sum() == df["is_goal"].sum()
    assert bins["exp"].sum() == pytest.approx(df["p"].sum())
