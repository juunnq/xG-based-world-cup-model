"""Figures: reliability curves and pitch residual heatmaps."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mplsoccer import Pitch

import config


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(curves, path_name, title="Reliability curves"):
    """curves: dict label -> reliability_table DataFrame."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for label, tab in curves.items():
        ax.errorbar(
            tab["p_mean"], tab["y_rate"],
            yerr=[tab["y_rate"] - tab["y_lo"], tab["y_hi"] - tab["y_rate"]],
            marker="o", ms=4, lw=1.2, capsize=2, label=label,
        )
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed goal rate")
    ax.set_title(title)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, path_name)


def plot_pitch_zmap(bins, path_name, title):
    """Half-pitch heatmap of per-bin standardized residuals z.

    Bins failing the min-shot threshold are hatched grey; red = model
    under-predicts (more goals than expected), blue = over-predicts.
    """
    cell = config.GRID_CELL
    nx = int((config.PITCH_LENGTH - config.GRID_X_MIN) / cell)
    ny = int(config.PITCH_WIDTH / cell)
    z = np.full((nx, ny), np.nan)
    qual = np.zeros((nx, ny), bool)
    for _, r in bins.iterrows():
        z[int(r["ix"]), int(r["iy"])] = r["z"]
        qual[int(r["ix"]), int(r["iy"])] = r["qualifies"]

    pitch = Pitch(pitch_type="statsbomb", half=True, line_color="black", linewidth=1)
    fig, ax = pitch.draw(figsize=(8, 6))
    xe = np.arange(config.GRID_X_MIN, config.PITCH_LENGTH + cell, cell)
    ye = np.arange(0, config.PITCH_WIDTH + cell, cell)
    vmax = max(2.5, np.nanmax(np.abs(z[qual])) if qual.any() else 2.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    z_q = np.where(qual, z, np.nan)
    pc = ax.pcolormesh(*np.meshgrid(xe, ye, indexing="ij"), z_q,
                       cmap="RdBu_r", norm=norm, alpha=0.85, zorder=0.9)
    z_nq = np.where(~qual & ~np.isnan(z), 0.0, np.nan)
    ax.pcolormesh(*np.meshgrid(xe, ye, indexing="ij"), z_nq,
                  cmap="Greys", vmin=-1, vmax=8, alpha=0.25, zorder=0.8)

    cbar = fig.colorbar(pc, ax=ax, shrink=0.8)
    cbar.set_label("standardized residual z (obs − exp goals)")
    ax.set_title(f"{title}\n(grey: < {config.MIN_SHOTS_PER_BIN} shots, excluded from tests)")
    save(fig, path_name)


def plot_shot_density(bins, path_name):
    """Companion map: shots per bin, so readers can judge support."""
    cell = config.GRID_CELL
    nx = int((config.PITCH_LENGTH - config.GRID_X_MIN) / cell)
    ny = int(config.PITCH_WIDTH / cell)
    n = np.full((nx, ny), np.nan)
    for _, r in bins.iterrows():
        n[int(r["ix"]), int(r["iy"])] = r["n"]
    pitch = Pitch(pitch_type="statsbomb", half=True, line_color="black", linewidth=1)
    fig, ax = pitch.draw(figsize=(8, 6))
    xe = np.arange(config.GRID_X_MIN, config.PITCH_LENGTH + cell, cell)
    ye = np.arange(0, config.PITCH_WIDTH + cell, cell)
    pc = ax.pcolormesh(*np.meshgrid(xe, ye, indexing="ij"), n,
                       cmap="viridis", alpha=0.85, zorder=0.9)
    cbar = fig.colorbar(pc, ax=ax, shrink=0.8)
    cbar.set_label("shots per bin")
    ax.set_title("Shot density (all four World Cups, penalties excluded)")
    save(fig, path_name)
