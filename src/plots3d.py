"""3D figures: the xG landscape and the residual skyline.

Both draw the pitch markings projected on the z=0 floor so the geometry
reads instantly. Sequential (Blues) for magnitude, diverging (RdBu) for
signed residuals — both ColorBrewer, CVD-safe.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource, TwoSlopeNorm

import config

LINE = "#5a6570"  # recessive pitch lines
INK = "#2b3238"


def _pitch_floor(ax, z=0.0):
    """Attacking-half StatsBomb pitch markings on the z=0 plane."""
    def line(xs, ys):
        ax.plot(xs, ys, zs=z, color=LINE, lw=1.0, alpha=0.9, zorder=1)

    line([60, 120, 120, 60, 60], [0, 0, 80, 80, 0])            # half outline
    line([102, 102, 120], [18, 62, 62]); line([102, 120], [18, 18])  # box
    line([114, 114, 120], [30, 50, 50]); line([114, 120], [30, 30])  # six-yard
    line([120, 120], [36, 44])                                  # goal mouth
    ax.scatter([108], [40], [z], color=LINE, s=6)               # penalty spot
    t = np.linspace(np.radians(127), np.radians(233), 40)       # D arc
    line(108 + 10 * np.cos(t), 40 + 10 * np.sin(t))


def _style_3d(ax):
    ax.set_proj_type("persp", focal_length=0.28)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
        pane.line.set_color("#c9ced4")
    ax.grid(False)
    ax.set_xlim(60, 120)
    ax.set_ylim(0, 80)
    ax.set_box_aspect((1.5, 2.0, 0.75))
    ax.tick_params(colors="#6b7480", labelsize=8, pad=-2)


def xg_landscape(model, ref_row, feature_cols, path):
    """Smooth surface of P(goal) by shot location for a reference open-play
    foot shot (all non-geometric features held at dataset-typical values)."""
    from src.features import shot_angle, shot_distance

    xs = np.linspace(60, 119.5, 120)
    ys = np.linspace(0.5, 79.5, 160)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    grid = np.tile(ref_row[feature_cols].to_numpy(float), (X.size, 1))
    di, ai = feature_cols.index("distance"), feature_cols.index("angle")
    grid[:, di] = shot_distance(X.ravel(), Y.ravel())
    grid[:, ai] = shot_angle(X.ravel(), Y.ravel())
    Z = model.predict_proba(grid)[:, 1].reshape(X.shape)

    fig = plt.figure(figsize=(10, 5.8))
    ax = fig.add_axes((-0.06, -0.12, 0.94, 1.24), projection="3d")
    ls = LightSource(azdeg=315, altdeg=50)
    rgb = ls.shade(Z, cmap=cm.Blues, vert_exag=140, blend_mode="soft",
                   vmin=0, vmax=Z.max())
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, zorder=5)
    _pitch_floor(ax)
    _style_3d(ax)
    ax.set_zlim(0, Z.max() * 1.02)
    ax.view_init(elev=26, azim=-58)
    ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_zticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax.set_xticks([60, 80, 100, 120])
    ax.set_yticks([0, 40, 80])
    fig.text(0.06, 0.94, "The xG landscape", color=INK, fontsize=15,
             fontweight="bold")
    fig.text(0.06, 0.895, "P(goal) for an open-play foot shot, by location "
             "(full logistic model, other features at typical values)",
             color="#6b7480", fontsize=10)

    m = cm.ScalarMappable(cmap=cm.Blues)
    m.set_array(Z)
    cb = fig.colorbar(m, ax=ax, shrink=0.42, pad=0.0, fraction=0.04,
                      format=lambda v, _: f"{v:.0%}")
    cb.set_label("scoring probability", color=INK)
    cb.outline.set_visible(False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def residual_skyline(bins_by_model, path, vmax=None):
    """Side-by-side 3D bars of per-bin standardized residuals z.

    Bars rise (red) where a model under-predicts goals and sink (blue)
    where it over-predicts; only bins with enough shots are drawn.
    """
    all_z = np.concatenate([b.loc[b["qualifies"], "z"].to_numpy()
                            for b in bins_by_model.values()])
    vmax = vmax or float(np.ceil(np.abs(all_z).max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cell = config.GRID_CELL

    def annotate(ax, row, text):
        ax.text(row["x_lo"] + cell / 2, row["y_lo"] - 1, max(row["z"], 0) + 1.1,
                text, color=INK, fontsize=8.5, ha="center", zorder=20,
                bbox=dict(fc="white", ec="#c9ced4", lw=0.5, alpha=0.85,
                          boxstyle="round,pad=0.3"))

    fig = plt.figure(figsize=(13, 6.0))
    for i, (label, bins) in enumerate(bins_by_model.items(), 1):
        ax = fig.add_axes((-0.04 + (i - 1) * 0.45, -0.12, 0.52, 1.12),
                          projection="3d")
        q = bins[bins["qualifies"]]
        z = q["z"].to_numpy(float)
        colors = cm.RdBu_r(norm(z))
        ax.bar3d(q["x_lo"], q["y_lo"], np.minimum(z, 0),
                 cell * 0.82, cell * 0.82, np.abs(z),
                 color=colors, edgecolor="white", linewidth=0.4,
                 shade=True, zsort="max")
        top = q.loc[q["z"].idxmax()]
        annotate(ax, top, f"{int(top['obs'])} goals vs {top['exp']:.1f} expected"
                          f"\n(z = {top['z']:.1f})")
        byline = q[(q["x_lo"] >= 110) & (q["z"] > 2) & (q.index != top.name)]
        if not byline.empty:
            b = byline.loc[byline["z"].idxmax()]
            annotate(ax, b, f"byline zone\n{int(b['obs'])} goals vs {b['exp']:.1f}"
                            f" (z = {b['z']:.1f})")
        _pitch_floor(ax)
        _style_3d(ax)
        ax.set_zlim(-vmax, vmax)
        ax.view_init(elev=30, azim=-55)
        ax.set_xticks([60, 90, 120])
        ax.set_yticks([0, 40, 80])
        ax.set_zticks([])  # height is redundant with the shared colorbar
        ax.set_title(label, color=INK, fontsize=12, y=0.92)

    m = cm.ScalarMappable(cmap=cm.RdBu_r, norm=norm)
    m.set_array([])
    cb = fig.colorbar(m, ax=fig.axes, shrink=0.5, pad=0.01, fraction=0.03)
    cb.set_label("standardized residual z  (red = scores more than xG says)",
                 color=INK, fontsize=9)
    cb.outline.set_visible(False)
    fig.text(0.05, 0.93, "Where each model is wrong — the residual skyline",
             color=INK, fontsize=15, fontweight="bold")
    fig.text(0.05, 0.885, f"5-yard bins with ≥ {config.MIN_SHOTS_PER_BIN} shots; "
             "bars rise where real goals beat the model's expectation",
             color="#6b7480", fontsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)
