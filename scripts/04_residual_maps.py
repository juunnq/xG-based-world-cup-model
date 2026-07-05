"""Step 4: the core contribution — spatial residual maps and structure tests,
run identically on our XGBoost model and StatsBomb's own xG."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.plots import plot_pitch_zmap, plot_shot_density
from src.residuals import bin_shots, morans_i, spatial_chi_square, subgroup_tests

TARGETS = {
    "p_xgboost": "XGBoost xG",
    "statsbomb_xg": "StatsBomb xG",
}

if __name__ == "__main__":
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(config.DATA_DIR / "oof.parquet")

    spatial_rows, subgroup_frames = [], []
    for col, label in TARGETS.items():
        bins = bin_shots(oof, col)
        slug = col.removeprefix("p_")
        bins.to_csv(config.TABLES_DIR / f"residual_bins_{slug}.csv", index=False)
        plot_pitch_zmap(bins, f"residual_zmap_{slug}",
                        f"Where {label} is over/under-confident")

        chi = spatial_chi_square(bins)
        moran = morans_i(bins)
        spatial_rows.append({"model": label, **{f"chi2_{k}": v for k, v in chi.items()},
                             **{f"moran_{k}": v for k, v in moran.items()}})
        print(f"{label}: chi2={chi['chi2']:.1f} (df={chi['df']}, p={chi['p_value']:.2e}); "
              f"Moran's I={moran['morans_i']:.3f} (p={moran['p_value']:.4f})")

        sub = subgroup_tests(oof, col)
        sub.insert(0, "model", label)
        subgroup_frames.append(sub)

    pd.DataFrame(spatial_rows).to_csv(config.TABLES_DIR / "spatial_tests.csv", index=False)
    pd.concat(subgroup_frames).to_csv(config.TABLES_DIR / "subgroup_tests.csv", index=False)
    plot_shot_density(bin_shots(oof, "p_xgboost"), "shot_density")
    print("Saved residual bins, z-maps, spatial tests, subgroup tests.")
