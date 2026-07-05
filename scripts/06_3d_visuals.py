"""Step 6: 3D figures — the xG landscape and the residual skyline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.features import ALL_FEATURES
from src.models import _logistic
from src.plots3d import residual_skyline, xg_landscape
from src.residuals import bin_shots

if __name__ == "__main__":
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(config.DATA_DIR / "oof.parquet")

    # Reference shot: every non-geometric feature at its dataset-typical value
    # (binary flags -> mode, counts/distances -> median), geometry varies.
    ref = oof[ALL_FEATURES].median()
    for flag in ["header", "body_other", "is_free_kick", "from_corner", "first_time",
                 "volley", "assist_cross", "assist_cutback", "assist_through_ball",
                 "gk_in_cone", "ff_missing", "under_pressure"]:
        ref[flag] = oof[flag].mode()[0]

    model = _logistic()
    model.fit(oof[ALL_FEATURES].to_numpy(float), oof["is_goal"].to_numpy(int))
    xg_landscape(model, ref, ALL_FEATURES, config.FIGURES_DIR / "xg_landscape_3d.png")
    print("Saved xg_landscape_3d.png")

    bins = {
        "Our model (XGBoost)": bin_shots(oof, "p_xgboost"),
        "StatsBomb xG": bin_shots(oof, "statsbomb_xg"),
    }
    residual_skyline(bins, config.FIGURES_DIR / "residual_skyline_3d.png")
    print("Saved residual_skyline_3d.png")
