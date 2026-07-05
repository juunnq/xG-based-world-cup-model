"""Step 3: reliability curves, Brier decomposition, CV isotonic recalibration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import config
from src.evaluate import brier_decomposition, isotonic_cv, reliability_table
from src.plots import plot_reliability

MODEL_LABELS = {
    "p_logit_baseline": "logistic (baseline)",
    "p_logit_full": "logistic (full)",
    "p_xgboost": "XGBoost",
    "statsbomb_xg": "StatsBomb xG",
}

if __name__ == "__main__":
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(config.DATA_DIR / "oof.parquet")
    y = oof["is_goal"]

    curves, curve_rows, decomp_rows = {}, [], []
    for col, label in MODEL_LABELS.items():
        tab = reliability_table(y, oof[col])
        tab["model"] = label
        curves[label] = tab
        curve_rows.append(tab)
        decomp_rows.append({"model": label, **brier_decomposition(y, oof[col])})

    pd.concat(curve_rows).to_csv(config.TABLES_DIR / "calibration_curves.csv", index=False)
    pd.DataFrame(decomp_rows).to_csv(config.TABLES_DIR / "brier_decomposition.csv", index=False)
    plot_reliability(curves, "reliability_all_models",
                     "Reliability: xG models vs StatsBomb (OOF, quantile bins)")

    # CV isotonic recalibration, before/after
    rows = []
    for col, label in MODEL_LABELS.items():
        if col == "statsbomb_xg":
            continue  # not ours to recalibrate; kept as external benchmark
        p_iso = isotonic_cv(oof, col)
        rows.append({
            "model": label,
            "log_loss_before": log_loss(y, oof[col]),
            "log_loss_after": log_loss(y, p_iso),
            "brier_before": brier_score_loss(y, oof[col]),
            "brier_after": brier_score_loss(y, p_iso),
        })
        oof[f"{col}_iso"] = p_iso
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "isotonic_before_after.csv", index=False)
    oof.to_parquet(config.DATA_DIR / "oof.parquet", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("Saved calibration curves, Brier decomposition, isotonic comparison.")
