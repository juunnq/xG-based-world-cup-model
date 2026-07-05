"""Step 2: features -> tuned XGBoost + logistic models -> OOF predictions,
metrics table with cluster-bootstrap CIs, feature importances."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.data import build_dataset
from src.evaluate import metrics_table
from src.features import build_features
from src.models import fit_final_xgb, fit_oof, tune_xgb

OOF_PARQUET = config.DATA_DIR / "oof.parquet"

if __name__ == "__main__":
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    shots = build_dataset()
    feats = build_features(shots)
    print(f"{len(feats)} shots, goal rate {feats['is_goal'].mean():.4f}")

    best, tuning = tune_xgb(feats)
    tuning.to_csv(config.TABLES_DIR / "xgb_tuning.csv", index=False)
    print(f"Best XGBoost params: {best}")

    oof = fit_oof(feats, xgb_params=best)
    oof.to_parquet(OOF_PARQUET, index=False)

    pred_cols = ["p_logit_baseline", "p_logit_full", "p_xgboost", "statsbomb_xg"]
    table = metrics_table(oof, pred_cols)
    table.to_csv(config.TABLES_DIR / "model_metrics.csv", index=False)
    print(table[["model", "log_loss", "brier", "auc"]].round(4).to_string(index=False))

    _, imp = fit_final_xgb(feats, xgb_params=best)
    imp.to_csv(config.TABLES_DIR / "xgb_feature_importance.csv", index=False)
    print("Saved OOF predictions, metrics, tuning table, feature importances.")
