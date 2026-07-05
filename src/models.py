"""xG models: baseline logistic, full logistic, XGBoost.

All predictions are out-of-fold under GroupKFold by match_id, so every
downstream calibration/residual statistic is computed on shots the model
never saw during training.
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
from src.features import ALL_FEATURES, BASELINE_FEATURES


def _logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def _xgb(params=None):
    kw = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=10,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.SEED,
        n_jobs=-1,
    )
    kw.update(params or {})
    return XGBClassifier(**kw)


MODELS = {
    "logit_baseline": (BASELINE_FEATURES, _logistic),
    "logit_full": (ALL_FEATURES, _logistic),
    "xgboost": (ALL_FEATURES, _xgb),
}

XGB_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
}


def _oof_single(feats, feature_cols, make_model, model_kwargs=None):
    """Out-of-fold predictions for one model spec."""
    X = feats[feature_cols].to_numpy(float)
    y = feats["is_goal"].to_numpy(int)
    groups = feats["match_id"].to_numpy()
    preds = np.full(len(feats), np.nan)
    folds = np.full(len(feats), -1)
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = make_model(model_kwargs) if model_kwargs is not None else make_model()
        model.fit(X[tr], y[tr])
        preds[te] = model.predict_proba(X[te])[:, 1]
        folds[te] = k
    return preds, folds


def tune_xgb(feats):
    """Small grid over depth x learning rate, scored by GroupKFold OOF log-loss."""
    rows = []
    for depth, lr in itertools.product(XGB_GRID["max_depth"], XGB_GRID["learning_rate"]):
        params = {"max_depth": depth, "learning_rate": lr}
        preds, _ = _oof_single(feats, ALL_FEATURES, _xgb, params)
        rows.append({**params, "oof_logloss": log_loss(feats["is_goal"], preds)})
    table = pd.DataFrame(rows).sort_values("oof_logloss").reset_index(drop=True)
    best = table.iloc[0][["max_depth", "learning_rate"]].to_dict()
    best["max_depth"] = int(best["max_depth"])
    return best, table


def fit_oof(feats, xgb_params=None):
    """OOF predictions for all three models; returns feats + p_<model> + fold."""
    out = feats.copy()
    for name, (cols, maker) in MODELS.items():
        if name == "xgboost":
            preds, folds = _oof_single(feats, cols, _xgb, xgb_params or {})
        else:
            preds, folds = _oof_single(feats, cols, maker)
        out[f"p_{name}"] = preds
    out["fold"] = folds
    return out


def fit_final_xgb(feats, xgb_params=None):
    """Fit XGBoost on all data (for feature-importance reporting only)."""
    model = _xgb(xgb_params or {})
    model.fit(feats[ALL_FEATURES].to_numpy(float), feats["is_goal"].to_numpy(int))
    imp = pd.DataFrame(
        {"feature": ALL_FEATURES, "gain_importance": model.feature_importances_}
    ).sort_values("gain_importance", ascending=False)
    return model, imp
