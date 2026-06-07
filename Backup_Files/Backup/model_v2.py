"""
model.py  (updated)
-------------------
Now uses build_training_data.py which automatically:
  - uses real scraped data where available
  - fills in synthetic only for courses with no history
"""

import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

import xgboost as xgb
import shap
from sklearn.model_selection import cross_val_score, KFold

from build_training_data import build
from synthetic_data_generator import FEATURE_COLS, TARGET_COL

MODEL_PATH          = Path("mileage_model.pkl")
SHAP_EXPLAINER_PATH = Path("shap_explainer.pkl")


def train(save: bool = True):
    print("Building training data...")
    df = build(verbose=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Use sample weights: real data rows count 3× more than synthetic
    sample_weight = None
    if "data_source" in df.columns:
        sample_weight = np.where(df["data_source"] == "real", 3.0, 1.0)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

    # CV (without sample weights for fair evaluation)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    print(f"\n5-fold CV MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f} pts")

    # Full fit with weights
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(X, y, **fit_kwargs)

    explainer = shap.TreeExplainer(model)

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(SHAP_EXPLAINER_PATH, "wb") as f:
            pickle.dump(explainer, f)
        print(f"Saved → {MODEL_PATH}")

    print("\nTop features discovered by model:")
    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(10)
    print(fi.to_string(index=False))

    return model, explainer


def load_model():
    if not MODEL_PATH.exists():
        return train()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SHAP_EXPLAINER_PATH, "rb") as f:
        explainer = pickle.load(f)
    return model, explainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
