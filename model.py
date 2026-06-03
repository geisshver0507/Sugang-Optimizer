"""
model.py
--------
Trains XGBoost to predict winning_threshold:
  "What is the minimum mileage bid that secured a seat in this course?"

This is the AI component of the system. It discovers which course
and student features correlate with high competition thresholds from
real historical data, then generalises to unseen courses.

The optimizer (optimizer.py) uses these predictions to allocate a
student's budget across their ranked list of courses.

USAGE:
  python3 model.py --train       # train and save
  python3 model.py --evaluate    # show feature importances
"""

from __future__ import annotations

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

from build_training_data import build, FEATURE_COLS, TARGET_COL

MODEL_PATH    = Path("mileage_model.pkl")
EXPLAINER_PATH = Path("shap_explainer.pkl")

# Human-readable labels for SHAP explanations shown to users
FEATURE_LABELS = {
    "eta_added":            "historical demand (ETA adds)",
    "review_score":         "professor review score",
    "is_major_elective":    "popular elective status",
    "is_major_req":         "major requirement status",
    "is_major_basic":       "major basic course",
    "difficulty_score":     "course difficulty",
    "workload_score":       "workload level",
    "is_relative_grading":  "relative grading policy",
    "num_prerequisites":    "number of prerequisites",
    "earliest_period":      "class time slot",
    "is_english":           "English instruction",
    "lecture_type_score":   "online/offline format",
    "exam_weight":          "exam weight in grading",
    "assignment_weight":    "assignment weight in grading",
    "student_year":         "your year level",
    "rank_in_list":         "your priority ranking for this course",
    "student_mileage":      "your available mileage",
    "num_courses_wanted":   "number of courses in your list",
    "priority_ratio":       "how important this course is to you",
    "budget_ratio":         "your budget relative to max",
    "major_year_target":    "target year for this course",
    "credits":              "credit count",
    "has_review":           "whether student reviews exist",
}


# ── Train ──────────────────────────────────────────────────────────────────────

def train(save: bool = True):
    print("Building training data...")
    df = build(verbose=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    w = df["sample_weight"].values if "sample_weight" in df.columns else None

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

    # Cross-validation without sample weights (fair evaluation)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_absolute_error"
    )
    print(f"\n5-fold CV MAE: {cv_mae.mean():.1f} ± {cv_mae.std():.1f} pts")
    print("(This means predictions are off by ~this many mileage points on average)")

    # Full fit with sample weights so real data dominates
    fit_kw = {"sample_weight": w} if w is not None else {}
    model.fit(X, y, **fit_kw)

    explainer = shap.TreeExplainer(model)

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(EXPLAINER_PATH, "wb") as f:
            pickle.dump(explainer, f)
        print(f"\nSaved model      → {MODEL_PATH}")
        print(f"Saved explainer  → {EXPLAINER_PATH}")

    print("\n── What the model discovered matters (feature importance) ──")
    fi = feature_importance_df(model)
    print(fi.head(15).to_string(index=False))

    return model, explainer


# ── Load ───────────────────────────────────────────────────────────────────────

def load_model():
    if not MODEL_PATH.exists():
        print("No saved model found — training now...")
        return train()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(EXPLAINER_PATH, "rb") as f:
        explainer = pickle.load(f)
    return model, explainer


# ── Predict ────────────────────────────────────────────────────────────────────

def predict_threshold(model, explainer, feature_row: dict) -> tuple:
    """
    Predict the winning threshold for one course + student scenario.

    Returns:
        predicted_threshold  (float)  — minimum bid likely needed
        shap_breakdown       (dict)   — feature → contribution in pts
    """
    X = np.array([[feature_row[f] for f in FEATURE_COLS]])
    pred = float(model.predict(X)[0])
    shap_vals = explainer.shap_values(X)[0]
    breakdown = {f: float(v) for f, v in zip(FEATURE_COLS, shap_vals)}
    return max(1.0, pred), breakdown


def top_reasons(breakdown: dict, n: int = 3) -> list:
    """Top N features by absolute SHAP value."""
    return sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True)[:n]


def explain_threshold(threshold: float, breakdown: dict, course_name: str) -> str:
    """Human-readable explanation of why the threshold is what it is."""
    reasons = top_reasons(breakdown, 3)
    lines = [f"**Predicted competition threshold: ~{int(round(threshold))} pts** "
             f"for {course_name}"]
    lines.append("Key factors:")
    for feat, val in reasons:
        label = FEATURE_LABELS.get(feat, feat)
        direction = "pushes competition up" if val > 0 else "lowers competition"
        lines.append(f"  • {label}: {direction} by ~{abs(val):.1f} pts")
    return "\n".join(lines)


# ── Feature importance ─────────────────────────────────────────────────────────

def feature_importance_df(model) -> pd.DataFrame:
    return pd.DataFrame({
        "feature":    FEATURE_COLS,
        "label":      [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.train or args.evaluate:
        m, e = train(save=args.train)
    else:
        print("Use --train to train and save, --evaluate to see importances")
