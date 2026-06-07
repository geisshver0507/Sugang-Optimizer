"""
model_robust.py
----------------
A stronger, small-data-friendly mileage threshold model.

Why this exists:
- The plain XGBoost model can overfit whichever semester dominates the data.
- Historical mileage rows are sparse and noisy across professors/semesters.
- Some thresholds are effectively censored at the 36 point bid cap.

This model blends two signals:
1. A hierarchical historical prior:
   course+professor+student-year -> base course+professor+student-year
   -> course/base history -> global history.
2. A conservative ML regressor/classifier trained on course features.

The result is intentionally less flashy and more stable: use exact history when it
is reliable, back off gracefully when history is sparse, and do not let old noisy
semesters wreck predictions for unseen courses.

USAGE:
  python3 model_robust.py --train
  python3 model_robust.py --evaluate
  python3 model_robust.py --predict-code CAS2103-02 --student-year 3

To try it in the app later, change imports from `model` to `model_robust`.
This file does not overwrite model.py or mileage_model.pkl.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - fallback for machines without xgboost
    xgb = None

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold

from feature_extractor import flatten_json, load_features
from synthetic_data_generator import FEATURE_COLS as BASE_FEATURE_COLS
from mileage_capacity_pipeline import (
    apply_cold_start_weight,
    build_average_mileage_training_frame,
    capacity_application_correlation,
)

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "mileage_training_real.csv"
HOLDOUT_CSV = ROOT / "mileage_holdout_2026_1.csv"
JSON_PATH = ROOT / "segmented_cs_courses.json"
MODEL_PATH = ROOT / "mileage_model_robust.pkl"
TARGET_COL = "average_mileage"
MAX_BID_PER_COURSE = 36.0

# Extra forward-looking features. We deliberately avoid post-registration columns
# like total_applicants and avg_mileage inside the ML model.
ML_FEATURE_COLS = list(dict.fromkeys(BASE_FEATURE_COLS + [
    "semester",
    "is_spring",
    "recency_age",
]))

FEATURE_LABELS = {
    "history_prior": "similar-course historical average mileage",
    "history_strength": "amount of matching history",
    "ml_estimate": "course-feature model estimate",
    "high_demand_probability": "chance this is a high-demand course",
    "eta_added": "historical demand from ETA adds",
    "cold_start_multiplier": "cold-start heuristic multiplier",
    "cold_start_condition": "cold-start condition code",
    "predicted_average_mileage": "predicted average mileage",
    "recommended_bid": "recommended bid after +3.0 safety buffer",
    "max_capacity": "classroom/applicant capacity",
    "demand_proxy": "future demand proxy from ETA/cart adds",
    "demand_capacity_ratio": "ETA demand divided by capacity",
    "capacity_demand_gap": "ETA demand minus capacity",
    "review_score": "professor review score",
    "is_major_elective": "major elective status",
    "is_major_req": "major requirement status",
    "is_major_basic": "major basic course",
    "difficulty_score": "course difficulty",
    "workload_score": "workload level",
    "is_relative_grading": "relative grading policy",
    "num_prerequisites": "number of prerequisites",
    "earliest_period": "class time slot",
    "is_english": "English instruction",
    "lecture_type_score": "online/offline format",
    "exam_weight": "exam weight in grading",
    "assignment_weight": "assignment weight in grading",
    "student_year": "student year level",
    "rank_in_list": "priority ranking for this course",
    "num_courses_wanted": "number of wanted courses",
    "priority_ratio": "course priority ratio",
    "semester": "target semester",
    "is_spring": "spring semester indicator",
    "recency_age": "distance from most recent training semester",
}


def _safe_print(text: Any = "") -> None:
    """Print Unicode safely from Windows terminals with legacy encodings."""
    try:
        print(text)
    except UnicodeEncodeError:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.flush()
            sys.stdout.buffer.write((str(text) + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()
        else:
            print(str(text).encode("ascii", errors="replace").decode("ascii"))

def _as_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _norm_prof(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower().replace(" ", "")


def _base_code(course_code: Any) -> str:
    text = str(course_code or "").strip()
    return text.split("-")[0] if text else ""


def _semester_index(year: Any, semester: Any) -> int:
    try:
        return int(year) * 2 + int(semester)
    except Exception:
        return 0


def _weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return float("nan")
    v = v[ok]
    w = w[ok]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cutoff = w.sum() / 2.0
    return float(v[np.searchsorted(np.cumsum(w), cutoff, side="left")])


def _clean_target(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    # 37 is used in this project as above the 36 point bidding cap. Treat it as
    # max-risk but keep model targets inside the legal bid range.
    y = y.clip(lower=1.0, upper=MAX_BID_PER_COURSE)
    return y


def _choose_base_target(df: pd.DataFrame, target_mode: str) -> pd.Series:
    if target_mode == "average_mileage":
        for column in (
            "average_mileage",
            "avg_all_bids",
            "avg_winning_bid",
            "major_quota_threshold",
            "winning_threshold",
        ):
            if column in df.columns:
                return df[column]
        raise ValueError("No usable average mileage target column found")
    if target_mode == "winning_threshold":
        if "winning_threshold" not in df.columns:
            raise ValueError("winning_threshold column is missing")
        return df["winning_threshold"]
    if target_mode == "major_quota_threshold":
        if "major_quota_threshold" in df.columns:
            return df["major_quota_threshold"].fillna(df.get("winning_threshold"))
        if "winning_threshold" in df.columns:
            return df["winning_threshold"]
        raise ValueError("No usable target column found")
    if target_mode == "by_year":
        if "major_quota_threshold" in df.columns:
            return df["major_quota_threshold"].fillna(df.get("winning_threshold"))
        return df["winning_threshold"]
    raise ValueError(f"Unknown target mode: {target_mode}")


def _numeric_column(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
    else:
        values = pd.Series(default, index=df.index, dtype=float)
    return values.fillna(default)


def _expand_student_year_targets(df: pd.DataFrame, target_mode: str) -> pd.DataFrame:
    """Create one row per student year when threshold_yr1..4 are available."""
    if target_mode != "by_year":
        out = df.copy()
        out["student_year"] = _numeric_column(out, "student_year", 3.0)
        out["target_raw"] = _choose_base_target(out, target_mode)
        return out

    year_cols = [f"threshold_yr{i}" for i in range(1, 5)]
    if not all(c in df.columns for c in year_cols):
        out = df.copy()
        out["student_year"] = _numeric_column(out, "student_year", 3.0)
        out["target_raw"] = _choose_base_target(out, "major_quota_threshold")
        return out

    parts = []
    fallback = _choose_base_target(df, "major_quota_threshold")
    for student_year in range(1, 5):
        part = df.copy()
        col = f"threshold_yr{student_year}"
        part["student_year"] = float(student_year)
        part["target_raw"] = pd.to_numeric(part[col], errors="coerce").fillna(fallback)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _load_professors_from_json(json_path: Path) -> Dict[str, str]:
    flat = flatten_json(str(json_path))
    profs = {}
    for code, course in flat.items():
        prof = course.get("professor") or course.get("metadata", {}).get("professor", "")
        profs[code] = _norm_prof(prof)
    return profs


def _prepare_frame(
    mileage_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_mode: str,
    max_semester_index: Optional[int] = None,
) -> Tuple[pd.DataFrame, int]:
    df = mileage_df.copy()
    if "course_code" not in df.columns:
        raise ValueError("course_code column is required")

    df["course_code"] = df["course_code"].astype(str)
    df["base_code"] = df.get("base_code", df["course_code"].map(_base_code)).fillna("").astype(str)
    df["professor_norm"] = df.get("professor_norm", df.get("professor", "")).map(_norm_prof)
    df["semester"] = _numeric_column(df, "semester", 1.0).astype(int)
    df["year"] = _numeric_column(df, "year", 0.0).astype(int)
    df["semester_index"] = [_semester_index(y, s) for y, s in zip(df["year"], df["semester"])]
    max_idx = int(max_semester_index or df["semester_index"].max())
    df["recency_age"] = (max_idx - df["semester_index"]).clip(lower=0)
    df["is_spring"] = (df["semester"] == 1).astype(float)
    df, _capacity_corr = build_average_mileage_training_frame(df)

    expanded = _expand_student_year_targets(df, target_mode)
    expanded["target"] = _clean_target(expanded["target_raw"])
    expanded = expanded.dropna(subset=["target"])

    if "recency_weight" not in expanded.columns:
        expanded["recency_weight"] = 1.0 / (1.0 + expanded["recency_age"].astype(float))
    expanded["sample_weight"] = pd.to_numeric(
        expanded.get("sample_weight", expanded["recency_weight"]), errors="coerce"
    ).fillna(expanded["recency_weight"])
    expanded["sample_weight"] = expanded["sample_weight"].clip(lower=0.25, upper=10.0)

    feat_reset = features_df.reset_index()
    expanded = expanded.merge(feat_reset, on="course_code", how="left", suffixes=("", "_feat"))
    for col in ("max_capacity", "eta_added", "demand_proxy", "demand_capacity_ratio", "capacity_demand_gap"):
        feat_col = f"{col}_feat"
        if col not in expanded.columns and feat_col in expanded.columns:
            expanded[col] = expanded[feat_col]
        elif feat_col in expanded.columns:
            current = pd.to_numeric(expanded[col], errors="coerce")
            fallback = pd.to_numeric(expanded[feat_col], errors="coerce")
            expanded[col] = current.where(current > 0, fallback)

    if "num_applied" in expanded.columns:
        applied = pd.to_numeric(expanded["num_applied"], errors="coerce").fillna(0.0)
        expanded["demand_proxy"] = applied.where(applied > 0, _numeric_column(expanded, "eta_added", 0.0))
        safe_capacity = _numeric_column(expanded, "max_capacity", 0.0).where(lambda s: s > 0, 1.0)
        expanded["demand_capacity_ratio"] = (expanded["demand_proxy"] / safe_capacity).clip(lower=0.0, upper=8.0)

    for col in BASE_FEATURE_COLS:
        if col not in expanded.columns:
            expanded[col] = 0.0
    expanded["num_courses_wanted"] = _numeric_column(expanded, "num_courses_wanted", 5.0)
    expanded["rank_in_list"] = _numeric_column(expanded, "rank_in_list", 1.0)
    expanded["priority_ratio"] = _numeric_column(expanded, "priority_ratio", 1.0)

    for col in ML_FEATURE_COLS:
        expanded[col] = _numeric_column(expanded, col, 0.0)

    return expanded, max_idx


def _group_stats(df: pd.DataFrame, keys: List[str]) -> Dict[Tuple[Any, ...], Dict[str, float]]:
    stats: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    for key, group in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        y = group["target"].astype(float)
        w = group["sample_weight"].astype(float)
        stats[key] = {
            "n": float(len(group)),
            "weight": float(w.sum()),
            "mean": float(np.average(y, weights=w)),
            "median": _weighted_median(y, w),
            "std": float(y.std(ddof=0) if len(y) > 1 else 8.0),
            "high_rate": float(np.average((y >= 24.0).astype(float), weights=w)),
        }
    return stats


def _build_priors(df: pd.DataFrame) -> Dict[str, Any]:
    w = df["sample_weight"].astype(float)
    y = df["target"].astype(float)
    global_median = _weighted_median(y, w)
    global_mean = float(np.average(y, weights=w))
    return {
        "global": {
            "n": float(len(df)),
            "weight": float(w.sum()),
            "mean": global_mean,
            "median": global_median,
            "std": float(y.std(ddof=0)),
            "high_rate": float(np.average((y >= 24.0).astype(float), weights=w)),
        },
        "course_prof_year": _group_stats(df, ["course_code", "professor_norm", "student_year"]),
        "base_prof_year": _group_stats(df, ["base_code", "professor_norm", "student_year"]),
        "course_year": _group_stats(df, ["course_code", "student_year"]),
        "base_year": _group_stats(df, ["base_code", "student_year"]),
        "course_prof": _group_stats(df, ["course_code", "professor_norm"]),
        "base_prof": _group_stats(df, ["base_code", "professor_norm"]),
        "course": _group_stats(df, ["course_code"]),
        "base": _group_stats(df, ["base_code"]),
    }


def _history_estimate(priors: Dict[str, Any], course_code: str, professor_norm: str, student_year: float) -> Tuple[float, float, Dict[str, float]]:
    base_code = _base_code(course_code)
    sy = float(student_year or 3.0)
    candidates = [
        ("course_prof_year", (course_code, professor_norm, sy), 4.0),
        ("base_prof_year", (base_code, professor_norm, sy), 3.2),
        ("course_prof", (course_code, professor_norm), 2.4),
        ("base_prof", (base_code, professor_norm), 2.0),
        # Same course with a different professor is useful, but much weaker.
        ("course_year", (course_code, sy), 0.8),
        ("base_year", (base_code, sy), 0.6),
        ("course", (course_code,), 0.5),
        ("base", (base_code,), 0.4),
    ]

    values = [float(priors["global"]["median"])]
    weights = [1.0]
    matched_weight = 0.0
    matched_n = 0.0

    for name, key, strength in candidates:
        if not all(str(part) for part in key if not isinstance(part, float)):
            continue
        stat = priors.get(name, {}).get(key)
        if not stat:
            continue
        reliability = stat["weight"] / (stat["weight"] + 4.0)
        weight = strength * reliability
        values.append(float(stat["median"]))
        weights.append(weight)
        matched_weight += weight
        matched_n += stat["n"]

    estimate = float(np.average(values, weights=weights))
    history_strength = float(np.clip(matched_weight / 6.0, 0.0, 1.0))
    details = {
        "history_prior": estimate,
        "history_strength": history_strength,
        "history_matches": matched_n,
    }
    return estimate, history_strength, details


def _make_regressor(seed: int = 42):
    if xgb is not None:
        return xgb.XGBRegressor(
            n_estimators=280,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_alpha=0.3,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=seed,
            verbosity=0,
        )
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=4,
        random_state=seed,
    )


def _make_classifier(seed: int = 42):
    if xgb is not None:
        return xgb.XGBClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_alpha=0.3,
            reg_lambda=5.0,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=4,
        random_state=seed,
    )


def _feature_row_from_course(
    bundle: Dict[str, Any],
    course_code: str,
    professor_norm: str = "",
    student_year: float = 3.0,
    num_courses_wanted: float = 5.0,
    rank_in_list: float = 1.0,
    semester: int = 1,
) -> Dict[str, float]:
    def numeric_or_zero(value: Any) -> float:
        converted = pd.to_numeric(value, errors="coerce")
        return 0.0 if pd.isna(converted) else float(converted)

    features_by_code = bundle.get("course_features", {})
    row = dict(features_by_code.get(course_code, {}))
    row["course_code"] = course_code
    row["base_code"] = _base_code(course_code)
    row["professor_norm"] = professor_norm or bundle.get("current_professors", {}).get(course_code, "")
    row["student_year"] = float(student_year)
    row["num_courses_wanted"] = float(num_courses_wanted)
    row["rank_in_list"] = float(rank_in_list)
    row["priority_ratio"] = (float(num_courses_wanted) - float(rank_in_list) + 1.0) / max(float(num_courses_wanted), 1.0)
    row["semester"] = int(semester)
    row["is_spring"] = 1.0 if int(semester) == 1 else 0.0
    row["recency_age"] = 0.0
    row["max_capacity"] = numeric_or_zero(row.get("max_capacity", 0.0))
    row["eta_added"] = numeric_or_zero(row.get("eta_added", 0.0))
    row["demand_proxy"] = row["eta_added"]
    safe_capacity = max(row["max_capacity"], 1.0)
    row["demand_capacity_ratio"] = min(max(row["demand_proxy"] / safe_capacity, 0.0), 8.0)
    row["capacity_demand_gap"] = row["demand_proxy"] - row["max_capacity"]
    row["historical_records"] = 0.0
    return row


def _ml_matrix_from_rows(rows: List[Dict[str, Any]]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append([float(pd.to_numeric(row.get(col, 0.0), errors="coerce") if row.get(col, 0.0) is not None else 0.0) for col in ML_FEATURE_COLS])
    arr = np.asarray(matrix, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def train(
    train_csv: str | Path = TRAIN_CSV,
    json_path: str | Path = JSON_PATH,
    target_mode: str = "average_mileage",
    save: bool = True,
    anchor_csv: str | Path | None = None,
    anchor_weight: float = 4.0,
    model_path: str | Path = MODEL_PATH,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], None]:
    train_csv = _as_path(train_csv)
    json_path = _as_path(json_path)
    model_path = _as_path(model_path)

    if verbose:
        print("Building robust mileage model...")
        print(f"Training data: {train_csv.name}")
        print(f"Target mode:   {target_mode}")

    raw = pd.read_csv(train_csv, encoding="utf-8-sig")
    anchor_rows = 0
    if anchor_csv:
        anchor_path = _as_path(anchor_csv)
        if anchor_path.exists():
            anchor = pd.read_csv(anchor_path, encoding="utf-8-sig")
            base_weight = _numeric_column(anchor, "sample_weight", 1.0)
            anchor["sample_weight"] = base_weight * float(anchor_weight)
            raw = pd.concat([raw, anchor], ignore_index=True)
            anchor_rows = len(anchor)
            if verbose:
                print(f"Anchor data:  {anchor_path.name} ({anchor_rows} rows, weight x{anchor_weight:g})")
        elif verbose:
            print(f"Anchor CSV not found, skipping: {anchor_path}")

    features_df = load_features(str(json_path))
    train_df, max_idx = _prepare_frame(raw, features_df, target_mode)
    capacity_corr = capacity_application_correlation(raw)

    X = train_df[ML_FEATURE_COLS].astype(float).values
    y = train_df["target"].astype(float).values
    w = train_df["sample_weight"].astype(float).values
    high_y = (y >= 24.0).astype(int)

    regressor = _make_regressor()
    classifier = _make_classifier()
    regressor.fit(X, y, sample_weight=w)
    if len(np.unique(high_y)) > 1:
        classifier.fit(X, high_y, sample_weight=w)
    else:
        classifier = None

    priors = _build_priors(train_df)
    features_reset = features_df.reset_index()
    course_features = {
        row["course_code"]: {col: float(row[col]) for col in features_df.columns}
        for _, row in features_reset.iterrows()
    }

    bundle: Dict[str, Any] = {
        "version": "robust-average-v2",
        "target_mode": target_mode,
        "ml_feature_cols": ML_FEATURE_COLS,
        "regressor": regressor,
        "classifier": classifier,
        "priors": priors,
        "course_features": course_features,
        "current_professors": _load_professors_from_json(json_path),
        "max_semester_index": max_idx,
        "training_rows": int(len(train_df)),
        "raw_training_rows": int(len(raw)),
        "anchor_rows": int(anchor_rows),
        "anchor_weight": float(anchor_weight) if anchor_csv else 0.0,
        "capacity_application_correlation": capacity_corr,
    }

    if verbose:
        print(f"Prepared rows: {len(train_df)} from {len(raw)} historical course-semester rows")
        print(f"Capacity/applicant correlation: {capacity_corr:.3f}")
        print(f"Global weighted median target: {priors['global']['median']:.1f} pts")
        _print_cv(train_df)

    if save:
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)
        if verbose:
            print(f"Saved robust model -> {model_path}")

    return bundle, None


def _predict_from_row(bundle: Dict[str, Any], feature_row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    row = dict(feature_row)
    course_code = str(row.get("course_code", "") or "")
    professor_norm = _norm_prof(row.get("professor_norm") or row.get("professor") or "")
    if course_code and not professor_norm:
        professor_norm = bundle.get("current_professors", {}).get(course_code, "")
    student_year = float(row.get("student_year", 3.0) or 3.0)

    for col in BASE_FEATURE_COLS:
        row[col] = float(pd.to_numeric(row.get(col, 0.0), errors="coerce") if row.get(col, 0.0) is not None else 0.0)
    row["semester"] = int(row.get("semester", 1) or 1)
    row["is_spring"] = 1.0 if int(row["semester"]) == 1 else 0.0
    row["recency_age"] = float(row.get("recency_age", 0.0) or 0.0)
    safe_capacity = max(float(row.get("max_capacity", 0.0) or 0.0), 1.0)
    if not row.get("demand_proxy"):
        row["demand_proxy"] = float(row.get("eta_added", 0.0) or 0.0)
    row["demand_capacity_ratio"] = float(np.clip(float(row.get("demand_proxy", 0.0)) / safe_capacity, 0.0, 8.0))
    row["capacity_demand_gap"] = float(row.get("demand_proxy", 0.0)) - float(row.get("max_capacity", 0.0) or 0.0)

    X = _ml_matrix_from_rows([row])
    ml_pred = float(bundle["regressor"].predict(X)[0])

    clf = bundle.get("classifier")
    if clf is not None and hasattr(clf, "predict_proba"):
        high_prob = float(clf.predict_proba(X)[0, 1])
    else:
        high_prob = 0.0

    if course_code:
        hist, hist_strength, hist_details = _history_estimate(
            bundle["priors"], course_code, professor_norm, student_year
        )
    else:
        hist = float(bundle["priors"]["global"]["median"])
        hist_strength = 0.0
        hist_details = {"history_prior": hist, "history_strength": 0.0, "history_matches": 0.0}

    # Strong exact-ish history should dominate; sparse history still regularizes ML.
    hist_weight = 0.10 + 0.55 * hist_strength
    pred = hist_weight * hist + (1.0 - hist_weight) * ml_pred

    # If classifier sees high-demand risk, pull the estimate upward gently. This is
    # especially useful for censored 36/37 point courses.
    high_anchor = 12.0 + 24.0 * high_prob
    if high_prob >= 0.65:
        pred = 0.82 * pred + 0.18 * max(pred, high_anchor)

    pred = float(np.clip(pred, 1.0, MAX_BID_PER_COURSE))
    pred, cold_start = apply_cold_start_weight(
        pred,
        historical_records=int(round(hist_details.get("history_matches", 0.0))),
        course_id=course_code,
        professor=professor_norm,
        review_score=row.get("review_score"),
        has_known_professor=bool(professor_norm),
    )
    recommended_bid = float(np.clip(pred + 3.0, 1.0, MAX_BID_PER_COURSE))

    breakdown = {
        "history_prior": hist_details["history_prior"],
        "history_strength": hist_details["history_strength"],
        "ml_estimate": ml_pred,
        "high_demand_probability": high_prob,
        "cold_start_multiplier": cold_start.multiplier,
        "cold_start_condition": cold_start.condition,
        "predicted_average_mileage": pred,
        "recommended_bid": recommended_bid,
        "max_capacity": float(row.get("max_capacity", 0.0) or 0.0),
        "demand_capacity_ratio": float(row.get("demand_capacity_ratio", 0.0) or 0.0),
    }
    return pred, breakdown


def predict_course(
    bundle: Dict[str, Any],
    course_code: str,
    professor: str = "",
    student_year: float = 3.0,
    num_courses_wanted: float = 5.0,
    rank_in_list: float = 1.0,
    semester: int = 1,
) -> Tuple[float, Dict[str, float]]:
    row = _feature_row_from_course(
        bundle,
        course_code=course_code,
        professor_norm=_norm_prof(professor),
        student_year=student_year,
        num_courses_wanted=num_courses_wanted,
        rank_in_list=rank_in_list,
        semester=semester,
    )
    return _predict_from_row(bundle, row)


def predict_threshold(model: Dict[str, Any], explainer: Any, feature_row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Compatibility wrapper matching model.py's API."""
    return _predict_from_row(model, feature_row)


def load_model(model_path: str | Path = MODEL_PATH) -> Tuple[Dict[str, Any], None]:
    model_path = _as_path(model_path)
    if not model_path.exists():
        print("No robust model found; training now...")
        return train(save=True, model_path=model_path)
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    if (
        not isinstance(bundle, dict)
        or bundle.get("version") != "robust-average-v2"
        or list(bundle.get("ml_feature_cols", [])) != ML_FEATURE_COLS
    ):
        print("Saved robust model is stale; retraining with capacity/average-mileage features...")
        return train(save=True, model_path=model_path)
    return bundle, None


def top_reasons(breakdown: Dict[str, float], n: int = 3) -> List[Tuple[str, float]]:
    return sorted(breakdown.items(), key=lambda x: abs(float(x[1])), reverse=True)[:n]


def explain_threshold(threshold: float, breakdown: Dict[str, float], course_name: str) -> str:
    lines = [f"Predicted competition threshold: ~{int(round(threshold))} pts for {course_name}"]
    lines.append("Key signals:")
    for feat, val in top_reasons(breakdown, 3):
        label = FEATURE_LABELS.get(feat, feat)
        lines.append(f"  - {label}: {val:.2f}")
    return "\n".join(lines)


def feature_importance_df(model: Dict[str, Any]) -> pd.DataFrame:
    reg = model.get("regressor") if isinstance(model, dict) else model
    if hasattr(reg, "feature_importances_"):
        importance = np.asarray(reg.feature_importances_, dtype=float)
    else:
        importance = np.zeros(len(ML_FEATURE_COLS), dtype=float)
    return pd.DataFrame({
        "feature": ML_FEATURE_COLS,
        "label": [FEATURE_LABELS.get(f, f) for f in ML_FEATURE_COLS],
        "importance": importance,
    }).sort_values("importance", ascending=False)


def _print_cv(train_df: pd.DataFrame) -> None:
    if len(train_df) < 20:
        return
    groups = train_df["base_code"].astype(str).values
    if len(set(groups)) >= 5:
        splitter = GroupKFold(n_splits=5).split(train_df, groups=groups)
        label = "5-fold grouped-by-base-course CV"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42).split(train_df)
        label = "5-fold CV"

    errors = []
    X_all = train_df[ML_FEATURE_COLS].astype(float).values
    y_all = train_df["target"].astype(float).values
    w_all = train_df["sample_weight"].astype(float).values

    for train_idx, test_idx in splitter:
        reg = _make_regressor(seed=100 + len(errors))
        reg.fit(X_all[train_idx], y_all[train_idx], sample_weight=w_all[train_idx])
        pred = np.clip(reg.predict(X_all[test_idx]), 1.0, MAX_BID_PER_COURSE)
        errors.append(mean_absolute_error(y_all[test_idx], pred))
    print(f"{label} MAE for ML part only: {np.mean(errors):.1f} +/- {np.std(errors):.1f} pts")
    print("Final predictions also blend in hierarchical course/professor history.")


def evaluate(
    bundle: Dict[str, Any],
    holdout_csv: str | Path = HOLDOUT_CSV,
    json_path: str | Path = JSON_PATH,
    target_mode: Optional[str] = None,
    student_year: Optional[int] = None,
) -> pd.DataFrame:
    holdout_csv = _as_path(holdout_csv)
    json_path = _as_path(json_path)
    target_mode = target_mode or bundle.get("target_mode", "by_year")
    raw = pd.read_csv(holdout_csv, encoding="utf-8-sig")
    features_df = load_features(str(json_path))
    eval_df, _ = _prepare_frame(raw, features_df, target_mode, max_semester_index=bundle.get("max_semester_index"))
    if student_year is not None:
        eval_df = eval_df[eval_df["student_year"] == float(student_year)]

    rows = []
    for _, row in eval_df.iterrows():
        feature_row = {col: row.get(col, 0.0) for col in ML_FEATURE_COLS}
        feature_row.update({
            "course_code": row["course_code"],
            "professor_norm": row.get("professor_norm", ""),
            "student_year": row.get("student_year", 3.0),
            "num_courses_wanted": 5.0,
            "rank_in_list": 1.0,
            "priority_ratio": 1.0,
            "semester": int(row.get("semester", 1)),
        })
        pred, breakdown = _predict_from_row(bundle, feature_row)
        actual = float(row["target"])
        rows.append({
            "course_code": row["course_code"],
            "course_name": row.get("course_name", row["course_code"]),
            "professor": row.get("professor", ""),
            "student_year": int(row.get("student_year", 0)),
            "actual": actual,
            "predicted": round(pred, 1),
            "abs_error": round(abs(pred - actual), 1),
            "history_strength": round(breakdown["history_strength"], 2),
            "high_prob": round(breakdown["high_demand_probability"], 2),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        print("No holdout rows could be evaluated.")
        return result

    mae = result["abs_error"].mean()
    med = result["abs_error"].median()
    within5 = (result["abs_error"] <= 5).mean() * 100
    within10 = (result["abs_error"] <= 10).mean() * 100
    print("ROBUST MODEL HOLDOUT EVALUATION")
    print(f"Rows: {len(result)} | target_mode={target_mode}")
    if student_year is not None:
        print(f"Student year: {student_year}")
    print(f"MAE: {mae:.1f} pts | median error: {med:.1f} pts")
    print(f"Within 5 pts: {within5:.0f}% | within 10 pts: {within10:.0f}%")
    print("\nWorst errors:")
    _safe_print(result.sort_values("abs_error", ascending=False).head(15).to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate robust Sugang mileage model")
    parser.add_argument("--train", action="store_true", help="Train and save robust model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on mileage_holdout_2026_1.csv")
    parser.add_argument("--predict-code", help="Predict one course code, e.g. CAS2103-02")
    parser.add_argument("--professor", default="", help="Professor name for prediction; defaults to JSON professor")
    parser.add_argument("--student-year", type=int, default=3, help="Student year 1-4")
    parser.add_argument("--num-courses", type=int, default=5, help="Number of courses wanted")
    parser.add_argument("--rank", type=int, default=1, help="Rank of this course in student preference list")
    parser.add_argument("--semester", type=int, default=1, help="Target semester number")
    parser.add_argument("--target-mode", choices=["average_mileage", "by_year", "major_quota_threshold", "winning_threshold"], default="average_mileage")
    parser.add_argument("--anchor-csv", default="", help="Optional current-semester CSV to anchor production training, e.g. mileage_holdout_2026_1.csv")
    parser.add_argument("--anchor-weight", type=float, default=4.0, help="Sample weight multiplier for --anchor-csv rows")
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--holdout-csv", default=str(HOLDOUT_CSV))
    parser.add_argument("--json", default=str(JSON_PATH))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    args = parser.parse_args()

    if args.train:
        bundle, _ = train(args.train_csv, args.json, args.target_mode, save=True, model_path=args.model_path, anchor_csv=args.anchor_csv or None, anchor_weight=args.anchor_weight)
    else:
        bundle, _ = load_model(args.model_path)

    if args.evaluate:
        evaluate(bundle, args.holdout_csv, args.json, args.target_mode, student_year=args.student_year)

    if args.predict_code:
        pred, breakdown = predict_course(
            bundle,
            course_code=args.predict_code,
            professor=args.professor,
            student_year=args.student_year,
            num_courses_wanted=args.num_courses,
            rank_in_list=args.rank,
            semester=args.semester,
        )
        print(explain_threshold(pred, breakdown, args.predict_code))
        print("Breakdown:")
        for key, val in breakdown.items():
            print(f"  {key}: {val:.3f}")

    if not (args.train or args.evaluate or args.predict_code):
        print("Use --train, --evaluate, or --predict-code. Example:")
        print("  python3 model_robust.py --train --evaluate")


if __name__ == "__main__":
    main()
