"""
test_pipeline.py
----------------
Tests every component of the pipeline independently.
Run this after training to verify everything works correctly.

    python3 test_pipeline.py

Each test prints PASS or FAIL with a specific reason.
"""

import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results = []

def test(name, fn):
    try:
        msg = fn()
        status = PASS
        detail = msg or ""
    except Exception as e:
        status = FAIL
        detail = str(e)
        traceback.print_exc()
    print(f"{status}  {name}")
    if detail:
        print(f"         {detail}")
    results.append((status, name))
    print()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Required files exist
# ══════════════════════════════════════════════════════════════════════════════

def test_files_exist():
    required = [
        "segmented_cs_courses.json",
        "mileage_model.pkl",
        "shap_explainer.pkl",
    ]
    optional = [
        "mileage_history_all.csv",
        "course_summary_all.csv",
        "mileage_training_real.csv",
        "training_final.csv",
    ]
    missing_required = [f for f in required if not Path(f).exists()]
    missing_optional = [f for f in optional if not Path(f).exists()]

    if missing_required:
        raise FileNotFoundError(f"Missing required files: {missing_required}")
    if missing_optional:
        return f"Optional files missing (OK if no real data yet): {missing_optional}"
    return "All files present"

test("Required files exist", test_files_exist)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Feature extractor works
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_extractor():
    from feature_extractor import load_features
    df = load_features("segmented_cs_courses.json")

    assert len(df) > 0, "No courses loaded"
    assert df.isnull().sum().sum() == 0, \
        f"NaN values in features: {df.isnull().sum()[df.isnull().sum()>0].to_dict()}"

    from synthetic_data_generator import FEATURE_COLS
    student_context_cols = {
        "student_year",
        "student_mileage",
        "num_courses_wanted",
        "rank_in_list",
        "priority_ratio",
        "budget_ratio",
    }
    course_feature_cols = [c for c in FEATURE_COLS if c not in student_context_cols]
    missing = [c for c in course_feature_cols if c not in df.columns]
    assert not missing, f"Missing feature columns: {missing}"

    return f"{len(df)} courses, {len(course_feature_cols)} course features, no NaNs"

test("Feature extractor", test_feature_extractor)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Training data has correct shape and target
# ══════════════════════════════════════════════════════════════════════════════

def test_training_data():
    if not Path("training_final.csv").exists():
        raise FileNotFoundError(
            "training_final.csv missing — run: python3 build_training_data.py"
        )

    from synthetic_data_generator import FEATURE_COLS, TARGET_COL
    df = pd.read_csv("training_final.csv", encoding="utf-8-sig")

    assert len(df) > 0, "Empty training set"

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"

    target = df[TARGET_COL]
    assert target.isnull().sum() == 0, \
        f"{target.isnull().sum()} NaN values in target column"
    assert (target >= 0).all(), "Negative target values found"
    assert target.max() <= 500, f"Unrealistically high target: {target.max()}"

    real_rows  = (df.get("data_source","") == "real").sum()
    synth_rows = (df.get("data_source","") == "synthetic").sum()

    return (f"{len(df)} total rows | "
            f"{real_rows} real | {synth_rows} synthetic | "
            f"target range: {target.min():.0f}–{target.max():.0f} pts")

test("Training data shape and target", test_training_data)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Model loads and predicts a single row
# ══════════════════════════════════════════════════════════════════════════════

def test_model_predict():
    from model import load_model, predict_threshold
    from feature_extractor import load_features
    from synthetic_data_generator import FEATURE_COLS

    model, explainer = load_model()
    df_features = load_features("segmented_cs_courses.json")

    # Use the first course as test input
    code = df_features.index[0]
    feat = df_features.loc[code].to_dict()
    feat.update({
        "student_year":       3.0,
        "student_mileage":    150.0,
        "num_courses_wanted": 5.0,
        "rank_in_list":       1.0,
        "priority_ratio":     0.8,
        "budget_ratio":       0.3,
    })

    threshold, shap = predict_threshold(model, explainer, feat)

    assert isinstance(threshold, float), "Threshold is not a float"
    assert threshold >= 1.0, f"Threshold too low: {threshold}"
    assert threshold <= 500.0, f"Threshold unrealistically high: {threshold}"
    assert len(shap) == len(FEATURE_COLS), \
        f"SHAP has {len(shap)} values, expected {len(FEATURE_COLS)}"

    # Check SHAP values sum approximately to prediction
    base = explainer.expected_value
    shap_sum = sum(shap.values())
    reconstructed = base + shap_sum
    diff = abs(reconstructed - threshold)
    assert diff < 1.0, \
        f"SHAP values don't sum to prediction (diff={diff:.2f}) — explainer mismatch"

    return f"Course: {code} | Predicted threshold: {threshold:.1f} pts | SHAP OK"

test("Model prediction + SHAP", test_model_predict)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Optimizer produces sensible allocations
# ══════════════════════════════════════════════════════════════════════════════

def test_optimizer():
    from optimizer import CourseInput, allocate_bids

    courses = [
        CourseInput("A", "High Competition Course", 1, predicted_threshold=80.0),
        CourseInput("B", "Medium Competition Course", 2, predicted_threshold=40.0),
        CourseInput("C", "Low Competition Course",  3, predicted_threshold=8.0),
    ]
    budget = 150

    results = allocate_bids(courses, total_mileage=budget, safety_margin=0.15)

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    total_bid = sum(r.recommended_bid for r in results)
    assert total_bid == budget, \
        f"Total bids {total_bid} ≠ budget {budget}"

    assert all(r.recommended_bid >= 1 for r in results), \
        "Some courses got 0 pts bid"

    # High competition course should get more than low competition
    bid_high = next(r.recommended_bid for r in results if r.code == "A")
    bid_low  = next(r.recommended_bid for r in results if r.code == "C")
    assert bid_high > bid_low, \
        f"High competition course ({bid_high}pts) should beat low ({bid_low}pts)"

    # Check risk levels are valid
    valid_risks = {"Safe", "Moderate", "Risky"}
    for r in results:
        assert r.risk_level in valid_risks, f"Invalid risk: {r.risk_level}"
        assert 0 <= r.confidence_pct <= 100, \
            f"Confidence out of range: {r.confidence_pct}"

    lines = [f"{r.name}: {r.recommended_bid}pts ({r.risk_level})"
             for r in results]
    return " | ".join(lines)

test("Optimizer allocation", test_optimizer)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Full end-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════════

def test_end_to_end():
    from strategy_engine import get_strategy_for_ranked_list, format_strategy_for_chat

    ranked_list = [
        {"code": "CAS3120-02",    "name": "Machine Learning",       "rank": 1},
        {"code": "CAS4160-01-00", "name": "Reinforcement Learning", "rank": 2},
        {"code": "CAS3205-01",    "name": "Computer Graphics",      "rank": 3},
    ]
    student = {"year": 3, "mileage": 150}

    results = get_strategy_for_ranked_list(ranked_list, student)

    assert len(results) > 0, "No results returned"

    total = sum(r.recommended_bid for r in results)
    assert 0 < total <= student["mileage"], \
        f"Total bids {total} ≠ student mileage {student['mileage']}"

    # Verify output is properly formatted
    chat_output = format_strategy_for_chat(results)
    assert "pts" in chat_output, "Output missing pts"
    assert "🎯" in chat_output, "Output missing strategy header"

    lines = []
    for r in results:
        lines.append(
            f"  #{r.rank} {r.name}: {r.recommended_bid}pts "
            f"(threshold ~{int(r.predicted_threshold)}pts, {r.risk_level})"
        )
    return "\n" + "\n".join(lines)

test("Full end-to-end pipeline", test_end_to_end)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Model quality check
# ══════════════════════════════════════════════════════════════════════════════

def test_model_quality():
    """
    Sanity checks on model predictions:
    - High ETA course should predict higher threshold than low ETA
    - Major elective with good reviews should predict higher than obscure required
    - Predictions should be in a realistic range (1-200 pts)
    """
    from model import load_model, predict_threshold
    from feature_extractor import load_features

    model, explainer = load_model()
    df = load_features("segmented_cs_courses.json")

    base_feat = {
        "student_year": 3.0, "student_mileage": 150.0,
        "num_courses_wanted": 5.0, "rank_in_list": 1.0,
        "priority_ratio": 0.8, "budget_ratio": 0.3,
        "credits": 3.0, "major_year_target": 3.0,
        "is_major_req": 0.0, "is_major_basic": 0.0,
        "is_major_elective": 1.0, "num_prerequisites": 0.0,
        "difficulty_score": 2.0, "workload_score": 2.0,
        "is_relative_grading": 0.0, "exam_weight": 60.0,
        "assignment_weight": 30.0, "is_english": 1.0,
        "lecture_type_score": 0.0, "earliest_period": 5.0,
        "has_review": 1.0,
    }

    # High demand course
    high_demand = {**base_feat, "eta_added": 150.0, "review_score": 5.0}
    pred_high, _ = predict_threshold(model, explainer, high_demand)

    # Low demand course
    low_demand  = {**base_feat, "eta_added": 15.0,  "review_score": 2.0}
    pred_low, _  = predict_threshold(model, explainer, low_demand)

    assert pred_high > pred_low, (
        f"High demand ({pred_high:.1f}pts) should predict higher than "
        f"low demand ({pred_low:.1f}pts) — model may be poorly trained"
    )

    # All real courses should predict in 1-200 range
    out_of_range = []
    for code in df.index:
        feat = df.loc[code].to_dict()
        feat.update(base_feat)
        pred, _ = predict_threshold(model, explainer, feat)
        if pred < 1 or pred > 200:
            out_of_range.append(f"{code}: {pred:.1f}pts")

    if out_of_range:
        return (f"High demand: {pred_high:.1f}pts > Low demand: {pred_low:.1f}pts ✓ | "
                f"Out of range predictions: {out_of_range}")

    return (f"High demand: {pred_high:.1f}pts > "
            f"Low demand: {pred_low:.1f}pts ✓ | "
            f"All {len(df)} courses predict in 1–200pt range ✓")

test("Model quality (directional sanity)", test_model_quality)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("═" * 60)
passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
print(f"Results: {passed}/{len(results)} passed")

if failed == 0:
    print("\n🎉 All tests passed — pipeline is working correctly")
else:
    print(f"\n{failed} test(s) failed — fix those before running the app")
    print("\nCommon fixes:")
    print("  'training_final.csv missing' → python3 build_training_data.py")
    print("  'mileage_model.pkl missing'  → python3 model.py --train")
    print("  'No courses loaded'          → check segmented_cs_courses.json path")
    print("  SHAP mismatch               → retrain: python3 model.py --train")
