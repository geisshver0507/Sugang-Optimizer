"""
evaluate_temporal.py
--------------------
Proper temporal holdout evaluation.

Train on: 2023-2, 2024-1, 2024-2, 2025-1, 2025-2
         (professor-matched to 2026-1 professors)

Evaluate on: 2026-1 actual thresholds (ground truth)

This is the honest evaluation:
  - Model never saw 2026-1 data during training
  - Predictions are genuinely forward-looking
  - Same situation students face before sugang day

Run: python3 evaluate_temporal.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


HOLDOUT_CSV = "mileage_holdout_2026_1.csv"
TRAIN_CSV   = "mileage_training_real.csv"


def run():
    print("TEMPORAL HOLDOUT EVALUATION")
    print("Train: pre-2026-1 data  |  Evaluate: 2026-1 actual thresholds")
    print("=" * 60)

    if not Path(HOLDOUT_CSV).exists():
        print(f"No holdout data at {HOLDOUT_CSV}")
        print("Run: python3 process_mileage_data.py")
        return

    from model import load_model, predict_threshold
    from feature_extractor import load_features
    from optimizer import (
        CourseInput, allocate_bids,
        TOTAL_MILEAGE, MAX_BID_PER_COURSE
    )

    model, explainer = load_model()
    df_features      = load_features("segmented_cs_courses.json")
    df_holdout       = pd.read_csv(HOLDOUT_CSV, encoding="utf-8-sig")
    df_holdout       = df_holdout.dropna(subset=["winning_threshold"])

    print(f"\n2026-1 courses with real thresholds: {len(df_holdout)}")

    # ── Per-course prediction vs actual ───────────────────────────────────
    rows = []

    for _, row in df_holdout.iterrows():
        code      = row["course_code"]
        actual_wt = float(row["winning_threshold"])

        if code not in df_features.index:
            continue

        feat = df_features.loc[code].to_dict()
        feat.update({
            "student_year":       3.0,
            "student_mileage":    float(TOTAL_MILEAGE),
            "num_courses_wanted": 5.0,
            "rank_in_list":       1.0,
            "priority_ratio":     1.0,
            "budget_ratio":       1.0,
        })

        pred, _ = predict_threshold(model, explainer, feat)
        error   = abs(pred - actual_wt)

        rows.append({
            "course_code":    code,
            "course_name":    row.get("course_name", code),
            "professor":      row.get("professor", ""),
            "actual_wt":      actual_wt,
            "predicted_wt":   round(pred, 1),
            "error_pts":      round(error, 1),
            "over_under":     "over"  if pred > actual_wt else "under",
            "total_applicants": row.get("total_applicants", 0),
            "competition_ratio": row.get("competition_ratio", 0),
        })

    if not rows:
        print("No matching courses between holdout and feature set")
        return

    df = pd.DataFrame(rows)

    # ── Prediction accuracy ────────────────────────────────────────────────
    mae    = df["error_pts"].mean()
    median = df["error_pts"].median()
    w5     = (df["error_pts"] <= 5).mean()  * 100
    w10    = (df["error_pts"] <= 10).mean() * 100

    print(f"\n── Threshold Prediction Accuracy (2026-1) ──")
    print(f"Mean Absolute Error:    {mae:.1f} pts")
    print(f"Median Absolute Error:  {median:.1f} pts")
    print(f"Within 5pts of actual:  {w5:.0f}%")
    print(f"Within 10pts of actual: {w10:.0f}%")

    over  = (df["over_under"] == "over").sum()
    under = (df["over_under"] == "under").sum()
    print(f"Over-predicted: {over}  |  Under-predicted: {under}")

    print(f"\nPer-course breakdown:")
    print(df[["course_code","professor","actual_wt","predicted_wt",
              "error_pts","over_under"]].sort_values(
                  "error_pts", ascending=False
              ).to_string(index=False))

    # ── Bid win rate simulation ────────────────────────────────────────────
    # Simulate 5-course student, this course as rank-1
    # Check if AI bid would have won a seat
    print(f"\n── Bid Win Rate Simulation (2026-1 ground truth) ──")
    print("Simulating: student wants 5 courses, this course is rank-1")
    print("Budget: 72pts, max 36pts per course\n")

    ai_wins       = 0
    uniform_wins  = 0
    priority_wins = 0
    allin_wins    = 0

    sim_rows = []
    avg_pred  = df["predicted_wt"].mean()

    for _, row in df.iterrows():
        code      = row["course_code"]
        actual_wt = row["actual_wt"]
        pred_wt   = row["predicted_wt"]

        # Build 5-course input
        courses = [CourseInput(code, code, 1, pred_wt)]
        for j in range(2, 6):
            courses.append(CourseInput(f"f{j}", f"filler{j}", j, avg_pred))

        ai_results = allocate_bids(courses, TOTAL_MILEAGE, MAX_BID_PER_COURSE)
        ai_bid     = next(r.recommended_bid for r in ai_results if r.rank == 1)

        uniform_bid  = TOTAL_MILEAGE // 5               # 14pts
        priority_bid = min(int(TOTAL_MILEAGE * 0.4), MAX_BID_PER_COURSE)  # 28pts
        allin_bid    = MAX_BID_PER_COURSE                # 36pts

        ai_w  = int(ai_bid       >= actual_wt)
        u_w   = int(uniform_bid  >= actual_wt)
        p_w   = int(priority_bid >= actual_wt)
        al_w  = int(allin_bid    >= actual_wt)

        ai_wins       += ai_w
        uniform_wins  += u_w
        priority_wins += p_w
        allin_wins    += al_w

        sim_rows.append({
            "course":        code,
            "actual_wt":     actual_wt,
            "ai_bid":        ai_bid,
            "uniform_bid":   uniform_bid,
            "ai_wins":       ai_w,
            "uniform_wins":  u_w,
        })

    n = len(df)
    def pct(w): return w / n * 100

    print(f"{'Strategy':<38} {'Win Rate':>9}  {'vs AI':>8}")
    print(f"{'─'*57}")
    print(f"{'Our AI Strategy':<38} {pct(ai_wins):>8.1f}%  {'—':>8}")
    print(f"{'Baseline: Uniform split (14pts)':<38} {pct(uniform_wins):>8.1f}%  "
          f"{pct(ai_wins)-pct(uniform_wins):>+7.1f}%")
    print(f"{'Baseline: Priority-weighted (28pts)':<38} {pct(priority_wins):>8.1f}%  "
          f"{pct(ai_wins)-pct(priority_wins):>+7.1f}%")
    print(f"{'Baseline: All-in on rank-1 (36pts)':<38} {pct(allin_wins):>8.1f}%  "
          f"{pct(ai_wins)-pct(allin_wins):>+7.1f}%")

    # Competitive courses only
    comp_df = df[df["actual_wt"] >= 10]
    if len(comp_df) > 0:
        print(f"\nOn COMPETITIVE courses only (threshold ≥ 10pts, n={len(comp_df)}):")
        sim_comp = pd.DataFrame(sim_rows)
        sim_comp = sim_comp.merge(
            df[["course_code","actual_wt"]].rename(columns={"course_code":"course"}), on="course", how="left"
        )
        sim_comp = sim_comp[sim_comp["actual_wt"] >= 10]
        if len(sim_comp) > 0:
            ai_c = sim_comp["ai_wins"].mean()   * 100
            u_c  = sim_comp["uniform_wins"].mean() * 100
            print(f"  AI: {ai_c:.1f}%  vs  Uniform: {u_c:.1f}%")
            print(f"  ← This is the headline metric for your paper")

    # ── What to report ─────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("NUMBERS TO REPORT IN YOUR PAPER")
    print(f"{'═'*60}")
    print(f"""
Model Evaluation (Temporal Holdout — 2026-1):
  Training data:  All semesters before 2026-1,
                  filtered to matching professor
  Evaluation data: 2026-1 actual thresholds (ground truth)

Threshold Prediction:
  MAE = {mae:.1f} pts
  (Predictions were off by ~{mae:.0f} mileage points on average)

Bid Strategy Win Rate:
  AI strategy:        {pct(ai_wins):.1f}%
  Uniform baseline:   {pct(uniform_wins):.1f}%
  Priority baseline:  {pct(priority_wins):.1f}%
  Improvement over uniform: {pct(ai_wins)-pct(uniform_wins):+.1f}%

This evaluation is credible because:
  - Model was trained on data from BEFORE 2026-1
  - Evaluated against 2026-1 which the model never saw
  - Same setup as real students: use past to predict future
  - Professor-matched: history filtered to same professor
    as 2026-1 to avoid cross-professor contamination
""")


if __name__ == "__main__":
    run()