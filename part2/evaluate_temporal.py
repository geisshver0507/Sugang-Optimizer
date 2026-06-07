"""
evaluate_temporal.py
--------------------
Proper temporal holdout evaluation using 2026-1 as ground truth.

WHAT WE'RE EVALUATING:
  "If a student used our system before 2026-1 sugang and submitted
   the recommended bid, would they have gotten enrolled based on
   what actually happened in 2026-1?"

TWO LEVELS OF ANALYSIS:

1. THRESHOLD ACCURACY
   How close were our predicted thresholds to the actual safe threshold?
   (MAE in mileage points)

2. ENROLLMENT PROBABILITY SIMULATION
   For the bid our model recommended, what was the actual
   enrollment rate among 2026-1 students who bid that amount?
   This accounts for tie-breaking — some students at the same bid
   got in, some didn't, depending on their academic profile.

3. BASELINE COMPARISON
   Does our model outperform naive strategies?
   (uniform split, priority-weighted, all-in)

4. FAILURE ANALYSIS
   Where did the model go wrong and why?
   This helps explain limitations in the paper.
"""

import pandas as pd
import numpy as np
from pathlib import Path

HOLDOUT_CSV = "mileage_holdout_2026_1.csv"
HISTORY_CSV = "mileage_history_all.csv"


def get_enrollment_probability(history_df: pd.DataFrame,
                                course_code: str,
                                bid_amount: int,
                                student_year: int = None) -> dict:
    """
    Given a course and a bid amount, compute the actual enrollment
    probability from 2026-1 history.

    Returns dict with:
        prob_overall    - P(enrolled | bid=X) across all students
        prob_for_year   - P(enrolled | bid=X, year=student_year)
        n_same_bid      - how many students bid exactly X
        n_enrolled_same - how many of those got enrolled
        interpretation  - human readable
    """
    course_data = history_df[
        (history_df["course_code"] == course_code) &
        (history_df["year"] == 2026) &
        (history_df["semester"] == 1) &
        (history_df["rank"] > 0)   # competitive only, not priority queue
    ].copy()

    if len(course_data) == 0:
        return {"prob_overall": None, "n_same_bid": 0,
                "interpretation": "no 2026-1 data"}

    course_data["enrolled_bool"] = course_data["enrolled"].map(
        {"Y": 1, "N": 0}
    ).fillna(0)
    course_data["mileage_bid"] = pd.to_numeric(
        course_data["mileage_bid"], errors="coerce"
    )

    # Students who bid >= bid_amount
    above_threshold = course_data[course_data["mileage_bid"] >= bid_amount]
    at_exact_bid    = course_data[course_data["mileage_bid"] == bid_amount]

    # P(enrolled | bid >= X)
    if len(above_threshold) > 0:
        prob_above = above_threshold["enrolled_bool"].mean()
    else:
        prob_above = 0.0

    # P(enrolled | bid == X exactly) — shows tie-break effect
    if len(at_exact_bid) > 0:
        prob_exact = at_exact_bid["enrolled_bool"].mean()
        n_exact    = len(at_exact_bid)
        n_enr_exact= at_exact_bid["enrolled_bool"].sum()
    else:
        prob_exact  = 0.0
        n_exact     = 0
        n_enr_exact = 0

    # P(enrolled | bid >= X, year == student_year)
    prob_for_year = None
    if student_year:
        yr_data = above_threshold[
            above_threshold.get("grade_year", pd.Series()) == student_year
        ] if "grade_year" in above_threshold.columns else pd.DataFrame()
        if len(yr_data) > 0:
            prob_for_year = yr_data["enrolled_bool"].mean()

    # Interpretation
    if prob_above >= 0.95:
        interp = f"✅ Very safe — {prob_above:.0%} of students bidding ≥{bid_amount}pts enrolled"
    elif prob_above >= 0.75:
        interp = f"✓  Likely — {prob_above:.0%} of students bidding ≥{bid_amount}pts enrolled"
    elif prob_above >= 0.5:
        interp = f"⚠  Uncertain — only {prob_above:.0%} success rate at ≥{bid_amount}pts"
    else:
        interp = f"❌ Risky — only {prob_above:.0%} success at ≥{bid_amount}pts (bid too low)"

    return {
        "prob_above":    round(prob_above,    3),
        "prob_exact":    round(prob_exact,    3),
        "prob_for_year": round(prob_for_year, 3) if prob_for_year else None,
        "n_same_bid":    n_exact,
        "n_enrolled_same": int(n_enr_exact),
        "n_above_bid":   len(above_threshold),
        "interpretation": interp,
    }


def run():
    print("TEMPORAL HOLDOUT EVALUATION — 2026-1 GROUND TRUTH")
    print("=" * 60)

    if not Path(HOLDOUT_CSV).exists():
        print(f"No holdout data — run process_mileage_data.py first")
        return

    if not Path(HISTORY_CSV).exists():
        print(f"No history data — run scrape_mileage.py --merge-only first")
        return

    from model import load_model, predict_threshold
    from feature_extractor import load_features
    from optimizer import (
        CourseInput, allocate_bids,
        TOTAL_MILEAGE, MAX_BID_PER_COURSE
    )

    model, explainer  = load_model()
    df_features       = load_features("segmented_cs_courses.json")
    df_holdout        = pd.read_csv(HOLDOUT_CSV,  encoding="utf-8-sig")
    df_history        = pd.read_csv(HISTORY_CSV,  encoding="utf-8-sig")

    df_history["rank"]        = pd.to_numeric(df_history["rank"],        errors="coerce")
    df_history["mileage_bid"] = pd.to_numeric(df_history["mileage_bid"], errors="coerce")
    df_history["grade_year"]  = pd.to_numeric(df_history["grade_year"],  errors="coerce")

    df_holdout = df_holdout.dropna(subset=["winning_threshold"])
    print(f"Courses in holdout: {len(df_holdout)}")

    # ── Part 1: Threshold Prediction Accuracy ─────────────────────────────
    print(f"\n{'─'*60}")
    print("PART 1: THRESHOLD PREDICTION ACCURACY")
    print("How close were predictions to actual safe thresholds?")
    print(f"{'─'*60}")

    pred_rows = []
    for _, row in df_holdout.iterrows():
        code      = row["course_code"]
        actual_wt = float(row["winning_threshold"])

        if code not in df_features.index:
            continue

        feat = df_features.loc[code].to_dict()
        feat.update({
            "student_year":       3.0,
            "num_courses_wanted": 5.0,
            "rank_in_list":       1.0,
            "priority_ratio":     1.0,
        })

        pred, _ = predict_threshold(model, explainer, feat)
        error   = abs(pred - actual_wt)

        pred_rows.append({
            "course":      code,
            "professor":   row.get("professor", ""),
            "actual_wt":   actual_wt,
            "predicted_wt":round(pred, 1),
            "error_pts":   round(error, 1),
            "direction":   "over" if pred > actual_wt else "under",
        })

    df_pred = pd.DataFrame(pred_rows)
    mae     = df_pred["error_pts"].mean()
    median  = df_pred["error_pts"].median()
    w5      = (df_pred["error_pts"] <= 5).mean()  * 100
    w10     = (df_pred["error_pts"] <= 10).mean() * 100
    over    = (df_pred["direction"] == "over").sum()
    under   = (df_pred["direction"] == "under").sum()

    print(f"\nMAE:                {mae:.1f} pts")
    print(f"Median error:       {median:.1f} pts")
    print(f"Within 5pts:        {w5:.0f}%")
    print(f"Within 10pts:       {w10:.0f}%")
    print(f"Over-predicted:     {over}  |  Under-predicted: {under}")

    print(f"\nPer-course breakdown:")
    print(df_pred.sort_values("error_pts", ascending=False).to_string(index=False))

    # ── Part 2: Enrollment Probability Simulation ─────────────────────────
    print(f"\n{'─'*60}")
    print("PART 2: ENROLLMENT PROBABILITY SIMULATION")
    print("For the AI's recommended bid, what was actual enrollment rate?")
    print(f"{'─'*60}")

    sim_rows = []
    avg_pred = df_pred["predicted_wt"].mean() if len(df_pred) > 0 else 10.0

    for _, row in df_pred.iterrows():
        code    = row["course"]
        pred_wt = row["predicted_wt"]

        # Build 5-course input, this course as rank 1
        courses = [CourseInput(code, code, 1, pred_wt)]
        for j in range(2, 6):
            courses.append(CourseInput(f"f{j}", f"f{j}", j, avg_pred))

        ai_results = allocate_bids(
            courses, TOTAL_MILEAGE, MAX_BID_PER_COURSE
        )
        ai_bid = next(r.recommended_bid for r in ai_results if r.rank == 1)

        # Baselines
        uniform_bid  = TOTAL_MILEAGE // 5
        priority_bid = min(int(TOTAL_MILEAGE * 0.40), MAX_BID_PER_COURSE)
        allin_bid    = MAX_BID_PER_COURSE

        # Get actual enrollment probability for each bid
        ai_prob       = get_enrollment_probability(
            df_history, code, ai_bid
        )
        uniform_prob  = get_enrollment_probability(
            df_history, code, uniform_bid
        )
        priority_prob = get_enrollment_probability(
            df_history, code, priority_bid
        )
        allin_prob    = get_enrollment_probability(
            df_history, code, allin_bid
        )

        sim_rows.append({
            "course":         code,
            "actual_wt":      row["actual_wt"],
            "ai_bid":         ai_bid,
            "ai_prob":        ai_prob["prob_above"],
            "uniform_bid":    uniform_bid,
            "uniform_prob":   uniform_prob["prob_above"],
            "priority_bid":   priority_bid,
            "priority_prob":  priority_prob["prob_above"],
            "allin_bid":      allin_bid,
            "allin_prob":     allin_prob["prob_above"],
            "interpretation": ai_prob["interpretation"],
        })

    df_sim = pd.DataFrame(sim_rows)
    df_sim = df_sim.dropna(subset=["ai_prob"])

    if len(df_sim) > 0:
        print(f"\nCourses simulated: {len(df_sim)}")
        print(f"\n{'Strategy':<38} {'Avg Enroll Prob':>16}")
        print(f"{'─'*56}")
        print(f"{'AI Strategy':<38} {df_sim['ai_prob'].mean():>15.1%}")
        print(f"{'Uniform (14pts)':<38} {df_sim['uniform_prob'].mean():>15.1%}")
        print(f"{'Priority-weighted (28pts)':<38} {df_sim['priority_prob'].mean():>15.1%}")
        print(f"{'All-in (36pts)':<38} {df_sim['allin_prob'].mean():>15.1%}")

        # Competitive courses only
        comp = df_sim[df_sim["actual_wt"] >= 10]
        if len(comp) > 0:
            print(f"\nOn COMPETITIVE courses (threshold ≥ 10pts, n={len(comp)}):")
            print(f"  AI:      {comp['ai_prob'].mean():.1%}")
            print(f"  Uniform: {comp['uniform_prob'].mean():.1%}")
            print(f"  ← This is the headline metric for your paper")

        print(f"\nPer-course detail:")
        cols = ["course","actual_wt","ai_bid","ai_prob","uniform_bid",
                "uniform_prob","interpretation"]
        print(df_sim[cols].sort_values(
            "actual_wt", ascending=False
        ).to_string(index=False))

    # ── Part 3: Failure Analysis ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("PART 3: FAILURE ANALYSIS")
    print("Where does the model go wrong and why?")
    print(f"{'─'*60}")

    if len(df_pred) > 0:
        bad = df_pred[df_pred["error_pts"] >= 10].copy()
        if len(bad) > 0:
            print(f"\nCourses with error ≥ 10pts ({len(bad)} courses):")
            for _, r in bad.iterrows():
                direction = "predicted too HIGH" if r["direction"] == "over" else \
                            "predicted too LOW"
                print(f"\n  {r['course']} (Prof. {r['professor']})")
                print(f"    Actual threshold: {r['actual_wt']:.0f}pts")
                print(f"    Predicted:        {r['predicted_wt']:.0f}pts")
                print(f"    Error:            {r['error_pts']:.0f}pts — {direction}")

                if r["direction"] == "over":
                    print(f"    → Model over-estimated competition")
                    print(f"    → Student would overbid, wasting points")
                    print(f"    → Possible cause: no same-professor history, "
                          f"model using Tier 2 data from more competitive prof")
                else:
                    print(f"    → Model under-estimated competition")
                    print(f"    → Student might not get enrolled")
                    print(f"    → Possible cause: course became more popular "
                          f"than historical data suggests (reputation growth)")

    # ── Summary for paper ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("NUMBERS TO REPORT IN YOUR PAPER")
    print(f"{'='*60}")
    print(f"""
Evaluation Method: Temporal Holdout
  Train: 2022-1 to 2025-2 (professor-matched)
  Evaluate: 2026-1 actual results (ground truth)

Threshold Prediction (MAE):
  {mae:.1f} pts average error

Enrollment Probability (actual 2026-1 data):
  AI strategy:   {df_sim['ai_prob'].mean():.1%} average enrollment probability
  Uniform split: {df_sim['uniform_prob'].mean():.1%}
  All-in:        {df_sim['allin_prob'].mean():.1%}

Key insight: The enrollment probability accounts for tie-breaking.
A bid above the safe threshold has near-100% enrollment probability
regardless of the student's academic profile. Our model aims to
recommend bids above this threshold.

Limitations:
  - Tie-breaking within the same bid tier is unpredictable
  - Yearly quota allocation not visible in historical data
  - Course reputation changes not captured in limited history
""")


if __name__ == "__main__":
    run()
