"""
evaluation_study.py
-------------------
Formal evaluation of the mileage strategy system.

Implements the three metrics requested by professors and TAs:
  1. Simulated enrollment success rate vs baselines (Prof. Kim)
  2. Advice acceptance rate tracking (TA Taewon Yoo)
  3. Intention alignment score (TA Yujin Gim)
  4. Backtesting simulation (no participants needed)

Run: python3 evaluation_study.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


HISTORY_CSV = "mileage_history_all.csv"
REAL_CSV    = "mileage_training_real.csv"

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: BACKTESTING SIMULATION
# Simulates what happens when students follow AI vs naive strategies
# across all courses in your historical dataset
# ══════════════════════════════════════════════════════════════════════════════

def run_backtesting():
    print("\n" + "═"*60)
    print("PART 1: BACKTESTING SIMULATION")
    print("Compares AI strategy vs 3 baselines across historical data")
    print("═"*60)

    if not Path(REAL_CSV).exists():
        print("No real data — run process_mileage_data.py first")
        return

    from model import load_model, predict_threshold
    from feature_extractor import load_features
    from optimizer import (
        CourseInput, allocate_bids,
        TOTAL_MILEAGE, MAX_BID_PER_COURSE
    )

    model, explainer = load_model()
    df_features      = load_features("segmented_cs_courses.json")
    df_real          = pd.read_csv(REAL_CSV, encoding="utf-8-sig")

    if "winning_threshold" not in df_real.columns:
        print("winning_threshold missing — run process_mileage_data.py")
        return

    df_real = df_real.dropna(subset=["winning_threshold"])

    # Simulate a student who wants 5 courses each semester
    # We test each course as if it were their rank-1 choice
    # paired with 4 filler courses of average difficulty
    N_COURSES = 5

    ai_results       = []
    uniform_results  = []
    priority_results = []
    allin_results    = []

    for _, row in df_real.iterrows():
        code      = row.get("course_code", "")
        actual_wt = float(row["winning_threshold"])

        if code not in df_features.index:
            continue

        # Build feature row
        feat = df_features.loc[code].to_dict()
        feat.update({
            "student_year":       3.0,
            "student_mileage":    float(TOTAL_MILEAGE),
            "num_courses_wanted": float(N_COURSES),
            "rank_in_list":       1.0,
            "priority_ratio":     1.0,
            "budget_ratio":       1.0,
        })

        pred, _ = predict_threshold(model, explainer, feat)

        # Build 5-course input (this course as rank 1 + 4 fillers)
        avg_threshold = df_real["winning_threshold"].mean()
        courses = [CourseInput(code, code, 1, pred)]
        for j in range(2, N_COURSES + 1):
            courses.append(CourseInput(
                f"filler_{j}", f"Filler {j}", j, avg_threshold
            ))

        # ── AI strategy ───────────────────────────────────────────────────
        ai_bids    = allocate_bids(courses, TOTAL_MILEAGE, MAX_BID_PER_COURSE)
        ai_bid_r1  = next(r.recommended_bid for r in ai_bids if r.rank == 1)
        ai_success = int(ai_bid_r1 >= actual_wt)

        # ── Baseline 1: Uniform split ─────────────────────────────────────
        # Student splits 72pts equally: 72/5 = 14pts each
        uniform_bid     = TOTAL_MILEAGE // N_COURSES
        uniform_success = int(uniform_bid >= actual_wt)

        # ── Baseline 2: Priority weighted ─────────────────────────────────
        # Student puts 40% on rank-1: 72 * 0.4 = 28pts
        priority_bid     = int(TOTAL_MILEAGE * 0.40)
        priority_success = int(priority_bid >= actual_wt)

        # ── Baseline 3: All-in on rank-1 ──────────────────────────────────
        # Student puts max allowed on rank-1: 36pts
        allin_bid     = MAX_BID_PER_COURSE
        allin_success = int(allin_bid >= actual_wt)

        ai_results.append(ai_success)
        uniform_results.append(uniform_success)
        priority_results.append(priority_success)
        allin_results.append(allin_success)

    if not ai_results:
        print("No results generated")
        return

    n = len(ai_results)

    def pct(lst): return np.mean(lst) * 100

    print(f"\nCourses simulated: {n}")
    print(f"\nSuccess rate (% of rank-1 course won if bid followed):")
    print(f"  {'Strategy':<30} {'Success Rate':>12}  {'vs AI':>8}")
    print(f"  {'─'*52}")
    print(f"  {'Our AI Strategy':<30} {pct(ai_results):>11.1f}%  {'—':>8}")
    print(f"  {'Baseline: Uniform (72/5=14pts)':<30} {pct(uniform_results):>11.1f}%  "
          f"{pct(ai_results)-pct(uniform_results):>+7.1f}%")
    print(f"  {'Baseline: Priority (40% = 28pts)':<30} {pct(priority_results):>11.1f}%  "
          f"{pct(ai_results)-pct(priority_results):>+7.1f}%")
    print(f"  {'Baseline: All-in (max 36pts)':<30} {pct(allin_results):>11.1f}%  "
          f"{pct(ai_results)-pct(allin_results):>+7.1f}%")

    print(f"\nInterpretation:")
    ai_rate = pct(ai_results)
    if ai_rate >= pct(uniform_results) and ai_rate >= pct(priority_results):
        print(f"  ✅ AI outperforms both naive baselines")
        print(f"  Use this as your main quantitative result in the paper")
    elif abs(ai_rate - max(pct(uniform_results), pct(priority_results))) < 5:
        print(f"  ✓  AI competitive with baselines (gap < 5%)")
        print(f"  Note: with median threshold=1pt, even 1pt bids win most courses")
        print(f"  The AI's value shows more clearly in high-competition courses")
    else:
        print(f"  ⚠  AI underperforms — more training data needed")

    # Show where AI specifically helps — high competition courses
    if Path(REAL_CSV).exists():
        df_real_tmp = pd.read_csv(REAL_CSV, encoding="utf-8-sig")
        df_real_tmp = df_real_tmp.dropna(subset=["winning_threshold"])
        high_comp = df_real_tmp[df_real_tmp["winning_threshold"] >= 10]
        if len(high_comp) > 0:
            print(f"\nHigh-competition courses (threshold ≥ 10pts): {len(high_comp)}")
            print("  These are where the AI matters most — it prevents underbidding")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: ADVICE ACCEPTANCE RATE TRACKER
# Use this during actual user study sessions
# Records whether participants follow AI recommendations
# ══════════════════════════════════════════════════════════════════════════════

def record_acceptance_session(
    participant_id:   str,
    ai_recommendations: list,   # list of {"course": str, "ai_bid": int}
    actual_bids:        list,   # list of {"course": str, "student_bid": int}
    output_csv:         str = "acceptance_log.csv",
):
    """
    Call this during user study after participant sees AI recommendations
    and decides their final bids.

    Example:
        record_acceptance_session(
            participant_id="P01",
            ai_recommendations=[
                {"course": "CAS3120-02", "ai_bid": 30},
                {"course": "CAS3205-01", "ai_bid": 28},
                {"course": "CAS2101-01", "ai_bid": 14},
            ],
            actual_bids=[
                {"course": "CAS3120-02", "student_bid": 30},   # accepted
                {"course": "CAS3205-01", "student_bid": 20},   # changed
                {"course": "CAS2101-01", "student_bid": 14},   # accepted
            ]
        )
    """
    rows = []
    for rec, act in zip(ai_recommendations, actual_bids):
        accepted = int(rec["ai_bid"] == act["student_bid"])
        rows.append({
            "participant_id": participant_id,
            "course":         rec["course"],
            "ai_bid":         rec["ai_bid"],
            "student_bid":    act["student_bid"],
            "accepted":       accepted,
            "delta":          act["student_bid"] - rec["ai_bid"],
        })

    df = pd.DataFrame(rows)
    path = Path(output_csv)

    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(path, index=False)
    acceptance_rate = df[df["participant_id"] == participant_id]["accepted"].mean()
    print(f"Participant {participant_id}: acceptance rate = {acceptance_rate:.0%}")
    return df


def summarize_acceptance(log_csv: str = "acceptance_log.csv"):
    """Summarize acceptance rates across all participants."""
    print("\n" + "═"*60)
    print("PART 2: ADVICE ACCEPTANCE RATE SUMMARY")
    print("═"*60)

    if not Path(log_csv).exists():
        print(f"No data yet — run user study and call record_acceptance_session()")
        print(f"\nTemplate for recording during sessions:")
        print("""
    from evaluation_study import record_acceptance_session

    # After participant P01 sees AI recommendations and decides their bids:
    record_acceptance_session(
        participant_id="P01",
        ai_recommendations=[
            {"course": "CAS3120-02", "ai_bid": 30},
            {"course": "CAS3205-01", "ai_bid": 28},
        ],
        actual_bids=[
            {"course": "CAS3120-02", "student_bid": 30},
            {"course": "CAS3205-01", "student_bid": 20},
        ]
    )
        """)
        return

    df = pd.read_csv(log_csv)
    n_participants  = df["participant_id"].nunique()
    overall_rate    = df["accepted"].mean()
    per_participant = df.groupby("participant_id")["accepted"].mean()

    print(f"\nParticipants: {n_participants}")
    print(f"Overall acceptance rate: {overall_rate:.1%}")
    print(f"\nPer participant:")
    for pid, rate in per_participant.items():
        print(f"  {pid}: {rate:.1%}")

    # When students changed bids, did they go up or down?
    changed = df[df["accepted"] == 0]
    if len(changed) > 0:
        went_up   = (changed["delta"] > 0).sum()
        went_down = (changed["delta"] < 0).sum()
        print(f"\nWhen recommendations were changed:")
        print(f"  Student bid HIGHER than AI: {went_up} times "
              f"(students felt AI was too conservative)")
        print(f"  Student bid LOWER than AI:  {went_down} times "
              f"(students felt AI was too aggressive)")

    print(f"\nFor your paper:")
    print(f"  Report overall acceptance rate: {overall_rate:.1%}")
    print(f"  Interpret: rates > 70% indicate students trust the system")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: INTENTION ALIGNMENT SCORE
# Measures whether AI output matches student's original priority intentions
# ══════════════════════════════════════════════════════════════════════════════

def compute_intention_alignment(
    student_priorities: list,  # [{"course": str, "rank": int}] — pre-study
    ai_output:          list,  # list[BidResult] — post-AI
    winning_thresholds: dict,  # {"course_code": threshold} — from real data
):
    """
    Checks whether the AI allocated enough points to secure the
    student's top-priority courses.

    Returns alignment score 0-1 where 1 = all top-priority courses
    got enough points to win.
    """
    top_courses = {
        item["course"] for item in student_priorities
        if item["rank"] <= 3
    }

    secured = 0
    total   = 0

    for result in ai_output:
        if result.code in top_courses:
            total += 1
            threshold = winning_thresholds.get(result.code, result.predicted_threshold)
            if result.recommended_bid >= threshold:
                secured += 1

    alignment = secured / max(total, 1)
    return alignment, secured, total


def run_alignment_analysis():
    """
    Runs alignment analysis using historical data to simulate
    whether the AI respects student priorities.
    """
    print("\n" + "═"*60)
    print("PART 3: INTENTION ALIGNMENT ANALYSIS")
    print("Measures: does AI output match student's original priorities?")
    print("═"*60)

    if not Path(REAL_CSV).exists():
        print("No real data found")
        return

    from model import load_model, predict_threshold
    from feature_extractor import load_features, flatten_json
    from optimizer import CourseInput, allocate_bids, TOTAL_MILEAGE, MAX_BID_PER_COURSE

    model, explainer = load_model()
    df_features      = load_features("segmented_cs_courses.json")
    df_real          = pd.read_csv(REAL_CSV, encoding="utf-8-sig")
    df_real          = df_real.dropna(subset=["winning_threshold"])

    # Build threshold lookup
    wt_lookup = df_real.set_index("course_code")["winning_threshold"].to_dict()

    # Test across all courses that have real thresholds
    valid_codes = [
        c for c in df_features.index if c in wt_lookup
    ]

    if len(valid_codes) < 3:
        print(f"Only {len(valid_codes)} courses with real thresholds — need at least 3")
        return

    alignment_scores = []

    # Simulate 20 different student priority orderings
    rng = np.random.default_rng(42)

    for sim in range(20):
        # Randomly pick 5 courses as a student's list
        chosen = rng.choice(valid_codes, size=min(5, len(valid_codes)), replace=False)
        ranks  = list(range(1, len(chosen) + 1))
        rng.shuffle(ranks)

        course_inputs = []
        for code, rank in zip(chosen, ranks):
            feat = df_features.loc[code].to_dict()
            feat.update({
                "student_year": 3.0,
                "student_mileage": float(TOTAL_MILEAGE),
                "num_courses_wanted": float(len(chosen)),
                "rank_in_list": float(rank),
                "priority_ratio": (len(chosen) - rank + 1) / len(chosen),
                "budget_ratio": 1.0,
            })
            pred, _ = predict_threshold(model, explainer, feat)
            course_inputs.append(CourseInput(code, code, rank, pred))

        results = allocate_bids(course_inputs, TOTAL_MILEAGE, MAX_BID_PER_COURSE)

        student_priorities = [
            {"course": code, "rank": rank}
            for code, rank in zip(chosen, ranks)
        ]

        score, secured, total = compute_intention_alignment(
            student_priorities, results, wt_lookup
        )
        alignment_scores.append(score)

    mean_alignment = np.mean(alignment_scores)
    print(f"\nSimulations run: 20")
    print(f"Mean alignment score: {mean_alignment:.1%}")
    print(f"  (% of top-3 priority courses that received enough points to win)")

    if mean_alignment >= 0.7:
        print(f"  ✅ Strong alignment — AI respects student priorities")
    elif mean_alignment >= 0.5:
        print(f"  ✓  Moderate alignment")
    else:
        print(f"  ⚠  Low alignment — budget constraint limits top-priority coverage")
        print(f"     This is expected when budgets are tight, not a system failure")

    print(f"\nFor your paper:")
    print(f"  Report mean alignment score: {mean_alignment:.1%}")
    print(f"  Compare with control group (manual allocation alignment)")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: USER STUDY SURVEY TEMPLATES
# Print these to use during your actual user study
# ══════════════════════════════════════════════════════════════════════════════

def print_survey_templates():
    print("\n" + "═"*60)
    print("PART 4: USER STUDY SURVEY TEMPLATES")
    print("═"*60)

    print("""
PRE-STUDY SURVEY (give to participant BEFORE using the system)
──────────────────────────────────────────────────────────────
Participant ID: _____    Year: _____    Date: _____

1. List the courses you want to register for this semester
   and your priority ranking (1 = most important):
   Rank 1: _________________ (Course code: _________)
   Rank 2: _________________ (Course code: _________)
   Rank 3: _________________ (Course code: _________)
   Rank 4: _________________ (Course code: _________)
   Rank 5: _________________ (Course code: _________)

2. How would you normally distribute your 72 mileage points?
   Course 1: ___pts  Course 2: ___pts  Course 3: ___pts
   Course 4: ___pts  Course 5: ___pts

3. How confident are you in this allocation? (circle one)
   1 (not at all)  2  3  4  5 (very confident)

4. How long did you spend deciding? _____ minutes

5. How familiar are you with the mileage system?
   1 (never used)  2  3  4  5 (used many times)


POST-STUDY SURVEY (give AFTER participant uses the system)
──────────────────────────────────────────────────────────
Participant ID: _____

1. Did you follow the AI's recommended bids?
   For each course, circle: FOLLOWED / CHANGED
   Course 1: FOLLOWED / CHANGED  (if changed, new bid: ___)
   Course 2: FOLLOWED / CHANGED  (if changed, new bid: ___)
   Course 3: FOLLOWED / CHANGED  (if changed, new bid: ___)
   Course 4: FOLLOWED / CHANGED  (if changed, new bid: ___)
   Course 5: FOLLOWED / CHANGED  (if changed, new bid: ___)

2. If you changed any bids, why? (circle all that apply)
   [ ] AI bid seemed too high   [ ] AI bid seemed too low
   [ ] I know this course better [ ] I didn't trust the AI
   [ ] Other: ___________________

3. How confident are you NOW in your allocation? (circle one)
   1 (not at all)  2  3  4  5 (very confident)

4. Did the AI's recommendations match what you originally wanted?
   1 (not at all)  2  3  4  5 (perfectly)

5. How long did the full AI-assisted process take? _____ minutes

6. Would you use this system for real registration? YES / NO

7. Overall satisfaction with the system: 1  2  3  4  5


CONTROL GROUP SURVEY (for non-AI group comparison)
───────────────────────────────────────────────────
Same as pre-study survey above. Additionally:

8. Were you confident in your manual allocation? 1  2  3  4  5

9. Did you feel you had enough information to decide? 1  2  3  4  5
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("FORMAL EVALUATION — MILEAGE STRATEGY SYSTEM")
    print("Addressing feedback from Prof. Kim, TA Taewon Yoo, TA Yujin Gim")

    run_backtesting()
    summarize_acceptance()
    run_alignment_analysis()
    print_survey_templates()

    print("\n" + "═"*60)
    print("SUMMARY FOR YOUR PAPER")
    print("═"*60)
    print("""
Report these three metrics:

1. Backtesting Success Rate (Prof. Kim — baseline comparison)
   "Our AI strategy achieved X% enrollment success rate on
   historical data, compared to Y% for uniform splitting and
   Z% for priority weighting."

2. Advice Acceptance Rate (TA Taewon Yoo — trust measure)
   "X% of participants followed the AI's recommendations
   without modification, indicating [high/moderate] trust."

3. Intention Alignment Score (TA Yujin Gim — relevance measure)
   "The system allocated sufficient points to secure X% of
   students' top-3 priority courses, demonstrating that the
   AI's optimization respects student preferences."

Plus qualitative: pre/post confidence delta and satisfaction scores.
""")
