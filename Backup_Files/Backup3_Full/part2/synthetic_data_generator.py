"""
synthetic_data_generator.py
----------------------------
Generates synthetic training rows ONLY for courses with no real data.

TARGET: winning_threshold = minimum bid the last enrolled student placed.
        This is what we actually want to predict — the competition threshold,
        not an individual student's bid.

The synthetic generation uses course features to create plausible
competition scenarios. The XGBoost model then learns which features
actually correlate with high thresholds from the REAL data rows,
and applies that learning to synthetic rows for unseen courses.

The formula here is NOT the model — it's just a way to generate
plausible training examples. The model finds its own weights.
"""

import numpy as np
import pandas as pd
from feature_extractor import load_features

# These are the exact feature columns the model trains on
FEATURE_COLS = [
    # Course-level features (from segmented_cs_courses.json)
    "eta_added",            # how many people added on ETA — direct demand signal
    "max_capacity",         # physical/applicant capacity from holdout enrichment
    "demand_proxy",         # historical num_applied when known, otherwise ETA/cart adds
    "demand_capacity_ratio", # demand pressure relative to seats
    "capacity_demand_gap",  # positive means demand exceeds available seats
    "credits",
    "major_year_target",    # which year the course is aimed at
    "is_major_req",         # mandatory = more seats usually
    "is_major_basic",
    "is_major_elective",    # popular electives = high competition
    "num_prerequisites",    # more prereqs = fewer eligible students
    "difficulty_score",     # 1=easy, 2=medium, 3=hard
    "workload_score",
    "is_relative_grading",  # relative grading scares people off
    "exam_weight",          # how exam-heavy is grading
    "assignment_weight",
    "is_english",
    "lecture_type_score",   # 0=offline, 1=blended, 2=online
    "earliest_period",      # time slot — lower = earlier in day
    "review_score",         # professor rating 1-5
    "has_review",           # whether any review exists
    # Student-level features (provided at prediction time)
    "student_year",         # student's year (1-4)
    "num_courses_wanted",   # how many courses in their list
    "rank_in_list",         # priority rank of this course (1=most wanted)
    "priority_ratio",       # (n - rank + 1) / n
]

TARGET_COL = "winning_threshold"


def generate_synthetic_bids(
    df_features: pd.DataFrame,
    n_samples_per_course: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each course in df_features, generate n_samples_per_course
    synthetic training rows simulating different student scenarios.

    Each row represents a plausible (features → winning_threshold) pair.
    The winning_threshold here is a course-level property for that
    simulated scenario — not an individual student bid.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for code, feat in df_features.iterrows():

        for _ in range(n_samples_per_course):

            # Simulate a student context
            student_year       = float(rng.integers(1, 5))
            student_mileage    = 76.0   # fixed per Yonsei rules
            num_courses_wanted = float(rng.integers(3, 8))
            rank_in_list       = float(rng.integers(1, int(num_courses_wanted) + 1))
            priority_ratio     = (num_courses_wanted - rank_in_list + 1) / num_courses_wanted
            budget_ratio       = 76.0 / 76.0   # always 1.0 (fixed budget)

            # ── Estimate competition pressure for this course ──────────────
            # This generates a plausible competition score.
            # Noise is added so the model must find signal rather than
            # memorising a deterministic formula.
            noise = lambda scale=0.25: rng.normal(1.0, scale)

            comp = (
                (feat["eta_added"] / 10.0)             * noise()   # raw demand
                + feat.get("demand_capacity_ratio", 0.0) * 5.0     * noise()
                - (feat.get("max_capacity", 0.0) / 100.0)          * noise()
                + feat["review_score"]       * 1.5     * noise()   # good prof = more demand
                + feat["is_major_elective"]  * 4.0     * noise()   # popular electives fought over
                - feat["is_major_req"]       * 1.0     * noise()   # req = more seats usually
                + feat["difficulty_score"]   * 0.4     * noise()   # harder = some extra demand
                - feat["num_prerequisites"]  * 1.0     * noise()   # prereqs filter out students
                + feat["lecture_type_score"] * 1.2     * noise()   # online = more applicants
                - feat["is_relative_grading"]* 0.6     * noise()   # relative grading deters some
                - feat["earliest_period"]    * 0.1     * noise()   # earlier slot = slightly less demand
                + feat["is_english"]         * 0.3     * noise()
            )

            # Year gap: if course targets year 3 and student is year 1,
            # fewer year-1 students are eligible → less competition for them
            year_gap = max(0.0, feat["major_year_target"] - student_year)
            comp -= year_gap * 0.4 * noise()

            comp = max(0.0, comp)

            # ── Map competition score → winning threshold ──────────────────
            # winning_threshold = minimum bid that secured a seat
            # Scale: comp → roughly 0-150 pts range
            threshold = float(np.clip(
                comp * 10.0 + rng.normal(0, 6),
                1.0,
                76.0   # max is the full mileage budget
            ))

            row = feat.to_dict()
            row["course_code"]        = code
            row["student_year"]       = student_year
            row["student_mileage"]    = 76.0
            row["num_courses_wanted"] = num_courses_wanted
            row["rank_in_list"]       = rank_in_list
            row["priority_ratio"]     = priority_ratio
            row["budget_ratio"]       = budget_ratio
            row[TARGET_COL]           = threshold
            row["data_source"]        = "synthetic"
            rows.append(row)

    return pd.DataFrame(rows)


def get_training_data(json_path: str, n_samples: int = 40) -> pd.DataFrame:
    df_features = load_features(json_path)
    return generate_synthetic_bids(df_features, n_samples)


if __name__ == "__main__":
    df = get_training_data("segmented_cs_courses.json")
    print(df[[TARGET_COL]].describe().round(2))
    print(f"Total synthetic rows: {len(df)}")
