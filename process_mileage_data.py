"""
process_mileage_data.py
-----------------------
Cleans raw scraped data and produces training-ready rows.

KEY FIXES:
1. Uses course_code + professor as the matching key, not just course_code.
   Same course taught by different professors = different competition environments.

2. Splits data into:
   - train_data.csv     : all history EXCEPT 2026-1 (for model training)
   - holdout_2026_1.csv : 2026-1 only (ground truth for evaluation)

This enables proper temporal holdout evaluation:
   train on past → predict 2026-1 → compare against actual 2026-1 thresholds
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_CSV         = "mileage_history_all.csv"
SUMMARY_CSV     = "course_summary_all.csv"
TRAIN_CSV       = "mileage_training_real.csv"   # pre-2026-1 only
HOLDOUT_CSV     = "mileage_holdout_2026_1.csv"  # 2026-1 only (evaluation ground truth)
ANALYSIS_CSV    = "mileage_analysis.csv"


def parse_major_field(val):
    if not isinstance(val, str):
        return False, False
    m = re.match(r"([YN])\(([YN])\)", str(val).strip())
    if m:
        return m.group(1) == "Y", m.group(2) == "Y"
    return False, False


def normalize_professor(name: str) -> str:
    """Normalize professor name for matching — strip whitespace, lowercase."""
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace(" ", "")


def compute_course_semester_stats(group: pd.DataFrame) -> dict:
    """
    Given all student rows for one (course, professor, year, semester),
    compute the key stats needed for training.
    """
    # Separate priority queue (rank=0) from competitive (rank>0)
    competitive         = group[group["rank"] > 0]
    enrolled_comp       = competitive[competitive["enrolled_bool"] == 1]
    enrolled_all        = group[group["enrolled_bool"] == 1]
    priority_enrolled   = group[(group["rank"] == 0) & (group["enrolled_bool"] == 1)]

    # Winning threshold = min bid among competitive enrolled students
    if len(enrolled_comp) > 0:
        winning_threshold = enrolled_comp["mileage_bid"].min()
        max_winning_bid   = enrolled_comp["mileage_bid"].max()
        avg_winning_bid   = enrolled_comp["mileage_bid"].mean()
    else:
        # All seats filled via priority queue — no real mileage competition
        winning_threshold = 0.0
        max_winning_bid   = 0.0
        avg_winning_bid   = 0.0

    inferred_class_size  = len(enrolled_all)
    priority_seats_used  = len(priority_enrolled)
    competitive_seats    = inferred_class_size - priority_seats_used
    total_applicants     = len(group.dropna(subset=["rank"]))
    competitive_apps     = len(competitive)

    acceptance_rate  = len(enrolled_comp) / max(competitive_apps, 1)
    competition_ratio= competitive_apps  / max(competitive_seats, 1)

    # Year distribution of winners
    yr_dist      = enrolled_comp.groupby("grade_year").size()
    dominant_year= int(yr_dist.idxmax()) if len(yr_dist) > 0 else 0
    pct_yr2 = yr_dist.get(2, 0) / max(len(enrolled_comp), 1)
    pct_yr3 = yr_dist.get(3, 0) / max(len(enrolled_comp), 1)
    pct_yr4 = yr_dist.get(4, 0) / max(len(enrolled_comp), 1)

    all_bids = competitive["mileage_bid"].dropna()
    bid_std  = all_bids.std()  if len(all_bids) > 1 else 0.0
    bid_p25  = all_bids.quantile(0.25) if len(all_bids) > 0 else 0.0
    bid_p75  = all_bids.quantile(0.75) if len(all_bids) > 0 else 0.0

    return {
        "winning_threshold":       winning_threshold,
        "inferred_class_size":     inferred_class_size,
        "priority_seats_used":     priority_seats_used,
        "competitive_seats":       competitive_seats,
        "total_applicants":        total_applicants,
        "competitive_applicants":  competitive_apps,
        "acceptance_rate":         round(acceptance_rate,  4),
        "competition_ratio":       round(competition_ratio,4),
        "max_winning_bid":         max_winning_bid,
        "avg_winning_bid":         round(avg_winning_bid,  2),
        "avg_all_bids":            round(all_bids.mean(),  2) if len(all_bids) > 0 else 0.0,
        "bid_std":                 round(bid_std,          2),
        "bid_p25":                 bid_p25,
        "bid_p75":                 bid_p75,
        "dominant_winner_year":    dominant_year,
        "pct_winners_yr2":         round(pct_yr2, 3),
        "pct_winners_yr3":         round(pct_yr3, 3),
        "pct_winners_yr4":         round(pct_yr4, 3),
    }


def process():

    if not Path(RAW_CSV).exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found — run scrape_mileage.py first"
        )

    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    print(f"Raw rows: {len(df)}")

    # ── Clean types ───────────────────────────────────────────────────────
    df["rank"]        = pd.to_numeric(df["rank"],        errors="coerce")
    df["mileage_bid"] = pd.to_numeric(df["mileage_bid"], errors="coerce")
    df["grade_year"]  = pd.to_numeric(df["grade_year"],  errors="coerce")
    df["no"]          = pd.to_numeric(df["no"],          errors="coerce")
    df["enrolled_bool"] = df["enrolled"].map(
        {"Y": 1, "N": 0, "y": 1, "n": 0}
    ).fillna(0).astype(int)

    # ── Parse major field ─────────────────────────────────────────────────
    major_col = next(
        (c for c in df.columns if "전공자" in c or "major" in c.lower()), None
    )
    if major_col:
        parsed = df[major_col].apply(parse_major_field)
        df["is_major_student"] = parsed.apply(lambda x: int(x[0]))
        df["in_major_quota"]   = parsed.apply(lambda x: int(x[1]))

    # ── Merge professor from summary CSV ──────────────────────────────────
    # This is essential — professor determines competition level
    if Path(SUMMARY_CSV).exists():
        df_sum = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")

        # Normalize professor names for reliable matching
        if "professor" in df_sum.columns:
            df_sum["professor_norm"] = df_sum["professor"].apply(normalize_professor)

        merge_cols = ["course_code", "year", "semester"]
        extra_cols = [c for c in
                      ["professor", "professor_norm", "lecture_time",
                       "classroom", "capacity", "major_quota", "avg_mileage"]
                      if c in df_sum.columns]

        df = df.merge(
            df_sum[merge_cols + extra_cols],
            on=merge_cols, how="left"
        )
        print(f"Merged professor data from {SUMMARY_CSV}")

        # How many rows got professor info
        if "professor" in df.columns:
            matched = df["professor"].notna().sum()
            print(f"  Rows with professor info: {matched}/{len(df)}")
    else:
        print(f"Warning: {SUMMARY_CSV} not found — professor data unavailable")
        print("  Run scrape_mileage.py to collect summary data")
        df["professor"]      = ""
        df["professor_norm"] = ""

    # ── Group by course + professor + semester ────────────────────────────
    # This is the key fix: same course, different professor = different group
    group_cols = ["course_code", "year", "semester"]
    group_cols = [c for c in group_cols if c in df.columns]

    records_train   = []
    records_holdout = []

    for keys, group in df.groupby(group_cols):
        code, year, semester = keys
        # Professor taken from most common value in this group
        if "professor" in group.columns:
            prof_vals = group["professor"].dropna()
            prof_norm = prof_vals.mode()[0].strip().lower().replace(" ", "") if len(prof_vals) > 0 else ""
        else:
            prof_norm = ""

        stats = compute_course_semester_stats(group)

        row = {
            "course_code":   code,
            "course_name":   group["course_name"].iloc[0]
                             if "course_name" in group else code,
            "professor":     group["professor"].iloc[0]
                             if "professor" in group.columns else "",
            "professor_norm":prof_norm,
            "year":          year,
            "semester":      semester,
            **stats,
        }

        # Split into train vs holdout
        is_holdout = (int(year) == 2026 and int(semester) == 1)

        if is_holdout:
            records_holdout.append(row)
        else:
            records_train.append(row)

    # ── Save ──────────────────────────────────────────────────────────────
    df_train   = pd.DataFrame(records_train)
    df_holdout = pd.DataFrame(records_holdout)

    df_train.to_csv(TRAIN_CSV,   index=False, encoding="utf-8-sig")
    df_holdout.to_csv(HOLDOUT_CSV, index=False, encoding="utf-8-sig")

    # Combined analysis file
    df_all = pd.concat([df_train, df_holdout], ignore_index=True)
    df_all["split"] = df_all.apply(
        lambda r: "holdout_2026_1" if (r["year"]==2026 and r["semester"]==1)
                  else "train", axis=1
    )
    df_all.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ Training data:  {TRAIN_CSV}   ({len(df_train)} course-semester rows)")
    print(f"✅ Holdout data:   {HOLDOUT_CSV} ({len(df_holdout)} course-semester rows)")
    print(f"✅ Full analysis:  {ANALYSIS_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────
    if len(df_train) > 0:
        print(f"\n── Training set (pre-2026-1) threshold distribution ──")
        print(df_train["winning_threshold"].describe().round(1))

    if len(df_holdout) > 0:
        print(f"\n── Holdout set (2026-1) threshold distribution ──")
        print(df_holdout["winning_threshold"].describe().round(1))

        print(f"\n── 2026-1 courses and their actual thresholds ──")
        cols = ["course_code", "professor", "winning_threshold",
                "total_applicants", "competitive_seats", "competition_ratio"]
        cols = [c for c in cols if c in df_holdout.columns]
        print(df_holdout[cols].sort_values(
            "winning_threshold", ascending=False
        ).to_string(index=False))

    # ── Professor matching check ──────────────────────────────────────────
    if len(df_train) > 0 and "professor_norm" in df_train.columns:
        print(f"\n── Professor coverage check ──")
        print("Training rows per professor per course:")
        prof_count = df_train.groupby(
            ["course_code", "professor_norm"]
        ).size().reset_index(name="n_semesters")
        print(prof_count.to_string(index=False))

    return df_train, df_holdout


if __name__ == "__main__":
    process()