"""
process_mileage_data.py
-----------------------
Cleans raw scraped CSVs and computes training targets.

KEY INSIGHT from data inspection:
  rank=0 rows = students who got in via PRIORITY QUEUE (major quota seats)
                NOT through mileage competition
  rank>0 rows = students who competed via mileage bidding

So winning_threshold must be computed only from rank>0 enrolled rows,
otherwise we'd think "12 pts won" when actually 20 pts was the real cutoff.

OUTPUTS:
  mileage_training_real.csv   → one row per course-semester, ready for model
  mileage_analysis.csv        → human-readable summary per course
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_CSV      = "mileage_history_all.csv"
SUMMARY_CSV  = "course_summary_all.csv"
OUTPUT_CSV   = "mileage_training_real.csv"
ANALYSIS_CSV = "mileage_analysis.csv"


def parse_major_field(val):
    """
    'Y(Y)' → is_major=True,  in_major_quota=True
    'Y(N)' → is_major=True,  in_major_quota=False
    'N(N)' → is_major=False, in_major_quota=False
    """
    import re
    if not isinstance(val, str):
        return False, False
    m = re.match(r"([YN])\(([YN])\)", str(val).strip())
    if m:
        return m.group(1) == "Y", m.group(2) == "Y"
    return False, False


def process(raw_csv: str = RAW_CSV) -> pd.DataFrame:

    if not Path(raw_csv).exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run scrape_mileage.py first."
        )

    df = pd.read_csv(raw_csv, encoding="utf-8-sig")
    print(f"Raw rows loaded: {len(df)}")
    print(f"Columns: {df.columns.tolist()}\n")

    # ── Clean types ───────────────────────────────────────────────────────
    df["rank"]         = pd.to_numeric(df["rank"],         errors="coerce")
    df["mileage_bid"]  = pd.to_numeric(df["mileage_bid"],  errors="coerce")
    df["grade_year"]   = pd.to_numeric(df["grade_year"],   errors="coerce")
    df["no"]           = pd.to_numeric(df["no"],           errors="coerce")
    df["num_courses_applied"] = pd.to_numeric(
        df.get("num_courses_applied", pd.Series()), errors="coerce"
    )

    for col in ["total_credit_ratio", "prev_semester_credit_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # enrolled: Y=1, N or blank=0
    df["enrolled_bool"] = df["enrolled"].map(
        {"Y": 1, "N": 0, "y": 1, "n": 0}
    ).fillna(0).astype(int)

    # ── Parse major field ─────────────────────────────────────────────────
    major_col = None
    for c in df.columns:
        if "전공자" in c or "major" in c.lower():
            major_col = c
            break

    if major_col:
        parsed = df[major_col].apply(parse_major_field)
        df["is_major_student"]    = parsed.apply(lambda x: int(x[0]))
        df["in_major_quota"]      = parsed.apply(lambda x: int(x[1]))

    # ── Separate priority-queue vs competitive enrollments ────────────────
    # rank=0 → got in through major quota priority, not mileage competition
    # rank>0 → competed via mileage bidding
    df["via_priority_queue"]   = (df["rank"] == 0).astype(int)
    df["via_mileage_bidding"]  = (df["rank"] > 0).astype(int)

    # Competitive enrolled = enrolled AND went through mileage competition
    df["competitive_enrolled"] = (
        (df["enrolled_bool"] == 1) & (df["rank"] > 0)
    ).astype(int)

    # ── Compute per course-semester stats ─────────────────────────────────
    records = []

    for (code, year, semester), group in df.groupby(
        ["course_code", "year", "semester"]
    ):
        competitive = group[group["via_mileage_bidding"] == 1]
        enrolled_competitive = competitive[competitive["enrolled_bool"] == 1]
        enrolled_all = group[group["enrolled_bool"] == 1]

        # ── Winning threshold ─────────────────────────────────────────────
        # Minimum bid among COMPETITIVE (rank>0) enrolled students
        # This is what a student actually needs to bid to get in
        if len(enrolled_competitive) > 0:
            winning_threshold = enrolled_competitive["mileage_bid"].min()
            max_winning_bid   = enrolled_competitive["mileage_bid"].max()
            avg_winning_bid   = enrolled_competitive["mileage_bid"].mean()
        else:
            # All seats filled via priority queue — no mileage competition
            winning_threshold = 0.0
            max_winning_bid   = 0.0
            avg_winning_bid   = 0.0

        # ── Class size (inferred from enrolled count) ─────────────────────
        inferred_class_size = len(enrolled_all)

        # Priority queue seats (rank=0 enrolled)
        priority_seats_used = len(group[
            (group["via_priority_queue"] == 1) & (group["enrolled_bool"] == 1)
        ])

        # Competitive seats available = total enrolled - priority queue
        competitive_seats = inferred_class_size - priority_seats_used

        # ── Competition metrics ───────────────────────────────────────────
        total_applicants        = len(group.dropna(subset=["rank"]))
        competitive_applicants  = len(competitive)
        acceptance_rate         = (
            len(enrolled_competitive) / max(competitive_applicants, 1)
        )
        competition_ratio       = (
            competitive_applicants / max(competitive_seats, 1)
        )

        # ── Year distribution of winners ──────────────────────────────────
        yr_dist = enrolled_competitive.groupby("grade_year").size()
        dominant_year = int(yr_dist.idxmax()) if len(yr_dist) > 0 else 0
        pct_yr2 = yr_dist.get(2, 0) / max(len(enrolled_competitive), 1)
        pct_yr3 = yr_dist.get(3, 0) / max(len(enrolled_competitive), 1)
        pct_yr4 = yr_dist.get(4, 0) / max(len(enrolled_competitive), 1)

        # ── Bid distribution stats ────────────────────────────────────────
        all_bids = competitive["mileage_bid"].dropna()
        bid_std  = all_bids.std() if len(all_bids) > 1 else 0.0
        bid_p25  = all_bids.quantile(0.25) if len(all_bids) > 0 else 0.0
        bid_p75  = all_bids.quantile(0.75) if len(all_bids) > 0 else 0.0

        records.append({
            "course_code":              code,
            "course_name":              group["course_name"].iloc[0]
                                        if "course_name" in group else code,
            "year":                     year,
            "semester":                 semester,

            # ── The training target ───────────────────────────────────────
            "winning_threshold":        winning_threshold,

            # ── Class size info ───────────────────────────────────────────
            "inferred_class_size":      inferred_class_size,
            "priority_seats_used":      priority_seats_used,
            "competitive_seats":        competitive_seats,

            # ── Demand / competition ──────────────────────────────────────
            "total_applicants":         total_applicants,
            "competitive_applicants":   competitive_applicants,
            "acceptance_rate":          round(acceptance_rate, 4),
            "competition_ratio":        round(competition_ratio, 4),

            # ── Bid stats ─────────────────────────────────────────────────
            "max_winning_bid":          max_winning_bid,
            "avg_winning_bid":          round(avg_winning_bid, 2),
            "avg_all_bids":             round(all_bids.mean(), 2)
                                        if len(all_bids) > 0 else 0.0,
            "bid_std":                  round(bid_std, 2),
            "bid_p25":                  bid_p25,
            "bid_p75":                  bid_p75,

            # ── Year distribution of winners ──────────────────────────────
            "dominant_winner_year":     dominant_year,
            "pct_winners_yr2":          round(pct_yr2, 3),
            "pct_winners_yr3":          round(pct_yr3, 3),
            "pct_winners_yr4":          round(pct_yr4, 3),
        })

    df_out = pd.DataFrame(records)

    # ── Merge professor + capacity from summary CSV ───────────────────────
    if Path(SUMMARY_CSV).exists():
        df_sum = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
        merge_cols = ["course_code", "year", "semester"]
        extra_cols = [c for c in
                      ["professor", "lecture_time", "classroom",
                       "capacity", "major_quota", "avg_mileage"]
                      if c in df_sum.columns]
        if extra_cols:
            df_out = df_out.merge(
                df_sum[merge_cols + extra_cols],
                on=merge_cols, how="left"
            )
            print(f"Merged summary data: {extra_cols}")
    else:
        print(f"Note: {SUMMARY_CSV} not found — professor/capacity not merged")
        print("Run scrape_mileage.py to get summary data")

    # ── Save ──────────────────────────────────────────────────────────────
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Training data: {OUTPUT_CSV}  ({len(df_out)} course-semester rows)")

    # ── Analysis CSV (human-readable) ─────────────────────────────────────
    df_out.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Analysis:      {ANALYSIS_CSV}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n── Winning thresholds across courses ──")
    print(df_out["winning_threshold"].describe().round(1))

    print("\n── Top 10 most competitive (highest winning threshold) ──")
    top = df_out.nlargest(10, "winning_threshold")[
        ["course_code", "course_name", "year", "semester",
         "winning_threshold", "competitive_applicants",
         "competitive_seats", "competition_ratio"]
    ]
    print(top.to_string(index=False))

    print("\n── Courses with zero mileage competition (all priority queue) ──")
    zero = df_out[df_out["winning_threshold"] == 0][
        ["course_code", "year", "semester", "priority_seats_used"]
    ]
    print(zero.to_string(index=False) if len(zero) > 0 else "None")

    return df_out


if __name__ == "__main__":
    process()
