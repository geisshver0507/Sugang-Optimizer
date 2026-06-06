"""
process_mileage_data.py
-----------------------
Cleans scraped data and produces training rows.

THREE-TIER DATA QUALITY:
  Tier 1: same new CAS code + same professor  → sample_weight = 5.0
  Tier 2: same base course + diff professor   → sample_weight = 2.0
  Tier 3: no real data at all                 → synthetic disabled

GROUPING:
  Groups by (base_course_code, professor_norm, year, semester)
  NOT by section number — so if Professor Park taught CCO2101-03 in 2022
  (not in mapping) but we got it via CCO2101-01 scrape, it still counts
  as long as the professor matches.

RECENCY WEIGHTING:
  More recent semesters get higher weight.
  2022-1 → 0.5x,  2025-2 → 1.0x
  Applied multiplicatively on top of tier weight.

TEMPORAL SPLIT:
  Pre-2026-1  → mileage_training_real.csv   (training)
  2026-1      → mileage_holdout_2026_1.csv  (evaluation ground truth)
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_CSV      = "mileage_history_all.csv"
TRAIN_CSV    = "mileage_training_real.csv"
HOLDOUT_CSV  = "mileage_holdout_2026_1.csv"
ANALYSIS_CSV = "mileage_analysis.csv"


def normalize_professor(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace(" ", "")


def get_base_code(code: str) -> str:
    """'CAS3116-01' → 'CAS3116'"""
    return str(code).strip().split("-")[0]


def parse_major_field(val):
    if not isinstance(val, str):
        return False, False
    m = re.match(r"([YN])\(([YN])\)", str(val).strip())
    if m:
        return m.group(1) == "Y", m.group(2) == "Y"
    return False, False


def compute_stats(group: pd.DataFrame) -> dict:
    """Compute winning threshold and competition stats for one group."""
    competitive     = group[group["rank"] > 0]
    enrolled_comp   = competitive[competitive["enrolled_bool"] == 1]
    enrolled_all    = group[group["enrolled_bool"] == 1]
    priority_enr    = group[(group["rank"] == 0) & (group["enrolled_bool"] == 1)]

    if len(enrolled_comp) > 0:
        winning_threshold = enrolled_comp["mileage_bid"].min()
        max_winning_bid   = enrolled_comp["mileage_bid"].max()
        avg_winning_bid   = enrolled_comp["mileage_bid"].mean()
    else:
        winning_threshold = 0.0
        max_winning_bid   = 0.0
        avg_winning_bid   = 0.0

    inferred_class_size = len(enrolled_all)
    priority_seats      = len(priority_enr)
    competitive_seats   = inferred_class_size - priority_seats
    total_applicants    = len(group.dropna(subset=["rank"]))
    competitive_apps    = len(competitive)
    acceptance_rate     = len(enrolled_comp) / max(competitive_apps, 1)
    competition_ratio   = competitive_apps  / max(competitive_seats, 1)

    yr_dist       = enrolled_comp.groupby("grade_year").size()
    dominant_year = int(yr_dist.idxmax()) if len(yr_dist) > 0 else 0

    all_bids = competitive["mileage_bid"].dropna()
    bid_std  = all_bids.std()  if len(all_bids) > 1 else 0.0

    return {
        "winning_threshold":      winning_threshold,
        "inferred_class_size":    inferred_class_size,
        "priority_seats_used":    priority_seats,
        "competitive_seats":      competitive_seats,
        "total_applicants":       total_applicants,
        "competitive_applicants": competitive_apps,
        "acceptance_rate":        round(acceptance_rate,  4),
        "competition_ratio":      round(competition_ratio,4),
        "max_winning_bid":        max_winning_bid,
        "avg_winning_bid":        round(avg_winning_bid,  2),
        "avg_all_bids":           round(all_bids.mean(),  2)
                                  if len(all_bids) > 0 else 0.0,
        "bid_std":                round(bid_std,          2),
        "dominant_winner_year":   dominant_year,
    }


def process():
    if not Path(RAW_CSV).exists():
        raise FileNotFoundError(f"{RAW_CSV} not found — run scrape_mileage.py")

    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    print(f"Raw rows loaded: {len(df)}")

    # ── Clean types ───────────────────────────────────────────────────────
    df["rank"]        = pd.to_numeric(df["rank"],        errors="coerce")
    df["mileage_bid"] = pd.to_numeric(df["mileage_bid"], errors="coerce")
    df["grade_year"]  = pd.to_numeric(df["grade_year"],  errors="coerce")
    df["enrolled_bool"] = df["enrolled"].map(
        {"Y": 1, "N": 0, "y": 1, "n": 0}
    ).fillna(0).astype(int)

    # Normalize professor names
    if "professor" not in df.columns:
        df["professor"]      = ""
        df["professor_norm"] = ""
    else:
        df["professor_norm"] = df["professor"].apply(normalize_professor)

    # Base course code (without section)
    df["base_code"] = df["course_code"].apply(get_base_code)

    # ── Recency weights ───────────────────────────────────────────────────
    all_sems = sorted(
        df[["year","semester"]].drop_duplicates().values.tolist()
    )
    n_sems = len(all_sems)
    recency_map = {
        (int(y), int(s)): 0.5 + 0.5 * (i / max(n_sems - 1, 1))
        for i, (y, s) in enumerate(all_sems)
    }
    print(f"Semesters found: {[f'{y}-{s}' for y,s in all_sems]}")

    # ── Get 2026-1 professor per new course code ──────────────────────────
    holdout_rows = df[(df["year"] == 2026) & (df["semester"] == 1)]
    prof_2026 = {}
    if len(holdout_rows) > 0:
        for code, grp in holdout_rows.groupby("course_code"):
            profs = grp["professor_norm"].dropna()
            profs = profs[profs != ""]
            if len(profs) > 0:
                prof_2026[code] = profs.mode()[0]
    print(f"2026-1 professor info: {len(prof_2026)} courses")

    # ── Group by base_code + professor_norm + year + semester ─────────────
    # Using base_code (not section) means Professor Park's data from
    # CCO2101-03 (not in mapping) is still captured if scraped
    group_cols = ["base_code", "professor_norm", "year", "semester"]

    records_train   = []
    records_holdout = []

    for keys, group in df.groupby(group_cols):
        base_code, prof_norm, year, semester = keys

        # Get the new CAS course code for this base code
        new_codes = group["course_code"].unique()
        new_code  = new_codes[0]  # primary new code

        stats      = compute_stats(group)
        recency_w  = recency_map.get((int(year), int(semester)), 1.0)
        is_holdout = (int(year) == 2026 and int(semester) == 1)

        # ── Three-tier sample weight ──────────────────────────────────────
        if is_holdout:
            sample_weight = 0.0  # holdout never used in training
        else:
            # Tier 1: same new code + same professor as 2026-1
            prof_2026_for_code = prof_2026.get(new_code, "")
            if prof_norm and prof_2026_for_code and prof_norm == prof_2026_for_code:
                tier = 1
                base_weight = 5.0
            # Tier 2: same base course, different professor
            elif prof_norm != prof_2026_for_code and prof_norm != "":
                tier = 2
                base_weight = 2.0
            # No professor info — keep but weight lower
            else:
                tier = 2
                base_weight = 1.5

            sample_weight = base_weight * recency_w

        row = {
            "course_code":      new_code,
            "base_code":        base_code,
            "course_name":      group["course_name"].iloc[0]
                                if "course_name" in group else new_code,
            "professor":        group["professor"].iloc[0]
                                if "professor" in group.columns else "",
            "professor_norm":   prof_norm,
            "year":             year,
            "semester":         semester,
            "recency_weight":   recency_w,
            "sample_weight":    sample_weight,
            "data_tier":        tier if not is_holdout else 0,
            **stats,
        }

        if is_holdout:
            records_holdout.append(row)
        else:
            records_train.append(row)

    df_train   = pd.DataFrame(records_train)   if records_train   else pd.DataFrame()
    df_holdout = pd.DataFrame(records_holdout) if records_holdout else pd.DataFrame()

    # ── Save ──────────────────────────────────────────────────────────────
    if not df_train.empty:
        df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    else:
        # Write empty file with header to avoid EmptyDataError
        pd.DataFrame(columns=["course_code","winning_threshold"]).to_csv(
            TRAIN_CSV, index=False, encoding="utf-8-sig"
        )

    df_holdout.to_csv(HOLDOUT_CSV, index=False, encoding="utf-8-sig")

    df_all = pd.concat([df_train, df_holdout], ignore_index=True) \
             if not df_train.empty else df_holdout
    df_all.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ Training:  {TRAIN_CSV}  ({len(df_train)} rows)")
    print(f"✅ Holdout:   {HOLDOUT_CSV} ({len(df_holdout)} rows)")

    if not df_train.empty:
        print(f"\n── Training data breakdown ──")
        if "data_tier" in df_train.columns:
            tier_counts = df_train["data_tier"].value_counts().sort_index()
            for tier, cnt in tier_counts.items():
                label = {1:"Tier 1 (same prof)",
                         2:"Tier 2 (diff prof/unknown)"}.get(tier, f"Tier {tier}")
                print(f"  {label}: {cnt} rows")
        print(f"\n── Winning threshold distribution (training) ──")
        print(df_train["winning_threshold"].describe().round(1))

    if not df_holdout.empty:
        print(f"\n── 2026-1 holdout courses ──")
        cols = ["course_code","professor","winning_threshold",
                "total_applicants","competition_ratio"]
        cols = [c for c in cols if c in df_holdout.columns]
        print(df_holdout[cols].sort_values(
            "winning_threshold", ascending=False
        ).to_string(index=False))

    return df_train, df_holdout


if __name__ == "__main__":
    process()