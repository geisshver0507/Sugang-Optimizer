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


def parse_major_quota(val: str) -> tuple:
    """
    Parse is_major_incl_quota field:
    Y(Y) = major student, counted in major quota
    Y(N) = major student, NOT counted in major quota
    N(N) = non-major student
    Returns (is_major, in_major_quota)
    """
    if not isinstance(val, str):
        return False, False
    import re as _re
    m = _re.match(r"([YN])\(([YN])\)", str(val).strip())
    if m:
        return m.group(1) == "Y", m.group(2) == "Y"
    return False, False


def safe_threshold(enrolled: pd.DataFrame, rejected: pd.DataFrame) -> float:
    """
    Compute the safe winning threshold:
    = max bid among rejected students + 1
    If nobody rejected, minimum enrolled bid is enough.
    """
    if len(enrolled) == 0:
        return 0.0
    if len(rejected) > 0:
        max_rejected  = rejected["mileage_bid"].max()
        min_enrolled  = enrolled["mileage_bid"].min()
        return float(max(max_rejected + 1, min_enrolled))
    return float(enrolled["mileage_bid"].min())


def compute_stats(group: pd.DataFrame) -> dict:
    """
    Compute winning threshold and competition stats for one group.

    Accounts for:
    - Priority queue vs competitive seats (rank=0 vs rank>0)
    - Major quota students Y(Y) vs general Y(N)/N(N)
    - Per-year threshold for year-aware predictions
    - Safe threshold (max rejected + 1) to avoid tie-break zone
    """
    competitive   = group[group["rank"] > 0]
    enrolled_comp = competitive[competitive["enrolled_bool"] == 1]
    not_enr_comp  = competitive[competitive["enrolled_bool"] == 0]
    enrolled_all  = group[group["enrolled_bool"] == 1]
    priority_enr  = group[(group["rank"] == 0) & (group["enrolled_bool"] == 1)]

    # ── Overall safe threshold ────────────────────────────────────────────
    winning_threshold = safe_threshold(enrolled_comp, not_enr_comp)
    max_winning_bid   = float(enrolled_comp["mileage_bid"].max())                         if len(enrolled_comp) > 0 else 0.0
    avg_winning_bid   = float(enrolled_comp["mileage_bid"].mean())                         if len(enrolled_comp) > 0 else 0.0
    raw_min_bid       = float(enrolled_comp["mileage_bid"].min())                         if len(enrolled_comp) > 0 else 0.0

    # ── Major quota threshold (for CS major students) ─────────────────────
    # Y(Y) = major student counted in major quota → this is our target user
    major_col = None
    for c in group.columns:
        if "major_incl" in c or "전공자" in c:
            major_col = c
            break

    if major_col and len(competitive) > 0:
        # Parse major quota status
        competitive = competitive.copy()
        competitive[["is_major", "in_major_quota"]] = pd.DataFrame(
            competitive[major_col].apply(parse_major_quota).tolist(),
            index=competitive.index
        )
        major_enrolled  = competitive[
            competitive["in_major_quota"] & (competitive["enrolled_bool"] == 1)
        ]
        major_rejected  = competitive[
            competitive["in_major_quota"] & (competitive["enrolled_bool"] == 0)
        ]
        major_threshold = safe_threshold(major_enrolled, major_rejected)
    else:
        major_threshold = winning_threshold

    # ── Per-year thresholds ───────────────────────────────────────────────
    # Useful for year-aware recommendations
    yr_thresholds = {}
    if "grade_year" in competitive.columns and len(enrolled_comp) > 0:
        for yr in [1, 2, 3, 4]:
            yr_enr = enrolled_comp[enrolled_comp["grade_year"] == yr]
            yr_rej = not_enr_comp[not_enr_comp["grade_year"] == yr]                      if len(not_enr_comp) > 0 else pd.DataFrame()
            if len(yr_enr) > 0:
                yr_thresholds[yr] = safe_threshold(yr_enr, yr_rej)

    # ── Competition stats ─────────────────────────────────────────────────
    inferred_class_size = len(enrolled_all)
    priority_seats      = len(priority_enr)
    competitive_seats   = inferred_class_size - priority_seats
    total_applicants    = len(group.dropna(subset=["rank"]))
    competitive_apps    = len(competitive)
    acceptance_rate     = len(enrolled_comp) / max(competitive_apps, 1)
    competition_ratio   = competitive_apps   / max(competitive_seats, 1)

    yr_dist       = enrolled_comp.groupby("grade_year").size()                     if "grade_year" in enrolled_comp.columns else pd.Series()
    dominant_year = int(yr_dist.idxmax()) if len(yr_dist) > 0 else 0

    all_bids = competitive["mileage_bid"].dropna()
    bid_std  = float(all_bids.std())  if len(all_bids) > 1 else 0.0

    return {
        # Primary training target
        "winning_threshold":         winning_threshold,
        # Major quota specific (for CS major users)
        "major_quota_threshold":     major_threshold,
        # Per-year thresholds
        "threshold_yr1":             yr_thresholds.get(1, winning_threshold),
        "threshold_yr2":             yr_thresholds.get(2, winning_threshold),
        "threshold_yr3":             yr_thresholds.get(3, winning_threshold),
        "threshold_yr4":             yr_thresholds.get(4, winning_threshold),
        # Class size info
        "inferred_class_size":       inferred_class_size,
        "priority_seats_used":       priority_seats,
        "competitive_seats":         competitive_seats,
        # Demand info
        "total_applicants":          total_applicants,
        "competitive_applicants":    competitive_apps,
        "acceptance_rate":           round(acceptance_rate,  4),
        "competition_ratio":         round(competition_ratio,4),
        # Bid distribution
        "max_winning_bid":           max_winning_bid,
        "avg_winning_bid":           round(avg_winning_bid,  2),
        "avg_all_bids":              round(all_bids.mean(),  2)
                                     if len(all_bids) > 0 else 0.0,
        "bid_std":                   round(bid_std,          2),
        "raw_min_winning_bid":       raw_min_bid,
        # Year distribution
        "dominant_winner_year":      dominant_year,
        "pct_yr2_winners":           round(yr_dist.get(2, 0) /
                                     max(len(enrolled_comp), 1), 3),
        "pct_yr3_winners":           round(yr_dist.get(3, 0) /
                                     max(len(enrolled_comp), 1), 3),
        "pct_yr4_winners":           round(yr_dist.get(4, 0) /
                                     max(len(enrolled_comp), 1), 3),
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

        stats      = compute_stats(group)
        recency_w  = recency_map.get((int(year), int(semester)), 1.0)
        is_holdout = (int(year) == 2026 and int(semester) == 1)

        # Get all unique new CAS codes in this group
        # (multiple new codes can map to same old code+professor+semester)
        new_codes = group["course_code"].unique().tolist()

        # ── Create one row PER new code ───────────────────────────────────
        # Each new code gets its own tier based on its own 2026-1 professor
        # This fixes: CAS2103-01, CAS2103-02, CAS2103-03 all sharing one row
        for new_code in new_codes:

            if is_holdout:
                sample_weight = 0.0
                tier = 0
            else:
                prof_2026_for_code = prof_2026.get(new_code, "")

                if prof_norm and prof_2026_for_code and prof_norm == prof_2026_for_code:
                    # Tier 1: same professor as 2026-1
                    tier = 1
                    base_weight = 5.0
                elif prof_norm and prof_norm != "":
                    # Tier 2: different professor, same course
                    tier = 2
                    base_weight = 2.0
                else:
                    # No professor info
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
                "data_tier":        tier,
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