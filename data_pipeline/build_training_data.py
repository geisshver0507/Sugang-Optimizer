"""
build_training_data.py
----------------------
Builds the final training set from professor-matched historical data.

Uses TRAIN_CSV (pre-2026-1 only) as real data.
Generates synthetic rows only for courses with zero professor-matched history.

The 2026-1 holdout is NOT touched here — it's reserved for evaluation only.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from feature_extractor import load_features, flatten_json
from synthetic_data_generator import (
    generate_synthetic_bids, FEATURE_COLS, TARGET_COL
)

ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = ROOT / "database"
TRAIN_CSV = str(DATABASE_DIR / "mileage_training_real.csv")   # pre-2026-1, professor-matched
JSON_PATH = str(DATABASE_DIR / "segmented_cs_courses.json")
OUTPUT_CSV = str(DATABASE_DIR / "training_final.csv")


def get_2026_professors(json_path: str) -> dict:
    """
    Get the professor for each 2026-1 course.
    Uses the scraped holdout CSV as the source of truth since it has
    actual professor names from the mileage helper site.
    Falls back to JSON if holdout not available.
    Returns {course_code: professor_norm}
    """
    import pandas as pd
    from pathlib import Path

    holdout_path = DATABASE_DIR / "mileage_holdout_2026_1.csv"
    if holdout_path.exists():
        try:
            df = pd.read_csv(holdout_path, encoding="utf-8-sig")
            if "professor_norm" in df.columns and "course_code" in df.columns:
                result = {}
                for _, row in df.iterrows():
                    code = row["course_code"]
                    prof = str(row.get("professor_norm", "")).strip()
                    if code and prof and prof != "nan":
                        result[code] = prof
                if result:
                    return result
        except Exception:
            pass

    # Fallback: try JSON
    flat = flatten_json(json_path)
    result = {}
    for code, c in flat.items():
        prof = c.get("professor") or c.get("metadata", {}).get("professor", "")
        if prof:
            result[code] = prof.strip().lower().replace(" ", "")
    return result


def build(
    train_csv:   str = TRAIN_CSV,
    json_path:   str = JSON_PATH,
    n_synthetic: int = 0,   # disabled — synthetic hurts with real data available
    verbose:     bool = True,
) -> pd.DataFrame:

    df_features    = load_features(json_path)
    all_codes      = set(df_features.index)
    parts          = []
    codes_w_real   = set()

    # ── Load real data ────────────────────────────────────────────────────
    if Path(train_csv).exists() and Path(train_csv).stat().st_size > 10:
        df_real = pd.read_csv(train_csv, encoding="utf-8-sig")

        # For CS major users, use major_quota_threshold as the target
        # This is the threshold specifically among Y(Y) major quota students
        # Fall back to winning_threshold if major_quota_threshold not available
        if "major_quota_threshold" in df_real.columns:
            df_real[TARGET_COL] = df_real["major_quota_threshold"].fillna(
                df_real.get("winning_threshold", df_real.get(TARGET_COL, 1.0))
            )
        elif TARGET_COL not in df_real.columns and "winning_threshold" in df_real.columns:
            df_real = df_real.rename(columns={"winning_threshold": TARGET_COL})

        df_real = df_real.dropna(subset=[TARGET_COL])
        df_real[TARGET_COL] = pd.to_numeric(df_real[TARGET_COL], errors="coerce")
        df_real = df_real.dropna(subset=[TARGET_COL])

        # Professor filtering is handled by sample_weight in process_mileage_data.py
        # Tier 1 (same professor) = weight 5.0
        # Tier 2 (different professor) = weight 2.0
        # No need to hard-filter here — weights handle the influence
        if verbose:
            t1 = (df_real.get("data_tier", pd.Series()) == 1).sum()
            t2 = (df_real.get("data_tier", pd.Series()) == 2).sum()
            print(f"Professor filter: kept all {len(df_real)} rows "
                  f"(Tier 1: {t1} rows weight=5x, Tier 2: {t2} rows weight=2x)")

        # Add student context defaults
        for col, val in [
            ("student_year",       3.0),
            ("student_mileage",    72.0),   # fixed Yonsei rule
            ("num_courses_wanted", 5.0),
            ("rank_in_list",       1.0),
            ("priority_ratio",     0.5),
            ("budget_ratio",       1.0),    # always 72/72
        ]:
            if col not in df_real.columns:
                df_real[col] = val

        # Join missing feature columns
        missing_feats = [c for c in FEATURE_COLS if c not in df_real.columns]
        if missing_feats and "course_code" in df_real.columns:
            feat_reset = df_features.reset_index()
            df_real = df_real.merge(
                feat_reset[["course_code"] +
                           [c for c in missing_feats if c in df_features.columns]],
                on="course_code", how="left"
            )

        for col in FEATURE_COLS:
            if col not in df_real.columns:
                df_real[col] = 0.0

        df_real["data_source"] = "real"
        # Sample weight = recency weight × 5 (real data bonus)
        # Recent semesters count more than old ones
        if "recency_weight" in df_real.columns:
            df_real["sample_weight"] = df_real["recency_weight"] * 5.0
        else:
            df_real["sample_weight"] = 5.0

        codes_w_real = set(df_real["course_code"].unique()) \
            if "course_code" in df_real.columns else set()

        parts.append(df_real)

        if verbose:
            print(f"Real data (professor-matched, pre-2026-1): "
                  f"{len(df_real)} rows ({len(codes_w_real)} courses)")
    else:
        if verbose:
            print(f"No training data found at {train_csv}")
            print("Run: python3 process_mileage_data.py")

    # ── Generate synthetic for courses with no professor-matched history ──
    codes_need_synth = all_codes - codes_w_real

    if verbose:
        print(f"Courses needing synthetic data: {len(codes_need_synth)}")

    if codes_need_synth and n_synthetic > 0:
        df_feat_sub = df_features.loc[
            df_features.index.intersection(codes_need_synth)
        ]
        df_synth = generate_synthetic_bids(df_feat_sub, n_synthetic)
        df_synth["data_source"]   = "synthetic"
        df_synth["sample_weight"] = 1.0
        parts.append(df_synth)
        if verbose:
            print(f"Synthetic rows generated: {len(df_synth)}")
    elif codes_need_synth:
        if verbose:
            print(f"Skipping synthetic data ({len(codes_need_synth)} courses "
                  f"have no history — predictions will use course features only)")

    if not parts:
        raise RuntimeError("No training data available")

    df = pd.concat(parts, ignore_index=True)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    if verbose:
        real_n  = (df["data_source"] == "real").sum()
        synth_n = (df["data_source"] == "synthetic").sum()
        print(f"\n✅ Final training set: {OUTPUT_CSV}")
        print(f"   Total: {len(df)} | Real: {real_n} | Synthetic: {synth_n}")
        print(f"\n{TARGET_COL} stats:")
        print(df[TARGET_COL].describe().round(2))

    return df


if __name__ == "__main__":
    build()



