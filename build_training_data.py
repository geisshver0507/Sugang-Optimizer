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

TRAIN_CSV  = "mileage_training_real.csv"   # pre-2026-1, professor-matched
JSON_PATH  = "segmented_cs_courses.json"
OUTPUT_CSV = "training_final.csv"


def get_2026_professors(json_path: str) -> dict:
    """
    Get the professor for each course in the 2026-1 semester
    from the segmented_cs_courses.json database.
    Returns {course_code: professor_norm}
    """
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
    prof_2026      = get_2026_professors(json_path)
    parts          = []
    codes_w_real   = set()

    # ── Load real data ────────────────────────────────────────────────────
    if Path(train_csv).exists() and Path(train_csv).stat().st_size > 10:
        df_real = pd.read_csv(train_csv, encoding="utf-8-sig")

        if TARGET_COL not in df_real.columns and "winning_threshold" in df_real.columns:
            df_real = df_real.rename(columns={"winning_threshold": TARGET_COL})

        df_real = df_real.dropna(subset=[TARGET_COL])
        df_real[TARGET_COL] = pd.to_numeric(df_real[TARGET_COL], errors="coerce")
        df_real = df_real.dropna(subset=[TARGET_COL])

        # ── Professor matching filter ─────────────────────────────────────
        # Only filter when BOTH the historical row AND 2026-1 have professor info.
        # If either is missing, keep the row — sparse data means we can't afford
        # to throw away valid history just because professor wasn't scraped.
        if "professor_norm" in df_real.columns:
            original_len = len(df_real)

            def professor_matches(row):
                code          = row.get("course_code", "")
                prof_hist     = str(row.get("professor_norm", "")).strip()
                prof_2026_val = prof_2026.get(code, "")

                # Either side missing info → keep (can't make a reliable judgment)
                if not prof_2026_val or not prof_hist or prof_hist == "nan":
                    return True
                # Both present → only keep if they match
                return prof_hist == prof_2026_val

            df_real["prof_match"] = df_real.apply(professor_matches, axis=1)
            df_filtered = df_real[df_real["prof_match"]].drop("prof_match", axis=1)

            filtered_out = original_len - len(df_filtered)
            if verbose:
                kept = len(df_filtered)
                print(f"Professor filter: kept {kept} rows, "
                      f"removed {filtered_out} confirmed professor mismatches")
            df_real = df_filtered

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

        df_real["data_source"]   = "real"
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