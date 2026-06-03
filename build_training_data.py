"""
build_training_data.py
----------------------
Combines real scraped data with synthetic data into one training set.

REAL DATA (from process_mileage_data.py → mileage_training_real.csv):
  - winning_threshold = min bid of the last enrolled student
  - One row per course per semester
  - These rows get 5× weight during training

SYNTHETIC DATA (from synthetic_data_generator.py):
  - Generated only for courses with ZERO real history
  - Plausible but not ground truth
  - These rows get 1× weight

The model trains on both, but real data dominates where it exists.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from feature_extractor import load_features, flatten_json
from synthetic_data_generator import (
    generate_synthetic_bids, FEATURE_COLS, TARGET_COL
)

REAL_CSV   = "mileage_training_real.csv"
JSON_PATH  = "segmented_cs_courses.json"
OUTPUT_CSV = "training_final.csv"


def build(real_csv: str = REAL_CSV, json_path: str = JSON_PATH,
          n_synthetic: int = 40, verbose: bool = True) -> pd.DataFrame:

    df_features = load_features(json_path)
    all_codes   = set(df_features.index)

    # ── Load real data ────────────────────────────────────────────────────
    real_exists  = Path(real_csv).exists()
    codes_w_real = set()
    parts        = []

    if real_exists:
        df_real = pd.read_csv(real_csv, encoding="utf-8-sig")

        # Rename target column if it comes in as 'winning_threshold'
        if TARGET_COL not in df_real.columns and "winning_threshold" in df_real.columns:
            df_real = df_real.rename(columns={"winning_threshold": TARGET_COL})

        # Drop rows with no target
        df_real = df_real.dropna(subset=[TARGET_COL])
        df_real[TARGET_COL] = pd.to_numeric(df_real[TARGET_COL], errors="coerce")
        df_real = df_real.dropna(subset=[TARGET_COL])

        # Add student context columns with realistic defaults if missing
        # (real data has course-level winning_threshold, not per-student context)
        defaults = {
            "student_year":       3.0,
            "student_mileage":    150.0,
            "num_courses_wanted": 5.0,
            "rank_in_list":       1.0,
            "priority_ratio":     0.5,
            "budget_ratio":       0.3,
        }
        for col, val in defaults.items():
            if col not in df_real.columns:
                df_real[col] = val

        # Join feature columns that are missing from real data
        missing_feats = [c for c in FEATURE_COLS if c not in df_real.columns]
        if missing_feats and "course_code" in df_real.columns:
            feat_reset = df_features.reset_index()
            df_real = df_real.merge(
                feat_reset[["course_code"] +
                           [c for c in missing_feats if c in df_features.columns]],
                on="course_code", how="left"
            )

        # Fill any still-missing feature columns with 0
        for col in FEATURE_COLS:
            if col not in df_real.columns:
                df_real[col] = 0.0

        df_real["data_source"]    = "real"
        df_real["sample_weight"]  = 5.0   # real rows count 5× more

        codes_w_real = set(df_real["course_code"].unique()) \
            if "course_code" in df_real.columns else set()

        parts.append(df_real)

        if verbose:
            print(f"Real data:       {len(df_real)} rows  "
                  f"({len(codes_w_real)} courses)")

    # ── Generate synthetic for courses WITHOUT real data ──────────────────
    codes_need_synthetic = all_codes - codes_w_real

    if verbose:
        print(f"Courses needing synthetic: {len(codes_need_synthetic)}")

    if codes_need_synthetic:
        df_feat_sub = df_features.loc[
            df_features.index.intersection(codes_need_synthetic)
        ]
        df_synth = generate_synthetic_bids(df_feat_sub, n_synthetic)
        df_synth["data_source"]   = "synthetic"
        df_synth["sample_weight"] = 1.0
        parts.append(df_synth)
        if verbose:
            print(f"Synthetic data:  {len(df_synth)} rows")

    if not parts:
        raise RuntimeError("No training data — run scrape_mileage.py first")

    df = pd.concat(parts, ignore_index=True)

    # Ensure all feature columns present
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"\n✅ Training set saved → {OUTPUT_CSV}")
        print(f"   Total rows:     {len(df)}")
        real_n  = (df["data_source"] == "real").sum()
        synth_n = (df["data_source"] == "synthetic").sum()
        print(f"   Real rows:      {real_n}")
        print(f"   Synthetic rows: {synth_n}")
        print(f"\n{TARGET_COL} stats:")
        print(df[TARGET_COL].describe().round(2))

    return df


if __name__ == "__main__":
    build()
