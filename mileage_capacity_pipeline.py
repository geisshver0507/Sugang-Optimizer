"""Capacity/demand training utilities for mileage average prediction.

The functions in this module are deterministic and intentionally plain pandas.
They avoid using post-registration cutoffs as future-facing features while still
using historical application counts to learn the capacity-demand relationship.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


MAX_BID_PER_COURSE = 36.0
DEFAULT_NEUTRAL_REVIEW_SCORE = 3.0


def get_course_review_score(course_id: str) -> float:
    """Placeholder for the real sentiment/review service."""
    return DEFAULT_NEUTRAL_REVIEW_SCORE


def _numeric(series: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series(dtype=float)


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_capacity_and_demand_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized max_capacity, num_applied, and demand-ratio columns.

    Historical rows use actual applicant counts when available. Future rows use
    ETA/cart additions as demand_proxy because num_applied is not knowable yet.
    """
    out = df.copy()

    capacity_col = _first_existing_column(
        out,
        (
            "max_capacity",
            "maximum_capacity",
            "capacity",
            "class_capacity",
            "inferred_class_size",
            "competitive_seats",
        ),
    )
    if capacity_col:
        out["max_capacity"] = _numeric(out[capacity_col])
    else:
        out["max_capacity"] = 0.0

    applied_col = _first_existing_column(
        out,
        ("num_applied", "total_applicants", "competitive_applicants", "applicants"),
    )
    if applied_col:
        out["num_applied"] = _numeric(out[applied_col])
    else:
        out["num_applied"] = 0.0

    eta_col = _first_existing_column(out, ("eta_added", "mileage_historical_eta", "eta_count"))
    if eta_col:
        out["eta_added"] = _numeric(out[eta_col])
    else:
        out["eta_added"] = 0.0

    out["demand_proxy"] = out["num_applied"].where(out["num_applied"] > 0, out["eta_added"])
    safe_capacity = out["max_capacity"].where(out["max_capacity"] > 0, 1.0)
    out["demand_capacity_ratio"] = (out["demand_proxy"] / safe_capacity).clip(lower=0.0, upper=8.0)
    out["capacity_demand_gap"] = (out["demand_proxy"] - out["max_capacity"]).clip(lower=-300.0, upper=500.0)
    return out


def capacity_application_correlation(df: pd.DataFrame) -> float:
    """Return Pearson correlation between max_capacity and num_applied."""
    work = normalize_capacity_and_demand_columns(df)
    valid = work[(work["max_capacity"] > 0) & (work["num_applied"] > 0)]
    if len(valid) < 2:
        return 0.0
    corr = valid["max_capacity"].corr(valid["num_applied"])
    return 0.0 if pd.isna(corr) else float(corr)


def create_average_mileage_target(df: pd.DataFrame) -> pd.DataFrame:
    """Set average_mileage as the training target, replacing cutoff targets.

    Preference order:
    1. avg_all_bids: all competitive student bids, best match to requested target.
    2. avg_winning_bid: enrolled competitive bids.
    3. winning_threshold / major_quota_threshold: deterministic fallback only.
    """
    out = normalize_capacity_and_demand_columns(df)
    candidates = [
        "average_mileage",
        "avg_all_bids",
        "avg_winning_bid",
        "major_quota_threshold",
        "winning_threshold",
    ]

    target = pd.Series(index=out.index, dtype=float)
    for column in candidates:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        target = target.fillna(values)

    out["average_mileage"] = target.clip(lower=1.0, upper=MAX_BID_PER_COURSE)
    return out


@dataclass(frozen=True)
class ColdStartDecision:
    multiplier: float
    condition: str
    reason: str


def cold_start_weight(
    historical_records: int,
    course_id: str = "",
    professor: str = "",
    review_score: float | None = None,
    has_known_professor: bool | None = None,
) -> ColdStartDecision:
    """Return deterministic multiplier for sparse-history predictions.

    Conditions implemented from the project brief:
    - A: 0 records and unknown professor/reviews -> severe demand penalty.
    - B: 0 records but highly rated professor -> moderate multiplier.
    - C: exactly 1 record -> adjust up/down by sentiment.
    """
    records = int(historical_records or 0)
    if records >= 2:
        return ColdStartDecision(1.0, "established", "Two or more historical records")

    if review_score is None:
        review_score = get_course_review_score(course_id)
    try:
        score = float(review_score)
    except (TypeError, ValueError):
        score = DEFAULT_NEUTRAL_REVIEW_SCORE

    if has_known_professor is None:
        prof_text = str(professor or "").strip().lower()
        has_known_professor = bool(prof_text and prof_text not in {"unknown", "not listed", "nan"})

    high_review = score >= 4.2
    low_review = score <= 2.6

    if records == 0:
        if not has_known_professor and not high_review:
            return ColdStartDecision(
                0.62,
                "A",
                "New subject with unknown professor; students are unlikely to risk high mileage",
            )
        if high_review:
            return ColdStartDecision(
                0.88,
                "B",
                "New subject with positive professor sentiment; rely on capacity and ETA demand",
            )
        return ColdStartDecision(
            0.74,
            "A",
            "New subject with limited trust signal; apply conservative demand penalty",
        )

    if high_review:
        return ColdStartDecision(1.12, "C", "One record plus high review score; popularity may rise")
    if low_review:
        return ColdStartDecision(0.86, "C", "One record plus low review score; demand may soften")
    return ColdStartDecision(1.0, "C", "One record with neutral sentiment")


def apply_cold_start_weight(
    predicted_average: float,
    historical_records: int,
    course_id: str = "",
    professor: str = "",
    review_score: float | None = None,
    has_known_professor: bool | None = None,
) -> tuple[float, ColdStartDecision]:
    decision = cold_start_weight(
        historical_records=historical_records,
        course_id=course_id,
        professor=professor,
        review_score=review_score,
        has_known_professor=has_known_professor,
    )
    weighted = float(predicted_average) * decision.multiplier
    return max(1.0, min(MAX_BID_PER_COURSE, weighted)), decision


def build_average_mileage_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Return enriched training frame and capacity/applicant correlation."""
