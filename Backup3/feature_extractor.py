"""
feature_extractor.py
--------------------
Reads segmented_cs_courses.json and converts every course into a
flat numeric feature vector that XGBoost can train on.

Nothing is hardcoded about WHICH direction a feature matters —
that is left entirely to the model to discover.
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_review_score(review_text: str) -> float:
    """Extract numeric score like 4.50/5 or 2.12/5.00 from review text.
    Returns NaN if no score found — model handles missing values fine."""
    if not review_text:
        return np.nan
    match = re.search(r"(\d+\.\d+)\s*/\s*5", review_text)
    if match:
        return float(match.group(1))
    # Positive/negative sentiment proxy when no numeric score exists
    pos_words = ["excellent", "perfect", "great", "good", "generous", "passionate",
                 "high quality", "well-liked", "positive", "praised"]
    neg_words = ["critical", "negative", "frustrated", "poor", "worst",
                 "chaotic", "confused", "difficult", "strict", "old-fashioned"]
    text_lower = review_text.lower()
    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)
    if pos_count + neg_count == 0:
        return np.nan
    # Map to 1-5 scale
    ratio = pos_count / (pos_count + neg_count)
    return 1.0 + ratio * 4.0


def _parse_earliest_hour(time_str: str) -> int:
    """
    Extract the earliest lecture period number from strings like:
    'Tue 2, Thu 2,3'  → 2
    'Wed 5 / Fri 5,6' → 5
    'Mon 2,3,4'       → 2
    Yonsei period mapping (approx): 1-3=morning, 4-6=midday, 7+=afternoon/evening
    Returns 99 if unparseable.
    """
    if not time_str:
        return 99
    numbers = re.findall(r"\b(\d+)\b", time_str)
    # Filter out anything that looks like a year/room number (>20)
    periods = [int(n) for n in numbers if 1 <= int(n) <= 15]
    return min(periods) if periods else 99


def _parse_grading_exam_weight(grading_str: str) -> float:
    """Sum of midterm + final percentages — proxy for exam-heaviness."""
    if not grading_str:
        return np.nan
    weights = re.findall(r"(\d+)%\s*(midterm|final)", grading_str, re.IGNORECASE)
    total = sum(int(w) for w, _ in weights)
    return float(total)


def _parse_assignment_weight(grading_str: str) -> float:
    """Sum of all assignment percentages."""
    if not grading_str:
        return np.nan
    weights = re.findall(r"(\d+)%\s*(?:individual\s+)?assignment", grading_str, re.IGNORECASE)
    return float(sum(int(w) for w in weights))


ORDINAL = {"low": 1, "light": 1, "easy": 1,
           "medium": 2, "moderate": 2,
           "high": 3, "heavy": 3, "hard": 3}

LECTURE_TYPE_MAP = {
    "in-person": 0,
    "offline": 0,
    "online": 2,
    "video recorded": 2,
    "pre-recorded": 2,
    "on-off blended": 1,
    "blended": 1,
}


def _lecture_type_score(lt: str) -> int:
    if not lt:
        return 0
    lt_lower = lt.lower()
    for key, val in LECTURE_TYPE_MAP.items():
        if key in lt_lower:
            return val
    return 0


# ── Main extractor ─────────────────────────────────────────────────────────

def flatten_json(json_path: str) -> dict[str, dict]:
    """
    Flatten the nested segmented JSON into a simple
    { course_code: metadata_dict } removing duplicates.
    The JSON stores the same course under multiple category keys
    (major_requirement, major_basic, major_elective) — we deduplicate.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    courses = {}
    for year_key, year_data in data.items():
        major_year = int(re.search(r"\d+", year_key).group()) if re.search(r"\d+", year_key) else 0
        for category_key, category_data in year_data.items():
            for code, course in category_data.items():
                if code not in courses:
                    entry = course["metadata"].copy()
                    entry["text_chunks"] = course["text_chunks"]
                    entry["major_year_label"] = major_year
                    entry["category"] = category_key  # first category seen
                    courses[code] = entry
    return courses


def extract_features(courses: dict[str, dict]) -> pd.DataFrame:
    """
    Convert the flat course dict into a DataFrame of numeric features.
    One row per course. Feature names are self-explanatory.
    XGBoost will figure out which ones matter and how.
    """
    rows = []
    for code, c in courses.items():
        txt = c.get("text_chunks", {})
        review_text = txt.get("student_reviews", "") or ""
        alt_review   = txt.get("alternative_professor_reviews", "") or ""
        grading_str  = txt.get("grading_and_syllabus", "") or ""

        # Combine reviews for scoring
        combined_review = review_text + " " + alt_review

        row = {
            "course_code": code,

            # ── Demand signal ────────────────────────────────────────────
            "eta_added": float(c.get("mileage_historical_eta", 0) or 0),
            "max_capacity": float(c.get("max_capacity", 0) or 0),
            "demand_proxy": float(c.get("mileage_historical_eta", 0) or 0),
            "demand_capacity_ratio": float(c.get("mileage_historical_eta", 0) or 0)
            / max(float(c.get("max_capacity", 0) or 0), 1.0),
            "capacity_demand_gap": float(c.get("mileage_historical_eta", 0) or 0) - float(c.get("max_capacity", 0) or 0),

            # ── Course structure ─────────────────────────────────────────
            "credits": float(c.get("credits", 3) or 3),
            "major_year_target": float(c.get("major_year_label", 0) or
                                       c.get("major_year", 0) or 0),
            "is_major_req": 1.0 if c.get("category") == "major_requirement" else 0.0,
            "is_major_basic": 1.0 if c.get("category") == "major_basic" else 0.0,
            "is_major_elective": 1.0 if c.get("category") == "major_elective" else 0.0,
            "num_prerequisites": float(len(c.get("prerequisites", []) or [])),

            # ── Difficulty / workload ────────────────────────────────────
            "difficulty_score": float(ORDINAL.get(
                str(c.get("difficulty", "medium")).lower(), 2)),
            "workload_score": float(ORDINAL.get(
                str(c.get("workload", "medium")).lower(), 2)),

            # ── Grading ──────────────────────────────────────────────────
            "is_relative_grading": 1.0 if str(
                c.get("evaluation_type", "")).lower() == "relative" else 0.0,
            "exam_weight": _parse_grading_exam_weight(grading_str),
            "assignment_weight": _parse_assignment_weight(grading_str),

            # ── Language / format ─────────────────────────────────────────
            "is_english": 1.0 if "english" in str(
                c.get("language_medium", "")).lower() else 0.0,
            "lecture_type_score": float(_lecture_type_score(
                str(c.get("lecture_type", "")))),

            # ── Time slot ────────────────────────────────────────────────
            # Lower period number = earlier in the day
            # Model will learn whether early/late matters — we just provide the number
            "earliest_period": float(_parse_earliest_hour(
                str(c.get("time", "")))),

            # ── Professor perception ──────────────────────────────────────
            "review_score": _parse_review_score(combined_review),
            "has_review": 1.0 if combined_review.strip() else 0.0,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("course_code")

    # Fill NaN review scores with median (neutral assumption)
    median_score = df["review_score"].median()
    df["review_score"] = df["review_score"].fillna(
        median_score if not np.isnan(median_score) else 3.0)
    df["exam_weight"] = df["exam_weight"].fillna(60.0)
    df["assignment_weight"] = df["assignment_weight"].fillna(20.0)

    return df


def load_features(json_path: str) -> pd.DataFrame:
    """One-call convenience: load JSON → feature DataFrame."""
    courses = flatten_json(json_path)
    return extract_features(courses)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "segmented_cs_courses.json"
    df = load_features(path)
    print(df.to_string())
    print(f"\nShape: {df.shape}")
