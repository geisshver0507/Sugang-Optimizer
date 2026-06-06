"""
strategy_engine.py
------------------
Integration seam between Part 1 (chatbot) and Part 2 (strategy).

YONSEI MILEAGE RULES (fixed — not user inputs):
  - Every student gets exactly 72 points per semester
  - Maximum bid on any single course: 36 points

Part 1 chatbot calls get_strategy_for_ranked_list() with the
student's ranked course list and year. Mileage is not needed.
"""

from __future__ import annotations

from feature_extractor import load_features, flatten_json
from model import load_model, predict_threshold, FEATURE_LABELS
from optimizer import (
    CourseInput, allocate_bids, BidResult,
    TOTAL_MILEAGE, MAX_BID_PER_COURSE
)

JSON_PATH = "segmented_cs_courses.json"


def get_strategy_for_ranked_list(
    ranked_list:     list,   # [{"code": "CAS3205-01", "name": "...", "rank": 1}, ...]
    student_profile: dict,   # {"year": 3}  — mileage no longer needed
    json_path:       str   = JSON_PATH,
    safety_margin:   float = 0.15,
) -> list:                   # list[BidResult]
    """
    Main function called by Part 1 chatbot.

    Parameters
    ----------
    ranked_list     List of dicts, each needs "code", "name", "rank".
                    rank=1 = most important to the student.

    student_profile Dict with "year" (int 1-4).
                    Mileage is NOT needed — it's always 72pts at Yonsei.

    Returns
    -------
    list[BidResult] sorted by rank. Each has:
        .recommended_bid        (int, out of 72 total)
        .predicted_threshold    (float, model estimate)
        .risk_level             (Safe / Moderate / Risky)
        .confidence_pct         (float)
        .note                   (str, human-readable tip)
        .shap_breakdown         (dict, for SHAP explanation)
    """
    model, explainer = load_model()
    df_features      = load_features(json_path)
    flat_courses     = flatten_json(json_path)

    student_year = float(student_profile.get("year", 2))
    n_courses    = len(ranked_list)

    course_inputs = []
    skipped       = []

    for item in ranked_list:
        code = item.get("code", "")
        name = item.get("name", code)
        rank = int(item.get("rank", 99))

        if code not in df_features.index:
            skipped.append(code)
            continue

        # Build feature row — mileage is always 72, budget_ratio always 1.0
        feat = df_features.loc[code].to_dict()
        feat.update({
            "student_year":       student_year,
            "num_courses_wanted": float(n_courses),
            "rank_in_list":       float(rank),
            "priority_ratio":     (n_courses - rank + 1) / max(n_courses, 1),
            # is_cs_major always True since system is for CS majors only
            "is_cs_major":        1.0,
        })

        threshold, shap = predict_threshold(model, explainer, feat)

        c_info = flat_courses.get(code, {})
        course_inputs.append(CourseInput(
            code                = code,
            name                = name,
            rank                = rank,
            predicted_threshold = threshold,
            is_major_req        = (c_info.get("category") == "major_requirement"),
            shap_breakdown      = shap,
        ))

    if skipped:
        print(f"[strategy_engine] Skipped unknown courses: {skipped}")

    return allocate_bids(
        course_inputs,
        total_mileage  = TOTAL_MILEAGE,
        max_per_course = MAX_BID_PER_COURSE,
        safety_margin  = safety_margin,
    )


def format_strategy_for_chat(results: list) -> str:
    """
    Converts BidResult list to markdown the chatbot appends to its response.
    """
    if not results:
        return "No strategy could be generated."

    total = sum(r.recommended_bid for r in results)
    EMOJI = {"Safe": "🟢", "Moderate": "🟡", "Risky": "🔴"}

    lines = [
        "## 🎯 Recommended Mileage Strategy\n",
        f"**Total: {total}/{TOTAL_MILEAGE} pts used**  "
        f"*(max {MAX_BID_PER_COURSE}pts per course)*\n",
    ]

    for r in results:
        e = EMOJI.get(r.risk_level, "⚪")
        lines.append(
            f"**{r.rank}. {r.name}**  →  Bid **{r.recommended_bid} pts**  "
            f"{e} {r.risk_level} ({r.confidence_pct:.0f}% confidence)  "
            f"*(est. threshold ~{int(r.predicted_threshold)} pts)*"
        )
        if r.note:
            lines.append(f"   > {r.note}")

        # Top 2 SHAP drivers
        top = sorted(r.shap_breakdown.items(),
                     key=lambda x: abs(x[1]), reverse=True)[:2]
        if top:
            drivers = "  |  ".join(
                f"{'↑' if v > 0 else '↓'} {FEATURE_LABELS.get(k, k)}"
                for k, v in top
            )
            lines.append(f"   *Key factors: {drivers}*")
        lines.append("")

    return "\n".join(lines)