"""
strategy_engine.py
------------------
The integration seam between Part 1 (chatbot) and Part 2 (strategy).

Part 1 calls get_strategy_for_ranked_list() with the chatbot's output.
This function handles everything else and returns BidResult objects.
"""

from __future__ import annotations

from feature_extractor import load_features, flatten_json
from model import load_model, predict_threshold, explain_threshold
from optimizer import CourseInput, allocate_bids, BidResult
from synthetic_data_generator import FEATURE_COLS

JSON_PATH = "segmented_cs_courses.json"


def get_strategy_for_ranked_list(
    ranked_list:    list,   # [{"code": "CAS3205-01-00", "name": "...", "rank": 1}, ...]
    student_profile: dict,  # {"year": 3, "mileage": 150}
    json_path:      str   = JSON_PATH,
    safety_margin:  float = 0.15,
) -> list:                  # list[BidResult]
    """
    Main function called by Part 1 chatbot.

    Parameters
    ----------
    ranked_list     List of dicts. Each needs "code", "name", "rank".
                    rank=1 means most important to the student.

    student_profile Dict with "year" (int 1-4) and "mileage" (int).

    Returns
    -------
    list[BidResult] — one per course, sorted by rank.
    Each has: .recommended_bid, .predicted_threshold,
              .risk_level, .confidence_pct, .note, .shap_breakdown
    """
    model, explainer = load_model()
    df_features      = load_features(json_path)
    flat_courses     = flatten_json(json_path)

    student_year   = float(student_profile.get("year",    2))
    total_mileage  = int(student_profile.get("mileage", 100))
    n_courses      = len(ranked_list)

    course_inputs = []
    skipped       = []

    for item in ranked_list:
        code = item.get("code", "")
        name = item.get("name", code)
        rank = int(item.get("rank", 99))

        if code not in df_features.index:
            skipped.append(code)
            continue

        # Build the feature row for this course + student context
        feat = df_features.loc[code].to_dict()
        feat["student_year"]       = student_year
        feat["student_mileage"]    = float(total_mileage)
        feat["num_courses_wanted"] = float(n_courses)
        feat["rank_in_list"]       = float(rank)
        feat["priority_ratio"]     = (n_courses - rank + 1) / max(n_courses, 1)
        feat["budget_ratio"]       = total_mileage / 500.0

        threshold, shap_breakdown = predict_threshold(model, explainer, feat)

        c_info = flat_courses.get(code, {})
        course_inputs.append(CourseInput(
            code                = code,
            name                = name,
            rank                = rank,
            predicted_threshold = threshold,
            is_major_req        = (c_info.get("category") == "major_requirement"),
            shap_breakdown      = shap_breakdown,
        ))

    if skipped:
        print(f"[strategy_engine] Skipped unknown courses: {skipped}")

    return allocate_bids(
        course_inputs,
        total_mileage  = total_mileage,
        safety_margin  = safety_margin,
    )


def format_strategy_for_chat(results: list) -> str:
    """
    Converts BidResult list to markdown string the chatbot
    can append directly to its response.
    """
    if not results:
        return "No strategy could be generated."

    total = sum(r.recommended_bid for r in results)
    EMOJI = {"Safe": "🟢", "Moderate": "🟡", "Risky": "🔴"}

    lines = [
        "## 🎯 Recommended Mileage Strategy\n",
        f"**Total points allocated: {total}**\n",
    ]

    for r in results:
        e = EMOJI.get(r.risk_level, "⚪")
        lines.append(
            f"**{r.rank}. {r.name}**  →  Bid **{r.recommended_bid} pts**  "
            f"{e} {r.risk_level} ({r.confidence_pct:.0f}% confidence)  "
            f"*(threshold est. ~{int(r.predicted_threshold)} pts)*"
        )
        if r.note:
            lines.append(f"   > {r.note}")

        # Top 2 SHAP drivers
        from model import FEATURE_LABELS
        top = sorted(r.shap_breakdown.items(),
                     key=lambda x: abs(x[1]), reverse=True)[:2]
        if top:
            drivers = "  |  ".join(
                f"{'↑' if v>0 else '↓'} {FEATURE_LABELS.get(k, k)}"
                for k, v in top
            )
            lines.append(f"   *Driven by: {drivers}*")
        lines.append("")

    return "\n".join(lines)
