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

from pathlib import Path

from feature_extractor import load_features, flatten_json
from model_robust import load_model, predict_threshold, FEATURE_LABELS
from optimizer import BidResult, TOTAL_MILEAGE, MAX_BID_PER_COURSE

ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = ROOT / "database"
JSON_PATH = str(DATABASE_DIR / "segmented_cs_courses.json")
BASE_SAFETY_BUFFER = 3.0
HIGH_DEMAND_RATIO = 1.18


def _calibrated_bid_adjustment(predicted_avg: float, shap: dict, rank: int, n_courses: int) -> tuple[float, list[str]]:
    """Post-model bid calibration using only deterministic course signals."""
    priority_ratio = (n_courses - rank + 1) / max(n_courses, 1)
    demand_ratio = float(shap.get("demand_capacity_ratio", 0.0) or 0.0)
    cold_condition = str(shap.get("cold_start_condition", "") or "")

    adjustment = 0.0
    reasons = []

    if demand_ratio >= HIGH_DEMAND_RATIO and priority_ratio >= 0.5 and cold_condition != "A":
        if predicted_avg < 12.0:
            adjustment += 6.0
            reasons.append("high ETA/capacity pressure with low predicted average")
        elif predicted_avg < 13.5:
            adjustment += 4.0
            reasons.append("high ETA/capacity pressure with likely underprediction")

    if cold_condition == "A":
        adjustment -= 2.5
        reasons.append("cold-start penalty for low-trust sparse history")

    if cold_condition in {"A", "B"} and priority_ratio < 0.5:
        adjustment -= 7.0
        reasons.append("low-priority sparse-history discount")

    return adjustment, reasons


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

    results = []
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

        predicted_avg, shap = predict_threshold(model, explainer, feat)
        bid_adjustment, adjustment_reasons = _calibrated_bid_adjustment(
            predicted_avg, shap, rank, n_courses
        )
        raw_bid = predicted_avg + BASE_SAFETY_BUFFER + bid_adjustment
        recommended_bid = int(round(min(MAX_BID_PER_COURSE, max(1.0, raw_bid))))
        shap["bid_adjustment"] = bid_adjustment
        ratio = recommended_bid / max(predicted_avg, 1.0)

        if ratio >= 1.20:
            risk_level = "Safe"
            confidence_pct = min(95.0, 55.0 + ratio * 20.0)
        elif ratio >= 0.90:
            risk_level = "Moderate"
            confidence_pct = 40.0 + ratio * 20.0
        else:
            risk_level = "Risky"
            confidence_pct = max(10.0, ratio * 50.0)

        c_info = flat_courses.get(code, {})
        if c_info.get("category") == "major_requirement" and risk_level != "Safe":
            note = "Major requirement; review whether this bid deserves extra priority."
        elif recommended_bid == MAX_BID_PER_COURSE:
            note = f"At university maximum ({MAX_BID_PER_COURSE} pts)."
        elif adjustment_reasons:
            note = "Calibrated from predicted average + 3.0 buffer: " + "; ".join(adjustment_reasons) + "."
        else:
            note = "Recommended bid = predicted average mileage + 3.0 safety buffer."

        results.append(BidResult(
            code                = code,
            name                = name,
            rank                = rank,
            recommended_bid     = recommended_bid,
            predicted_threshold = round(predicted_avg, 1),
            risk_level          = risk_level,
            confidence_pct      = round(confidence_pct, 1),
            shap_breakdown      = shap,
            note                = note,
        ))

    if skipped:
        print(f"[strategy_engine] Skipped unknown courses: {skipped}")

    return sorted(results, key=lambda item: item.rank)


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
        f"**Total suggested: {total}/{TOTAL_MILEAGE} pts**  "
        f"*(max {MAX_BID_PER_COURSE}pts per course)*\n",
    ]
    if total > TOTAL_MILEAGE:
        lines.append(
            f"Warning: these per-course secure bids exceed the {TOTAL_MILEAGE} point semester budget. "
            "Drop or reprioritize lower-ranked courses before final submission.\n"
        )

    for r in results:
        e = EMOJI.get(r.risk_level, "⚪")
        lines.append(
            f"**{r.rank}. {r.name}**  →  Bid **{r.recommended_bid} pts**  "
            f"{e} {r.risk_level} ({r.confidence_pct:.0f}% confidence)  "
            f"*(predicted avg ~{r.predicted_threshold:.1f} pts; +3.0 buffer)*"
        )
        if r.note:
            lines.append(f"   > {r.note}")

        # Top 2 SHAP drivers
        numeric_breakdown = {
            key: value
            for key, value in r.shap_breakdown.items()
            if isinstance(value, (int, float))
        }
        top = sorted(numeric_breakdown.items(),
                     key=lambda x: abs(x[1]), reverse=True)[:2]
        if top:
            drivers = "  |  ".join(
                f"{'↑' if v > 0 else '↓'} {FEATURE_LABELS.get(k, k)}"
                for k, v in top
            )
            lines.append(f"   *Key factors: {drivers}*")
        lines.append("")

    return "\n".join(lines)




