"""
strategy_engine.py  (v2 — survival-curve based)
===============================================
Integration seam between Part 1 (chatbot) and Part 2 (strategy).

YONSEI MILEAGE RULES (fixed, not user inputs):
  - 72 points per semester
  - Max 36 points on any single course

WHAT CHANGED FROM v1
--------------------
v1 trained one XGBoost regressor on a single collapsed "winning_threshold"
per course and ignored the professor and the full bid distribution. It
systematically overbid (rank-1 courses got 36pt even when winnable for 2pt).

v2 builds a per-course *survival curve* P(enroll | bid) from the per-applicant
data, finds the cheapest safe bid, and allocates the 72pt budget to win the
MOST courses for the LEAST points — funding genuinely-hard low-priority
courses instead of starving them, and being honest when the list can't all
be won.

Part 1 calls get_strategy_for_ranked_list() exactly as before, but each item
may now optionally carry the 2026 professor. If it doesn't, we resolve it from
the scraped data automatically.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from allocator import build_strategy, format_strategy, CourseBid, TOTAL_MILEAGE

RAW_CSV   = "mileage_history_all.csv"
JSON_PATH = "segmented_cs_courses.json"


def _resolve_current_professors(raw_csv: str = RAW_CSV) -> dict[str, str]:
    """
    Map each course_code → its most recent (2026-1) professor, so Part 1 doesn't
    have to supply it. Falls back to most recent available year.
    """
    try:
        df = pd.read_csv(raw_csv, encoding="utf-8-sig")
    except Exception:
        return {}
    df = df.dropna(subset=["professor"])
    out = {}
    for code, grp in df.groupby("course_code"):
        recent = grp.sort_values(["year", "semester"], ascending=False)
        profs = recent["professor"].dropna()
        if len(profs) > 0:
            out[code] = profs.iloc[0]
    return out


def get_strategy_for_ranked_list(
    ranked_list: list,            # [{"code","name","rank", optional "professor"}]
    student_profile: dict,        # {"year": 3}
    raw_csv: str = RAW_CSV,
    json_path: str = JSON_PATH,
    target_confidence: float = 0.90,
) -> list[CourseBid]:
    """
    Main function called by the Part 1 chatbot.

    Parameters
    ----------
    ranked_list      List of dicts with "code", "name", "rank" (1 = top choice).
                     "professor" is optional — resolved from data if absent.
    student_profile  {"year": int 1-4}. Mileage is always 72 at Yonsei.
    target_confidence  Win-probability the safe bid aims for (default 90%).

    Returns
    -------
    list[CourseBid] sorted by rank, each carrying bid / win_prob / status /
    honest note / data-quality confidence.
    """
    student_year = int(student_profile.get("year", 3))
    prof_map = _resolve_current_professors(raw_csv)

    enriched = []
    for item in ranked_list:
        code = item.get("code", "")
        enriched.append({
            "code": code,
            "name": item.get("name", code),
            "rank": int(item.get("rank", 99)),
            "professor": item.get("professor") or prof_map.get(code, ""),
        })

    return build_strategy(
        enriched,
        student_year=student_year,
        target_conf=target_confidence,
        raw_csv=raw_csv,
        json_path=json_path,
    )


def format_strategy_for_chat(results: list[CourseBid]) -> str:
    """Markdown block the chatbot appends to its reply."""
    return format_strategy(results)
