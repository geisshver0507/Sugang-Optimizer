"""
optimizer.py
------------
Portfolio optimizer: distributes a mileage budget across a ranked
list of courses, balancing competition difficulty and student priority.

YONSEI MILEAGE RULES (hardcoded — not user inputs):
  - Every student gets exactly 72 mileage points per semester
  - Maximum bid on any single course is 36 points
  - These are fixed university rules, not student preferences

CORE LOGIC:
  1. Predict minimum bid needed per course (from ML model)
  2. Cap predictions at MAX_BID_PER_COURSE (36pts)
  3. Weight allocation: 60% competition difficulty + 40% student priority
  4. Always use the full 72pt budget — unused points are wasted
  5. When budget tight: protect high-priority courses first
  6. When budget sufficient: distribute surplus as safety buffer
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# ── Yonsei fixed constants ─────────────────────────────────────────────────
TOTAL_MILEAGE      = 72   # every student gets exactly this per semester
MAX_BID_PER_COURSE = 36   # university hard cap per course


@dataclass
class CourseInput:
    code:                str
    name:                str
    rank:                int      # 1 = highest priority
    predicted_threshold: float    # model's min bid estimate
    is_major_req:        bool  = False
    shap_breakdown:      dict  = field(default_factory=dict)


@dataclass
class BidResult:
    code:                str
    name:                str
    rank:                int
    recommended_bid:     int
    predicted_threshold: float
    risk_level:          str
    confidence_pct:      float
    shap_breakdown:      dict  = field(default_factory=dict)
    note:                str   = ""


def allocate_bids(
    courses:       list,                        # list[CourseInput]
    total_mileage: int   = TOTAL_MILEAGE,       # default 72, can override in tests
    max_per_course: int  = MAX_BID_PER_COURSE,  # default 36
    safety_margin: float = 0.15,
    min_bid:       int   = 1,
) -> list:                                      # list[BidResult]
    """
    Allocate total_mileage across courses respecting Yonsei's hard caps.
    """

    if not courses:
        return []

    n          = len(courses)
    thresholds = np.array([c.predicted_threshold for c in courses], dtype=float)
    ranks      = np.array([c.rank               for c in courses], dtype=float)

    # ── Cap predictions at the university max ─────────────────────────────
    # No point predicting above 36 — it's impossible to bid more
    thresholds = np.minimum(thresholds, float(max_per_course))

    # ── Weights ───────────────────────────────────────────────────────────
    threshold_range = thresholds.max() - thresholds.min()

    if threshold_range > 2.0:
        # Meaningful spread — use competition weight
        comp_weight = thresholds / thresholds.sum()
    else:
        # All thresholds nearly equal — weight purely by priority
        # Prevents the "everyone gets the same" problem
        comp_weight = np.ones(n) / n

    # Priority weight: steeper dropoff so rank 1 gets meaningfully more
    prio_weight = 1.0 / (ranks ** 1.2)
    prio_weight = prio_weight / prio_weight.sum()

    # Combined: 60% competition-driven, 40% priority-driven
    combined = 0.6 * comp_weight + 0.4 * prio_weight

    # ── Minimum bids needed (capped at max_per_course) ────────────────────
    min_needed = thresholds * (1.0 + safety_margin)
    min_needed = np.minimum(min_needed, float(max_per_course))
    min_needed = np.maximum(min_needed, float(min_bid))

    # ── Distribute budget ─────────────────────────────────────────────────
    total_min = min_needed.sum()

    if total_min <= total_mileage:
        # Cover all minimums, then distribute surplus by combined weight
        # Surplus also capped per course at max_per_course
        surplus   = total_mileage - total_min
        allocated = min_needed + combined * surplus
        # Apply per-course cap
        allocated = np.minimum(allocated, float(max_per_course))
        # If capping created leftover, give to next-highest priority
        leftover = total_mileage - allocated.sum()
        if leftover > 1:
            for idx in np.argsort(ranks):   # priority order
                room = max_per_course - allocated[idx]
                if room > 0 and leftover > 0:
                    add = min(room, leftover)
                    allocated[idx] += add
                    leftover -= add

    else:
        # Budget too tight — protect high-priority courses first
        allocated = np.zeros(n)
        remaining = float(total_mileage)

        for idx in np.argsort(ranks):   # rank 1 first
            give = min(min_needed[idx], remaining, float(max_per_course))
            allocated[idx] = max(give, float(min_bid))
            remaining -= allocated[idx]
            if remaining <= 0:
                break

        # Distribute any remaining budget by combined weight
        if remaining > 0:
            for idx in np.argsort(ranks):
                room = max_per_course - allocated[idx]
                if room > 0 and remaining > 0:
                    extra = min(combined[idx] * remaining, room)
                    allocated[idx] += extra
                    remaining -= extra

    # ── Integer enforcement ───────────────────────────────────────────────
    # Scale down if somehow over budget
    if allocated.sum() > total_mileage:
        allocated = allocated * (total_mileage / allocated.sum())

    allocated = np.maximum(allocated, float(min_bid))
    allocated = np.minimum(allocated, float(max_per_course))
    int_bids  = np.round(allocated).astype(int)

    # Fix rounding drift — lowest priority first for removal, highest for addition
    drift = int_bids.sum() - total_mileage
    if drift > 0:
        for idx in np.argsort(-ranks):
            if int_bids[idx] > min_bid and drift > 0:
                int_bids[idx] -= 1
                drift -= 1
    elif drift < 0:
        for idx in np.argsort(ranks):
            if int_bids[idx] < max_per_course and drift < 0:
                int_bids[idx] += 1
                drift += 1
            if drift == 0:
                break

    # Final safety check — enforce caps
    int_bids = np.minimum(int_bids, max_per_course)
    int_bids = np.maximum(int_bids, min_bid)

    # ── Risk assessment ───────────────────────────────────────────────────
    results = []
    for i, course in enumerate(courses):
        bid   = int(int_bids[i])
        pred  = course.predicted_threshold
        ratio = bid / max(pred, 1.0)

        if ratio >= 1.20:
            risk, confidence = "Safe",     min(95.0, 55.0 + ratio * 20.0)
        elif ratio >= 0.90:
            risk, confidence = "Moderate", 40.0 + ratio * 20.0
        else:
            risk, confidence = "Risky",    max(10.0, ratio * 50.0)

        note = ""
        if bid == max_per_course:
            note = f"⚠ At university maximum ({max_per_course}pts) — highest possible bid."
        elif course.is_major_req and ratio < 0.85:
            note = "⚠ Major requirement — consider bidding higher."
        elif ratio > 2.5:
            note = "✓ Well above threshold — could redistribute some points."
        elif confidence < 35:
            note = "⚠ Budget too thin for this many courses — consider dropping lower-priority ones."

        results.append(BidResult(
            code                = course.code,
            name                = course.name,
            rank                = course.rank,
            recommended_bid     = bid,
            predicted_threshold = round(pred, 1),
            risk_level          = risk,
            confidence_pct      = round(confidence, 1),
            shap_breakdown      = course.shap_breakdown,
            note                = note,
        ))

    results.sort(key=lambda r: r.rank)
    return results


def strategy_summary(results: list, total_mileage: int = TOTAL_MILEAGE) -> str:
    used = sum(r.recommended_bid for r in results)
    lines = [
        f"Strategy: {len(results)} courses | {used}/{total_mileage} pts "
        f"(max {MAX_BID_PER_COURSE}pts/course)\n",
        f"{'#':<3} {'Course':<40} {'Bid':>6}  {'Threshold':>10}  "
        f"{'Risk':<10} {'Conf':>6}",
        "─" * 76,
    ]
    for r in results:
        name = r.name[:38] + ".." if len(r.name) > 40 else r.name
        lines.append(
            f"{r.rank:<3} {name:<40} {r.recommended_bid:>5}pt"
            f"  ~{int(r.predicted_threshold):>7}pt  "
            f"{r.risk_level:<10} {r.confidence_pct:>5.0f}%"
        )
        if r.note:
            lines.append(f"    → {r.note}")
    return "\n".join(lines)