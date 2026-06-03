"""
optimizer.py
------------
Portfolio optimizer: given competition thresholds per course and a
fixed mileage budget, allocates points to maximise the probability
of getting into the student's highest-priority courses.

This is the MATH component — not AI. It takes the model's competition
estimates as inputs and solves a constrained allocation problem.

The two key tensions it balances:
  1. Competition vs priority  — a rank-3 course might be MORE competitive
                                than rank-1 and need a higher bid despite
                                being less preferred
  2. Safety vs efficiency     — overbidding wastes points; underbidding
                                loses the course entirely
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class CourseInput:
    code:                   str
    name:                   str
    rank:                   int     # 1 = highest priority to student
    predicted_threshold:    float   # model's estimate of min winning bid
    is_major_req:           bool    = False
    shap_breakdown:         dict    = field(default_factory=dict)


@dataclass
class BidResult:
    code:                   str
    name:                   str
    rank:                   int
    recommended_bid:        int
    predicted_threshold:    float
    risk_level:             str     # Safe / Moderate / Risky
    confidence_pct:         float
    shap_breakdown:         dict    = field(default_factory=dict)
    note:                   str     = ""


def allocate_bids(
    courses:        list,           # list[CourseInput]
    total_mileage:  int,
    safety_margin:  float = 0.15,   # bid this % above threshold as buffer
    min_bid:        int   = 1,
) -> list:                          # list[BidResult]
    """
    Allocate total_mileage across courses, balancing:
      - competition difficulty  (predicted_threshold — higher = needs more pts)
      - student priority        (rank — lower rank number = more important)
      - budget constraints      (sum of bids ≤ total_mileage)

    ALGORITHM:
    Step 1 — Compute raw target bid per course
             raw_bid = predicted_threshold × (1 + safety_margin)
             This is the ideal bid ignoring budget.

    Step 2 — Compute a combined weight per course
             competition_weight = predicted_threshold / sum(thresholds)
               → courses that NEED more points get proportionally more
             priority_weight = 1 / rank^0.5
               → higher-priority courses also get a boost
             combined = 0.6 × competition_weight + 0.4 × priority_weight
               → competition is the dominant factor (you can't get in by
                 wanting it more if you don't bid enough), but priority
                 still influences allocation at the margin

    Step 3 — Scale to budget
             allocated = combined_weight × total_mileage
             Cap each course at its raw_bid (no overbidding beyond buffer)

    Step 4 — Redistribute leftover
             Points saved from capped courses go to the most
             competitive courses that are still below their raw_bid.
             Priority-weighted so rank-1 benefits most.

    Step 5 — Enforce budget exactly
             Round to integers, fix any drift by adjusting
             lowest-priority courses first.

    Step 6 — Assign risk levels
             bid / predicted_threshold ratio determines risk.
    """

    if not courses:
        return []

    n = len(courses)

    thresholds = np.array([c.predicted_threshold for c in courses], dtype=float)
    ranks      = np.array([c.rank               for c in courses], dtype=float)

    # ── Step 1: Raw target bids ───────────────────────────────────────────
    raw_bids = thresholds * (1.0 + safety_margin)

    # ── Step 2: Combined weights ──────────────────────────────────────────
    # Competition weight: courses with higher predicted thresholds need more pts
    if thresholds.sum() > 0:
        comp_weight = thresholds / thresholds.sum()
    else:
        comp_weight = np.ones(n) / n

    # Priority weight: rank 1 gets most, drops off with 1/rank^0.5
    prio_weight = 1.0 / (ranks ** 0.5)
    prio_weight = prio_weight / prio_weight.sum()

    # Combined: 60% competition-driven, 40% priority-driven
    # Rationale: you MUST meet the competition threshold to get in,
    # so competition dominates. But at the margin, the student's
    # preference breaks ties.
    combined = 0.6 * comp_weight + 0.4 * prio_weight

    # ── Step 3: Initial allocation ────────────────────────────────────────
    allocated = combined * total_mileage
    # Never allocate more than the raw target (no point overbidding further)
    allocated = np.minimum(allocated, raw_bids)

    # ── Step 4: Redistribute leftover ────────────────────────────────────
    leftover = total_mileage - allocated.sum()

    if leftover > 0:
        # How much each course still needs to reach its raw_bid
        deficit = np.maximum(0.0, raw_bids - allocated)
        if deficit.sum() > 0:
            # Redistribute proportionally to deficit × priority
            redistrib_weight = deficit * prio_weight
            redistrib_weight = redistrib_weight / redistrib_weight.sum()
            extra = redistrib_weight * leftover
            # Still cap at raw_bid
            allocated = np.minimum(allocated + extra, raw_bids)

    # ── Step 5: Enforce budget exactly ───────────────────────────────────
    # Scale down proportionally if over budget
    if allocated.sum() > total_mileage:
        allocated = allocated * (total_mileage / allocated.sum())

    # Floor at min_bid
    allocated = np.maximum(allocated, float(min_bid))

    # Round to integers
    int_bids = np.round(allocated).astype(int)

    # Fix rounding drift by adjusting lowest-priority courses
    drift = int_bids.sum() - total_mileage
    order = np.argsort(-ranks)   # lowest priority first (highest rank number)

    if drift > 0:
        for idx in order:
            if int_bids[idx] > min_bid and drift > 0:
                int_bids[idx] -= 1
                drift -= 1
    elif drift < 0:
        for idx in np.argsort(ranks):   # highest priority first
            int_bids[idx] += 1
            drift += 1
            if drift == 0:
                break

    # ── Step 6: Risk assessment ───────────────────────────────────────────
    results = []
    for i, course in enumerate(courses):
        bid   = int(int_bids[i])
        pred  = course.predicted_threshold
        ratio = bid / max(pred, 1.0)

        if ratio >= 1.10:
            risk, confidence = "Safe",     min(95.0, 55.0 + ratio * 25.0)
        elif ratio >= 0.90:
            risk, confidence = "Moderate", 40.0 + ratio * 20.0
        else:
            risk, confidence = "Risky",    max(10.0, ratio * 50.0)

        note = ""
        if course.is_major_req and ratio < 0.85:
            note = "⚠ Major requirement — strongly consider bidding higher."
        elif ratio > 1.4:
            note = "✓ Comfortably above threshold — you may have room to save points."
        elif confidence < 35:
            note = "⚠ Budget too spread thin — consider dropping a lower-priority course."

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


def strategy_summary(results: list, total_mileage: int) -> str:
    used = sum(r.recommended_bid for r in results)
    lines = [
        f"Strategy: {len(results)} courses | {used}/{total_mileage} pts used\n",
        f"{'#':<3} {'Course':<44} {'Bid':>6}  {'Threshold':>10}  {'Risk':<10} {'Conf':>6}",
        "─" * 82,
    ]
    for r in results:
        name = r.name[:42] + ".." if len(r.name) > 44 else r.name
        lines.append(
            f"{r.rank:<3} {name:<44} {r.recommended_bid:>5}pt"
            f"  ~{int(r.predicted_threshold):>7}pt  "
            f"{r.risk_level:<10} {r.confidence_pct:>5.0f}%"
        )
        if r.note:
            lines.append(f"    → {r.note}")
    lines.append(f"\nUnused: {total_mileage - used} pts")
    return "\n".join(lines)
