"""
allocator.py
============
Turns per-course survival curves into a full 72-point bidding strategy.

PHILOSOPHY (from the proposal — mathematically safe, no overbidding, no
underbidding, realistic about what's possible):

  1. For every course, find the CHEAPEST bid that hits the student's target
     confidence (default 90%) — the survival-curve "min safe bid".
  2. If the sum of all min-safe-bids <= 72, win everything: spend the minimum,
     then distribute LEFTOVER points to the riskiest courses for extra safety.
  3. If the sum > 72, the list is over-subscribed. Triage by priority, fund
     cheapest-first within order, and HONESTLY flag what can't be funded.

This inverts the naive "rank 1 gets the most points" heuristic: an important
course that's cheap to win (large class, low demand) gets a small bid, freeing
points for genuinely contested courses.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from survival_model import (
    load_applicant_data, load_course_meta, build_curve,
    confidence_label, SurvivalCurve, MAX_BID,
)

TOTAL_MILEAGE = 72
DEFAULT_TARGET = 0.90
STRETCH_TARGET = 0.97


@dataclass
class CourseBid:
    rank: int
    code: str
    name: str
    professor: str
    bid: int
    win_prob: float
    min_safe_bid: int
    confidence: str
    confidence_pct: float
    competition: str
    status: str          # Secured / Likely / Risky / Drop
    note: str
    data_source: str


def _competition_from_curve(curve: SurvivalCurve, target=DEFAULT_TARGET) -> str:
    msb = curve.min_bid_for(target)
    return _competition_label(msb)


def _competition_label(safe_bid: int) -> str:
    if safe_bid <= 3:  return "Easy"
    if safe_bid <= 10: return "Moderate"
    if safe_bid <= 20: return "Hard"
    return "Very Hard"


def _status(win_prob: float, bid: int | None = None, min_safe_bid: int | None = None) -> str:
    if bid is not None and min_safe_bid is not None and bid < min_safe_bid:
        if win_prob >= 0.60:
            return "Risky"
        return "Drop"
    if win_prob >= 0.95: return "Secured"
    if win_prob >= 0.85: return "Likely"
    if win_prob >= 0.60: return "Risky"
    return "Drop"


def build_strategy(
    ranked_list: list,
    student_year: int = 3,
    target_conf: float = DEFAULT_TARGET,
    raw_csv: str = "mileage_history_all.csv",
    json_path: str = "segmented_cs_courses.json",
) -> list[CourseBid]:
    df   = load_applicant_data(raw_csv)
    meta = load_course_meta(json_path, raw_csv)
    items = sorted(ranked_list, key=lambda x: x.get("rank", 99))

    curves, msb = {}, {}
    for it in items:
        code = it["code"]
        curve = build_curve(df, code, it.get("professor", ""),
                            student_year=student_year, meta=meta)
        curves[code] = curve
        msb[code] = curve.min_bid_for(target_conf)

        # ── Demand-ratio sanity check ─────────────────────────────────────
        # When total applicants < capacity, the course is structurally
        # undersubscribed. The few people who didn't get in lost to priority
        # seating or major-quota edge cases, not mileage competition. In that
        # case the survival curve can overestimate the safe bid because small
        # samples at high bid levels create artificial jumps (e.g. 8/8 = 100%
        # at bid=20 vs 29/34 = 85% at bid=3). Cap the safe bid based on what
        # the curve shows at the plateau level.
        info = meta.get(code, {})
        applicants = info.get("eta_added", 0)   # actually 지원자 count
        capacity   = info.get("capacity", 0)
        if capacity > 0 and applicants > 0 and applicants <= capacity:
            # Undersubscribed — find the cheapest bid achieving the plateau
            plateau_p = curve.prob_at(3)  # P(enroll) at a low bid
            if plateau_p >= 0.80:
                # Course is easy — most people get in even at low bids
                # Cap safe bid: cheapest bid where P >= plateau level
                capped = curve.min_bid_for(plateau_p)
                if capped < msb[code]:
                    msb[code] = max(capped, 1)

    total_min = sum(msb[it["code"]] for it in items)
    alloc = {it["code"]: 0 for it in items}
    budget_note = None

    if total_min <= TOTAL_MILEAGE:
        for it in items:
            alloc[it["code"]] = msb[it["code"]]
        leftover = TOTAL_MILEAGE - total_min
        # Greedy: give each leftover point to wherever it buys the most extra
        # win probability (the riskiest course), below the stretch target.
        while leftover > 0:
            best_code, best_gain = None, 0.0
            for it in items:
                code = it["code"]; cur = alloc[code]
                if cur >= MAX_BID:
                    continue
                cur_p = curves[code].prob_at(cur)
                nxt_p = curves[code].prob_at(cur + 1)
                priority_weight = 1.0 / max(float(it.get("rank", 99)), 1.0)
                gain = (nxt_p - cur_p) * (1.0 + priority_weight)
                if cur_p < STRETCH_TARGET and gain > best_gain:
                    best_gain, best_code = gain, code
            if best_code is None or best_gain <= 0.0005:
                break
            alloc[best_code] += 1
            leftover -= 1

        # After smart distribution, if points STILL remain, push up bids
        # anyway — unused points are wasted (don't carry over between
        # semesters). At this point the remaining points are mostly symbolic
        # insurance, so respect the user's stated priority order.
        if leftover > 0:
            for it in items:
                code = it["code"]
                room = MAX_BID - alloc[code]
                add = min(room, leftover)
                alloc[code] += add
                leftover -= add
                if leftover <= 0:
                    break
    else:
        remaining = TOTAL_MILEAGE
        funded_codes = set()
        # Pass 1: priority order, fund what fits
        for it in items:
            code = it["code"]; need = msb[code]
            if need <= remaining:
                alloc[code] = need; remaining -= need; funded_codes.add(code)
            elif remaining > 0 and curves[code].prob_at(remaining) >= 0.60:
                alloc[code] = remaining; remaining = 0; funded_codes.add(code)
        # Pass 2: opportunistic — if points remain, fund the cheapest still-
        # unfunded courses that fit (a smart student grabs cheap wins even if
        # they're lower priority, rather than leaving points unused).
        if remaining > 0:
            leftovers = sorted(
                [it for it in items if it["code"] not in funded_codes],
                key=lambda x: msb[x["code"]])
            for it in leftovers:
                code = it["code"]; need = msb[code]
                if need <= remaining:
                    alloc[code] = need; remaining -= need; funded_codes.add(code)
        budget_note = (
            f"⚠️ This {len(items)}-course list needs ~{total_min}pt to win all at "
            f"{int(target_conf*100)}% confidence, but you only have {TOTAL_MILEAGE}pt. "
            f"Funded by priority, then cheap wins squeezed in; rest flagged honestly."
        )

    out = []
    for it in items:
        code = it["code"]; curve = curves[code]; bid = int(alloc[code])
        wp = curve.prob_at(bid) if bid > 0 else 0.0
        conf, conf_pct = confidence_label(curve)
        comp = _competition_label(msb[code])
        status = _status(wp, bid, msb[code]) if bid > 0 else "Unfunded"

        if bid == 0:
            note = (f"Not funded — needs ~{msb[code]}pt to be safe but budget ran out "
                    f"on higher-priority courses. Realistically not winnable with this list.")
        elif curve.cutoff_floor_reason and bid <= msb[code]:
            note = (f"{bid}pt → calibrated odds ~{wp:.0%}. {curve.cutoff_floor_reason} "
                    f"Treat this as the minimum defensive bid, not a guarantee.")
        elif bid > msb[code] + 3:
            # Bid is well above safe — check if extra points actually help
            safe_p = curves[code].prob_at(msb[code])
            if abs(wp - safe_p) < 0.02:
                # Plateau — extra points don't change odds at all
                if wp >= 0.95:
                    note = (f"Safe at ~{msb[code]}pt ({safe_p:.0%}). Extra "
                            f"{bid - msb[code]}pt is priority/leftover insurance; "
                            f"unused points don't carry over.")
                else:
                    note = (f"Safe at ~{msb[code]}pt ({safe_p:.0%}). Bidding higher doesn't improve "
                            f"odds — {wp:.0%} is the ceiling for year-{student_year} students "
                            f"(year/graduation quota). Extra {bid - msb[code]}pt is leftover redistribution; "
                            f"unused points don't carry over.")
            else:
                if curves[code].cutoff_floor_reason:
                    note = (f"~{msb[code]}pt is the defensive cutoff from recent same-year outcomes "
                            f"(calibrated odds there ~{safe_p:.0%}). Extra {bid - msb[code]}pt "
                            f"pushes calibrated odds to ~{wp:.0%}. Unused points don't carry over, "
                            f"so they're spent as insurance.")
                else:
                    note = (f"Safe at ~{msb[code]}pt ({safe_p:.0%}), but {bid - msb[code]}pt extra "
                            f"redistributed here from leftover budget pushes odds to {wp:.0%}. "
                            f"Unused points don't carry over, so they're spent as insurance.")
        elif status == "Secured":
            if msb[code] <= 3 and curve.source != "demand_proxy":
                note = (f"Low contention — {bid}pt secures it. No need to overspend.")
            else:
                note = f"{bid}pt → ~{wp:.0%} odds. Safely secured."
        elif status == "Likely":
            note = f"{bid}pt → calibrated odds ~{wp:.0%}. Good chance, but still not guaranteed."
        elif status == "Risky":
            reach = curve.reachable_max()
            if reach < 0.85 and bid >= msb[code]:
                note = (f"{bid}pt → calibrated odds ~{wp:.0%}, and that's about the ceiling — for a year-{student_year} "
                        f"student this course caps around {reach:.0%} no matter how high you bid "
                        f"(seats go by year/graduation quota). Bidding past ~{msb[code]}pt is wasted.")
            else:
                note = (f"{bid}pt → calibrated odds ~{wp:.0%}. Risky: the safe number is ~{msb[code]}pt.")
        else:
            note = f"{bid}pt → calibrated odds only ~{wp:.0%}. Recommend dropping or raising toward ~{msb[code]}pt."

        if curve.source == "demand_proxy":
            note += " (No bid history — estimated from ETA demand, rough.)"
        elif curve.source in ("same_course", "same_course_any_prof"):
            note += f" (Course history; prof {it.get('professor','?')} not directly matched.)"

        out.append(CourseBid(
            rank=it["rank"], code=code, name=it.get("name", code),
            professor=it.get("professor", ""), bid=bid, win_prob=wp,
            min_safe_bid=msb[code], confidence=conf, confidence_pct=conf_pct,
            competition=comp, status=status, note=note, data_source=curve.source,
        ))

    if budget_note and out:
        out[0].note = budget_note + "\n" + out[0].note
    return out


def format_strategy(results: list[CourseBid]) -> str:
    if not results:
        return "No strategy could be generated."
    used = sum(r.bid for r in results)
    STATUS = {"Secured": "🟢", "Likely": "🟡", "Risky": "🟠", "Drop": "⛔", "Unfunded": "⛔"}
    CONF = {"High": "📊", "Medium": "🔍", "Low": "❓"}
    lines = [
        "## 🎯 Mileage Strategy\n",
        f"**{used}/{TOTAL_MILEAGE} pts used** — funds safe bids first, "
        f"then spends leftover as priority/risk insurance.\n",
    ]
    for r in results:
        s = STATUS.get(r.status, "⚪"); c = CONF.get(r.confidence, "")
        lines.append(
            f"**{r.rank}. {r.name}** ({r.professor})\n"
            f"   Bid **{r.bid}pt** {s} {r.status} — calibrated odds ~{r.win_prob:.0%}  "
            f"| safe bid ~{r.min_safe_bid}pt [{r.competition}]  "
            f"| {c} {r.confidence} data ({r.confidence_pct:.0f}%)"
        )
        for ln in r.note.split("\n"):
            if ln.strip():
                lines.append(f"   > {ln.strip()}")
        lines.append("")
    return "\n".join(lines)

