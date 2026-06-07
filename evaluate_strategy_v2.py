"""
evaluate_strategy_v2.py
-----------------------
Evaluation script for Part 2: the survival-curve mileage strategy engine.

Why this exists:
  The proposal/feedback asks for objective metrics, baseline comparison, and
  simulation under realistic registration conditions. This evaluates the
  strategy engine as a decision-support system, not just a point predictor.

Method:
  - Temporal holdout: train/build curves from semesters BEFORE the target.
  - Evaluate recommendations against actual target-semester outcomes.
  - Compare against simple student baselines.

Default:
  Train:    all data before 2026-1
  Evaluate: 2026-1

Run:
  python3 evaluate_strategy_v2.py
  python3 evaluate_strategy_v2.py --scenarios 200 --courses-per-scenario 5
  python3 evaluate_strategy_v2.py --output-prefix /tmp/strategy_v2_eval
"""

from __future__ import annotations

import argparse
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from allocator import TOTAL_MILEAGE, MAX_BID
from strategy_engine_v2 import get_strategy_for_ranked_list


HISTORY_CSV = "mileage_history_all.csv"
JSON_PATH = "segmented_cs_courses.json"


@dataclass
class StrategyMetrics:
    strategy: str
    expected_wins: float
    weighted_success: float
    top1_success: float
    safe_coverage: float
    underbid_rate: float
    waste_points: float


def enrolled_bool(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.upper() == "Y").astype(int)


def load_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["semester"] = pd.to_numeric(df["semester"], errors="coerce")
    df["grade_year"] = pd.to_numeric(df["grade_year"], errors="coerce")
    df["mileage_bid"] = pd.to_numeric(df["mileage_bid"], errors="coerce")
    df["enrolled_bool"] = enrolled_bool(df["enrolled"])
    return df.dropna(subset=["course_code", "year", "semester"]).copy()


def temporal_split(df: pd.DataFrame, target_year: int, target_semester: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_key = target_year * 10 + target_semester
    keys = df["year"] * 10 + df["semester"]
    train = df[keys < target_key].copy()
    holdout = df[(df["year"] == target_year) & (df["semester"] == target_semester)].copy()
    return train, holdout


def write_temp_training_csv(train_df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_strategy_train.csv",
        prefix="sugang_",
        delete=False,
        encoding="utf-8-sig",
    )
    train_df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def build_candidate_offerings(holdout: pd.DataFrame, min_applicants: int) -> list[dict]:
    rows = []
    usable = holdout.dropna(subset=["mileage_bid"]).copy()
    for (code, prof), grp in usable.groupby(["course_code", "professor"], dropna=False):
        if len(grp) < min_applicants:
            continue
        name = grp["course_name"].dropna()
        rows.append({
            "code": code,
            "name": name.iloc[0] if len(name) else code,
            "professor": "" if pd.isna(prof) else str(prof),
            "n_applicants": len(grp),
            "capacity": float(grp["capacity"].dropna().iloc[0]) if grp["capacity"].notna().any() else np.nan,
            "total_applicants": float(grp["total_applicants"].dropna().iloc[0]) if grp["total_applicants"].notna().any() else len(grp),
        })
    return rows


def course_holdout_rows(holdout: pd.DataFrame, code: str, student_year: int | None) -> pd.DataFrame:
    sub = holdout[(holdout["course_code"] == code) & holdout["mileage_bid"].notna()].copy()
    if student_year is not None:
        year_sub = sub[sub["grade_year"] == student_year].copy()
        if len(year_sub) >= 5:
            return year_sub
    return sub


def actual_prob_at_or_above(holdout: pd.DataFrame, code: str, bid: int, student_year: int | None) -> float:
    if bid <= 0:
        return 0.0
    sub = course_holdout_rows(holdout, code, student_year)
    above = sub[sub["mileage_bid"] >= bid]
    if len(above) == 0:
        return 0.0
    return float(above["enrolled_bool"].mean())


def empirical_safe_bid(
    holdout: pd.DataFrame,
    code: str,
    student_year: int | None,
    target: float = 0.90,
) -> tuple[int, float, bool]:
    """
    Returns (safe_bid_or_knee, probability_at_that_bid, target_reachable).
    If 90% is unreachable because of quota/tie effects, return the cheapest bid
    within 2% of the max observed probability, matching the engine philosophy.
    """
    probs = [(bid, actual_prob_at_or_above(holdout, code, bid, student_year)) for bid in range(1, MAX_BID + 1)]
    for bid, prob in probs:
        if prob >= target:
            return bid, prob, True
    max_prob = max(prob for _, prob in probs) if probs else 0.0
    for bid, prob in probs:
        if prob >= max_prob - 0.02:
            return bid, prob, False
    return MAX_BID, max_prob, False


def uniform_baseline(n: int) -> list[int]:
    base = TOTAL_MILEAGE // n
    bids = [base] * n
    leftover = TOTAL_MILEAGE - sum(bids)
    for i in range(leftover):
        bids[i] += 1
    return [min(MAX_BID, b) for b in bids]


def priority_weighted_baseline(ranks: list[int]) -> list[int]:
    weights = np.array([1.0 / r for r in ranks], dtype=float)
    raw = weights / weights.sum() * TOTAL_MILEAGE
    bids = np.maximum(1, np.minimum(MAX_BID, np.round(raw).astype(int)))
    while bids.sum() > TOTAL_MILEAGE:
        for idx in np.argsort(-np.array(ranks)):
            if bids.sum() <= TOTAL_MILEAGE:
                break
            if bids[idx] > 1:
                bids[idx] -= 1
    while bids.sum() < TOTAL_MILEAGE:
        for idx in np.argsort(np.array(ranks)):
            if bids.sum() >= TOTAL_MILEAGE:
                break
            if bids[idx] < MAX_BID:
                bids[idx] += 1
    return bids.tolist()


def safe_greedy_oracle(holdout: pd.DataFrame, scenario: list[dict], student_year: int, target: float) -> list[int]:
    """
    Upper-bound baseline that knows the holdout safe bids. This is not a fair
    student baseline; it shows the best possible safe-first portfolio.
    """
    bids = [0] * len(scenario)
    remaining = TOTAL_MILEAGE
    for original_idx, course in sorted(enumerate(scenario), key=lambda row: row[1]["rank"]):
        need, _, _ = empirical_safe_bid(holdout, course["code"], student_year, target)
        give = min(need, remaining, MAX_BID)
        bids[original_idx] = give
        remaining -= give
        if remaining <= 0:
            break
    return bids


def evaluate_bids(
    holdout: pd.DataFrame,
    scenario: list[dict],
    bids: list[int],
    strategy_name: str,
    student_year: int,
    target: float,
) -> tuple[StrategyMetrics, list[dict]]:
    ranks = np.array([c["rank"] for c in scenario], dtype=float)
    priority_weights = (len(scenario) - ranks + 1) / len(scenario)

    detail_rows = []
    probs = []
    safe_hits = []
    underbids = []
    wastes = []

    for course, bid, weight in zip(scenario, bids, priority_weights):
        actual_prob = actual_prob_at_or_above(holdout, course["code"], int(bid), student_year)
        safe_bid, safe_prob, reachable = empirical_safe_bid(holdout, course["code"], student_year, target)
        safe_hit = int(int(bid) >= safe_bid)
        underbid = max(0, safe_bid - int(bid))
        waste = max(0, int(bid) - safe_bid)

        probs.append(actual_prob)
        safe_hits.append(safe_hit)
        underbids.append(1 if underbid > 0 else 0)
        wastes.append(waste)
        detail_rows.append({
            "strategy": strategy_name,
            "rank": course["rank"],
            "course_code": course["code"],
            "course_name": course["name"],
            "professor": course["professor"],
            "bid": int(bid),
            "actual_prob": round(actual_prob, 4),
            "empirical_safe_bid": safe_bid,
            "safe_target_reachable": reachable,
            "safe_hit": safe_hit,
            "underbid_points": underbid,
            "waste_points": waste,
            "priority_weight": round(float(weight), 4),
        })

    weighted_success = float(np.dot(priority_weights, probs) / priority_weights.sum())
    top1_idx = int(np.argmin(ranks))
    metrics = StrategyMetrics(
        strategy=strategy_name,
        expected_wins=float(np.sum(probs)),
        weighted_success=weighted_success,
        top1_success=float(probs[top1_idx]),
        safe_coverage=float(np.mean(safe_hits)),
        underbid_rate=float(np.mean(underbids)),
        waste_points=float(np.sum(wastes)),
    )
    return metrics, detail_rows


def make_scenarios(candidates: list[dict], n_scenarios: int, courses_per_scenario: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    scenarios = []
    if len(candidates) < courses_per_scenario:
        return scenarios
    for _ in range(n_scenarios):
        sampled = rng.sample(candidates, courses_per_scenario)
        ranks = list(range(1, courses_per_scenario + 1))
        rng.shuffle(ranks)
        scenario = []
        for course, rank in zip(sampled, ranks):
            row = dict(course)
            row["rank"] = rank
            scenario.append(row)
        scenario.sort(key=lambda item: item["rank"])
        scenarios.append(scenario)
    return scenarios


def evaluate_priority_responsiveness(
    scenarios: list[list[dict]],
    train_csv: str,
    json_path: str,
    student_year: int,
) -> dict:
    deltas = []
    changed = []
    for scenario in scenarios:
        if len(scenario) < 2:
            continue
        original = [dict(c) for c in scenario]
        base_results = get_strategy_for_ranked_list(
            original,
            {"year": student_year},
            raw_csv=train_csv,
            json_path=json_path,
        )
        base_bids = {r.code: r.bid for r in base_results}

        promoted = [dict(c) for c in scenario]
        promoted_code = promoted[-1]["code"]
        for item in promoted:
            if item["code"] == promoted_code:
                item["rank"] = 1
            else:
                item["rank"] += 1
        promoted_results = get_strategy_for_ranked_list(
            promoted,
            {"year": student_year},
            raw_csv=train_csv,
            json_path=json_path,
        )
        promoted_bids = {r.code: r.bid for r in promoted_results}
        delta = promoted_bids.get(promoted_code, 0) - base_bids.get(promoted_code, 0)
        deltas.append(delta)
        changed.append(1 if delta > 0 else 0)

    if not deltas:
        return {"mean_promoted_bid_delta": 0.0, "promotion_increase_rate": 0.0}
    return {
        "mean_promoted_bid_delta": float(np.mean(deltas)),
        "promotion_increase_rate": float(np.mean(changed)),
    }


def summarize(metrics_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(metrics_rows)
    agg = df.groupby("strategy").agg(
        expected_wins=("expected_wins", "mean"),
        weighted_success=("weighted_success", "mean"),
        top1_success=("top1_success", "mean"),
        safe_coverage=("safe_coverage", "mean"),
        underbid_rate=("underbid_rate", "mean"),
        waste_points=("waste_points", "mean"),
    ).reset_index()
    return agg.sort_values("weighted_success", ascending=False)


def run(args: argparse.Namespace) -> None:
    history_path = Path(args.history_csv)
    json_path = Path(args.json_path)
    if not history_path.exists():
        raise FileNotFoundError(f"Missing {history_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")

    df = load_history(str(history_path))
    train, holdout = temporal_split(df, args.target_year, args.target_semester)
    if train.empty or holdout.empty:
        raise ValueError("Temporal split produced empty train or holdout data.")

    candidates = build_candidate_offerings(holdout, args.min_applicants)
    scenarios = make_scenarios(candidates, args.scenarios, args.courses_per_scenario, args.seed)
    if not scenarios:
        raise ValueError("Not enough holdout courses to build evaluation scenarios.")

    train_csv = write_temp_training_csv(train)
    metrics_rows = []
    detail_rows = []
    ai_status_rows = []

    try:
        for scenario_id, scenario in enumerate(scenarios, start=1):
            ranked_list = [
                {
                    "code": c["code"],
                    "name": c["name"],
                    "rank": c["rank"],
                    "professor": c["professor"],
                }
                for c in scenario
            ]

            ai_results = get_strategy_for_ranked_list(
                ranked_list,
                {"year": args.student_year},
                raw_csv=train_csv,
                json_path=str(json_path),
                target_confidence=args.target_confidence,
            )
            ai_by_code = {r.code: r for r in ai_results}
            ai_bids = [ai_by_code[c["code"]].bid for c in scenario]

            baseline_bids = {
                "AI survival strategy": ai_bids,
                "Baseline uniform split": uniform_baseline(len(scenario)),
                "Baseline priority weighted": priority_weighted_baseline([c["rank"] for c in scenario]),
                "Oracle safe greedy": safe_greedy_oracle(holdout, scenario, args.student_year, args.target_confidence),
            }

            for strategy_name, bids in baseline_bids.items():
                metrics, details = evaluate_bids(
                    holdout,
                    scenario,
                    bids,
                    strategy_name,
                    args.student_year,
                    args.target_confidence,
                )
                row = metrics.__dict__
                row["scenario_id"] = scenario_id
                metrics_rows.append(row)
                for detail in details:
                    detail["scenario_id"] = scenario_id
                    detail_rows.append(detail)

            for c in scenario:
                r = ai_by_code[c["code"]]
                actual_prob = actual_prob_at_or_above(holdout, c["code"], r.bid, args.student_year)
                ai_status_rows.append({
                    "scenario_id": scenario_id,
                    "course_code": c["code"],
                    "rank": c["rank"],
                    "ai_bid": r.bid,
                    "ai_status": r.status,
                    "ai_predicted_prob": r.win_prob,
                    "actual_prob": actual_prob,
                    "prob_abs_error": abs(r.win_prob - actual_prob),
                    "unsafe_positive": int(r.status in ("Likely", "Secured") and actual_prob < args.likely_threshold),
                })

        summary = summarize(metrics_rows)
        status_df = pd.DataFrame(ai_status_rows)
        priority = evaluate_priority_responsiveness(
            scenarios[: min(len(scenarios), args.priority_probe_scenarios)],
            train_csv,
            str(json_path),
            args.student_year,
        )

        print("\nPART 2 STRATEGY ENGINE EVALUATION")
        print("=" * 72)
        print(f"Temporal holdout: train < {args.target_year}-{args.target_semester}, evaluate {args.target_year}-{args.target_semester}")
        print(f"Scenarios: {len(scenarios)} x {args.courses_per_scenario} courses")
        print(f"Student year: {args.student_year} | target confidence: {args.target_confidence:.0%}")

        print("\nPortfolio metrics, averaged across scenarios")
        print("-" * 72)
        printable = summary.copy()
        for col in ["expected_wins", "waste_points"]:
            printable[col] = printable[col].map(lambda x: f"{x:.2f}")
        for col in ["weighted_success", "top1_success", "safe_coverage", "underbid_rate"]:
            printable[col] = printable[col].map(lambda x: f"{x:.1%}")
        print(printable.to_string(index=False))

        print("\nAI honesty / probability realism")
        print("-" * 72)
        print(f"Probability MAE:       {status_df['prob_abs_error'].mean():.3f}")
        print(f"Unsafe positive rate:  {status_df['unsafe_positive'].mean():.1%}")
        print(f"  Definition: AI says Likely/Secured, but actual holdout P(enroll | bid) < {args.likely_threshold:.0%}")

        print("\nPriority responsiveness")
        print("-" * 72)
        print(f"Mean promoted-course bid delta: {priority['mean_promoted_bid_delta']:.2f} pts")
        print(f"Promotion increase rate:        {priority['promotion_increase_rate']:.1%}")

        print("\nRecommended report metrics")
        print("-" * 72)
        ai = summary[summary["strategy"] == "AI survival strategy"].iloc[0]
        uniform = summary[summary["strategy"] == "Baseline uniform split"].iloc[0]
        weighted = summary[summary["strategy"] == "Baseline priority weighted"].iloc[0]
        print(f"1. Weighted registration success: AI {ai['weighted_success']:.1%}, "
              f"uniform {uniform['weighted_success']:.1%}, priority baseline {weighted['weighted_success']:.1%}")
        print(f"2. Top-priority course success: AI {ai['top1_success']:.1%}")
        print(f"3. Safe coverage / underbid rate: {ai['safe_coverage']:.1%} / {ai['underbid_rate']:.1%}")
        print(f"4. Honesty: unsafe positive rate {status_df['unsafe_positive'].mean():.1%}, "
              f"probability MAE {status_df['prob_abs_error'].mean():.3f}")
        print(f"5. Priority sensitivity: promoted courses gained "
              f"{priority['mean_promoted_bid_delta']:.2f} pts on average")

        if args.output_prefix:
            prefix = Path(args.output_prefix)
            summary.to_csv(f"{prefix}_summary.csv", index=False)
            pd.DataFrame(metrics_rows).to_csv(f"{prefix}_scenario_metrics.csv", index=False)
            pd.DataFrame(detail_rows).to_csv(f"{prefix}_details.csv", index=False)
            status_df.to_csv(f"{prefix}_ai_honesty.csv", index=False)
            print("\nSaved CSV outputs:")
            print(f"  {prefix}_summary.csv")
            print(f"  {prefix}_scenario_metrics.csv")
            print(f"  {prefix}_details.csv")
            print(f"  {prefix}_ai_honesty.csv")
    finally:
        Path(train_csv).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the v2 survival strategy engine.")
    parser.add_argument("--history-csv", default=HISTORY_CSV)
    parser.add_argument("--json-path", default=JSON_PATH)
    parser.add_argument("--target-year", type=int, default=2026)
    parser.add_argument("--target-semester", type=int, default=1)
    parser.add_argument("--student-year", type=int, default=3)
    parser.add_argument("--target-confidence", type=float, default=0.90)
    parser.add_argument("--likely-threshold", type=float, default=0.85)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--courses-per-scenario", type=int, default=5)
    parser.add_argument("--min-applicants", type=int, default=15)
    parser.add_argument("--priority-probe-scenarios", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
