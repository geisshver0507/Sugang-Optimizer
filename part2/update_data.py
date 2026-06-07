"""Enrich course JSON with classroom capacity from the 2026-1 holdout CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_HOLDOUT = ROOT / "mileage_holdout_2026_1.csv"
DEFAULT_JSON = ROOT / "segmented_cs_courses.json"

CAPACITY_CANDIDATES = (
    "max_capacity",
    "maximum_capacity",
    "capacity",
    "class_capacity",
    "inferred_class_size",
    "competitive_seats",
)


def _base_code(course_code: Any) -> str:
    text = str(course_code or "").strip()
    return text.split("-")[0] if text else ""


def _choose_capacity_column(df: pd.DataFrame) -> str:
    for column in CAPACITY_CANDIDATES:
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            if numeric.notna().any():
                return column
    raise ValueError(
        "No capacity column found. Expected one of: "
        + ", ".join(CAPACITY_CANDIDATES)
    )


def extract_capacity_map(holdout_csv: str | Path = DEFAULT_HOLDOUT) -> tuple[dict[str, int], str]:
    """Return {course_code: max_capacity} from the holdout CSV."""
    holdout_path = Path(holdout_csv)
    df = pd.read_csv(holdout_path, encoding="utf-8-sig")
    if "course_code" not in df.columns:
        raise ValueError("holdout CSV must contain course_code")

    capacity_col = _choose_capacity_column(df)
    work = df[["course_code", capacity_col]].copy()
    work["course_code"] = work["course_code"].astype(str).str.strip()
    work["max_capacity"] = pd.to_numeric(work[capacity_col], errors="coerce")
    work = work.dropna(subset=["course_code", "max_capacity"])
    work = work[work["max_capacity"] > 0]

    grouped = work.groupby("course_code")["max_capacity"].max().round().astype(int)
    return grouped.to_dict(), capacity_col


def inject_capacity_into_json(
    capacity_by_code: dict[str, int],
    json_path: str | Path = DEFAULT_JSON,
    output_path: str | Path | None = None,
    source_column: str = "unknown",
) -> dict[str, int]:
    """Write max_capacity into every matching course metadata block."""
    json_path = Path(json_path)
    output_path = Path(output_path) if output_path else json_path

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    by_base: dict[str, list[int]] = {}
    for code, capacity in capacity_by_code.items():
        by_base.setdefault(_base_code(code), []).append(int(capacity))
    base_fallback = {
        base: int(round(sum(values) / len(values)))
        for base, values in by_base.items()
        if values
    }

    exact_updates = 0
    fallback_updates = 0
    missing = 0

    for year_bucket in data.values():
        for category_bucket in year_bucket.values():
            for code, course in category_bucket.items():
                metadata = course.setdefault("metadata", {})
                capacity = capacity_by_code.get(code)
                source = "holdout_exact"
                if capacity is None:
                    capacity = base_fallback.get(_base_code(code))
                    source = "holdout_base_average"

                if capacity is None:
                    metadata.setdefault("max_capacity", None)
                    metadata.setdefault("capacity_source", "not_available")
                    missing += 1
                    continue

                metadata["max_capacity"] = int(capacity)
                metadata["capacity_source"] = source
                metadata["capacity_source_column"] = source_column
                if source == "holdout_exact":
                    exact_updates += 1
                else:
                    fallback_updates += 1

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "exact_updates": exact_updates,
        "fallback_updates": fallback_updates,
        "missing_capacity": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject max_capacity into segmented_cs_courses.json")
    parser.add_argument("--holdout", default=str(DEFAULT_HOLDOUT))
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--output", default=None, help="Defaults to overwriting --json")
    args = parser.parse_args()

    capacity_by_code, source_column = extract_capacity_map(args.holdout)
    stats = inject_capacity_into_json(
        capacity_by_code,
        json_path=args.json,
        output_path=args.output,
        source_column=source_column,
    )

    print(f"Capacity source column: {source_column}")
    print(f"Courses with capacity in holdout: {len(capacity_by_code)}")
    print(
        "JSON updates: "
        f"{stats['exact_updates']} exact, "
        f"{stats['fallback_updates']} base-code fallback, "
        f"{stats['missing_capacity']} missing"
    )


if __name__ == "__main__":
    main()

