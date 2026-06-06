"""Build and retrieve course competitiveness data from mileage history."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from course_utils import display_course_name, format_field, normalize_text


DATASET_PATH = Path("course_competitiveness.json")
COURSE_SUMMARY_CSV = Path("course_summary_all.csv")
MILEAGE_HISTORY_CSV = Path("mileage_history_all.csv")


def _to_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    return int(round(_to_float(value, default)))


def _course_key(row):
    return (
        str(row.get("course_name", "")).strip(),
        str(row.get("professor", "")).strip(),
        str(row.get("year", "")).strip(),
        str(row.get("semester", "")).strip(),
    )


def _name_key(value):
    return re.sub(r"[^a-z0-9]+", " ", display_course_name(value).lower()).strip()


def competition_level(applicants, capacity, avg_mileage):
    ratio = applicants / max(capacity, 1)
    if ratio >= 1.2 or avg_mileage >= 30:
        return "High"
    if ratio >= 0.8 or avg_mileage >= 15:
        return "Moderate"
    return "Low"


def build_course_competitiveness(
    summary_csv=COURSE_SUMMARY_CSV,
    history_csv=MILEAGE_HISTORY_CSV,
    output_path=DATASET_PATH,
):
    with open(summary_csv, newline="", encoding="utf-8-sig") as handle:
        summary_rows = list(csv.DictReader(handle))
    with open(history_csv, newline="", encoding="utf-8-sig") as handle:
        history_rows = list(csv.DictReader(handle))

    offerings = {}
    for row in summary_rows:
        key = _course_key(row)
        if not key[0]:
            continue
        offering = offerings.setdefault(
            key,
            {
                "course_name": key[0],
                "year": _to_int(key[2]),
                "semester": _to_int(key[3]),
                "professor": key[1],
                "capacity": 0,
                "summary_applicants": 0,
                "summary_avg_mileage": 0.0,
                "summary_rows": 0,
            },
        )
        offering["capacity"] += _to_int(row.get("capacity"))
        offering["summary_applicants"] += _to_int(row.get("total_applicants"))
        offering["summary_avg_mileage"] += _to_float(row.get("avg_mileage"))
        offering["summary_rows"] += 1

    bids_by_offering = defaultdict(list)
    for row in history_rows:
        bid = _to_float(row.get("mileage_bid"), None)
        if bid is None:
            continue
        bids_by_offering[_course_key(row)].append(bid)

    records = []
    for key, offering in offerings.items():
        bids = bids_by_offering.get(key, [])
        capacity = offering["capacity"]
        applicants = len(bids) if bids else offering["summary_applicants"]
        if bids:
            avg_mileage = round(sum(bids) / len(bids), 2)
            accepted = sorted(bids, reverse=True)[: max(capacity, 0)]
            estimated_cutoff = int(min(accepted)) if accepted else 0
        else:
            avg_mileage = round(
                offering["summary_avg_mileage"] / max(offering["summary_rows"], 1),
                2,
            )
            estimated_cutoff = 0

        records.append(
            {
                "course_name": offering["course_name"],
                "year": offering["year"],
                "semester": offering["semester"],
                "professor": offering["professor"],
                "avg_mileage": avg_mileage,
                "estimated_cutoff": estimated_cutoff,
                "applicants": applicants,
                "capacity": capacity,
                "competition_level": competition_level(applicants, capacity, avg_mileage),
            }
        )

    records.sort(key=lambda item: (item["course_name"], item["year"], item["semester"], item["professor"]))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return records


def is_historical_mileage_query(prompt):
    text = normalize_text(prompt)
    history_terms = ("history", "historical", "past", "previous", "trend", "semester")
    mileage_terms = ("mileage", "bid", "bidding", "cutoff", "competition", "competitive")
    direct_competition_terms = ("cutoff", "competition", "competitive")
    return (
        any(term in text for term in direct_competition_terms)
        or (
            any(term in text for term in history_terms)
            and any(term in text for term in mileage_terms)
        )
    )


def load_course_competitiveness(path=DATASET_PATH):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def records_for_schedule(schedule, records=None):
    records = records if records is not None else load_course_competitiveness()
    if not schedule or not records:
        return []

    selected_names = {_name_key(course.get("course_name")) for course in schedule.values()}
    selected_professors = {normalize_text(course.get("professor")) for course in schedule.values()}

    matches = []
    for record in records:
        if _name_key(record.get("course_name")) not in selected_names:
            continue
        professor = normalize_text(record.get("professor"))
        if professor and selected_professors and professor in selected_professors:
            matches.insert(0, record)
        else:
            matches.append(record)
    return matches


def format_competitiveness_context(records):
    if not records:
        return "No matching course competitiveness records were found."
    rows = []
    for record in records:
        rows.append(
            f"{record['course_name']} | year {record['year']} | semester {record['semester']} | "
            f"professor {format_field(record.get('professor'))} | avg mileage {record['avg_mileage']} | "
            f"applicants {record['applicants']} | capacity {record['capacity']} | "
            f"estimated cutoff {record['estimated_cutoff']} | competition {record['competition_level']}"
        )
    return "\n".join(rows)


def build_competitiveness_prompt(records):
    return f"""
You are NightHawk AI. The user has confirmed their timetable and is asking about historical mileage competitiveness.
Use only the COURSE COMPETITIVENESS RECORDS below. Do not display raw JSON.

Requirements:
- Mention each semester separately.
- Mention the professor if available.
- Mention average mileage, applicants, capacity, estimated cutoff, and competition level.
- Compare semesters when multiple historical offerings exist for the same course.
- Identify increasing or decreasing competitiveness trends only when the records support it.
- If only one semester is available, say that trend direction cannot be determined from one offering.

COURSE COMPETITIVENESS RECORDS:
{format_competitiveness_context(records)}
"""


if __name__ == "__main__":
    built = build_course_competitiveness()
    print(f"Wrote {DATASET_PATH} with {len(built)} records.")
