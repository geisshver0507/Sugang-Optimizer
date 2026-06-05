"""Shared helpers for course text normalization, display formatting, and time expansion."""

import re


PERIOD_TIME_RANGES = {
    1: "9:00 AM-10:00 AM",
    2: "10:00 AM-11:00 AM",
    3: "11:00 AM-12:00 PM",
    4: "12:00 PM-1:00 PM",
    5: "1:00 PM-2:00 PM",
    6: "2:00 PM-3:00 PM",
    7: "3:00 PM-4:00 PM",
    8: "4:00 PM-5:00 PM",
    9: "5:00 PM-6:00 PM",
    10: "6:00 PM-7:00 PM",
    11: "7:00 PM-8:00 PM",
    12: "8:00 PM-9:00 PM",
    13: "9:00 PM-10:00 PM",
}

DAY_PATTERN = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b\s*([^A-Za-z]*)")


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value)
    return str(value).strip().lower()


def clean_focus_area(area):
    return normalize_text(str(area).split(" (")[0])


def period_to_time(period):
    try:
        period_num = int(period)
    except (TypeError, ValueError):
        return "Unknown time"
    return PERIOD_TIME_RANGES.get(period_num, "Unknown time")


def expand_course_time(time_value):
    raw_time = str(time_value or "").strip()
    if not raw_time:
        return "Not listed"

    expanded_days = []
    for day, period_text in DAY_PATTERN.findall(raw_time):
        periods = [int(value) for value in re.findall(r"\d+", period_text)]
        if not periods:
            continue
        expanded_periods = [
            f"period {period} ({period_to_time(period)})"
            for period in periods
        ]
        expanded_days.append(f"{day}: " + ", ".join(expanded_periods))

    if not expanded_days:
        return raw_time
    return "; ".join(expanded_days)


def course_search_blob(code, course_obj):
    meta = course_obj.get("metadata", {})
    chunks = course_obj.get("text_chunks", {})
    parts = [code, normalize_text(expand_course_time(meta.get("time")))]
    for value in meta.values():
        parts.append(normalize_text(value))
    for value in chunks.values():
        parts.append(normalize_text(value))
    return " ".join(parts)


def clip_text(text, limit=520):
    cleaned = " ".join(str(text or "Not listed.").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + "..."


def format_field(value):
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "Not listed"
    if value in (None, "", []):
        return "Not listed"
    return str(value)
