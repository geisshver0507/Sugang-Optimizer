"""Helpers for chatbot-controlled course timetable state."""

import json
import re

from course_utils import display_course_name, extract_time_slots


DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DAY_ALIASES = {
    "mon": "Mon",
    "monday": "Mon",
    "tue": "Tue",
    "tues": "Tue",
    "tuesday": "Tue",
    "wed": "Wed",
    "wednesday": "Wed",
    "thu": "Thu",
    "thur": "Thu",
    "thurs": "Thu",
    "thursday": "Thu",
    "fri": "Fri",
    "friday": "Fri",
}
PERIOD_START_HOUR = {
    1: 9,
    2: 10,
    3: 11,
    4: 12,
    5: 13,
    6: 14,
    7: 15,
    8: 16,
    9: 17,
    10: 18,
    11: 19,
    12: 20,
    13: 21,
}
COURSE_COLORS = [
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#0f766e",
    "#a16207",
]

SCHEDULE_EDIT_RE = re.compile(
    r"\b("
    r"add|put|include|insert|remove|drop|delete|take out|swap|replace|change|"
    r"optimize|build|make|create|update|schedule"
    r")\b",
    re.IGNORECASE,
)
SCHEDULE_TARGET_RE = re.compile(
    r"\b(timetable|schedule|calendar|selected courses|my courses|course list)\b",
    re.IGNORECASE,
)
COURSE_ACTION_RE = re.compile(
    r"\b(add|put|include|insert|remove|drop|delete|swap|replace)\b",
    re.IGNORECASE,
)


def is_schedule_edit_request(text):
    """Return True only when the user is explicitly asking to edit the timetable."""
    user_text = str(text or "")
    lowered = user_text.lower()
    if not SCHEDULE_EDIT_RE.search(user_text):
        return False
    if COURSE_ACTION_RE.search(user_text):
        return True
    if "optimize" in lowered and SCHEDULE_TARGET_RE.search(user_text):
        return True
    if SCHEDULE_TARGET_RE.search(user_text) and any(
        word in lowered for word in ("build", "make", "create", "update", "change")
    ):
        return True
    return False


def normalize_action_name(action_name):
    value = str(action_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "add": "add_course",
        "add_course": "add_course",
        "remove": "remove_course",
        "delete": "remove_course",
        "drop": "remove_course",
        "remove_course": "remove_course",
        "optimize": "optimize_courses",
        "optimize_course": "optimize_courses",
        "optimize_courses": "optimize_courses",
    }
    return aliases.get(value, value)


def course_id_from_action(action):
    course = action.get("course") or {}
    return (
        action.get("course_id")
        or action.get("course id")
        or course.get("course_id")
        or course.get("course id")
        or course.get("id")
        or course.get("code")
    )


def normalize_day(day):
    return DAY_ALIASES.get(str(day or "").strip().lower())


def hour_to_label(hour):
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour if hour <= 12 else hour - 12
    return f"{display_hour}:00 {suffix}"


def period_to_clock(period):
    start_hour = PERIOD_START_HOUR.get(int(period))
    if start_hour is None:
        return None
    return hour_to_label(start_hour), hour_to_label(start_hour + 1)


def parse_time_to_meetings(time_value):
    """Convert database period strings into renderable day/time meetings."""
    raw_time = str(time_value or "").strip()
    if not raw_time:
        return []

    meetings = []
    for day, period_text in re.findall(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b\s*([^A-Za-z]*)", raw_time):
        normalized_day = normalize_day(day)
        if not normalized_day or normalized_day not in DAY_LABELS:
            continue
        periods = sorted({int(value) for value in re.findall(r"\d+", period_text)})
        if not periods:
            continue

        start = previous = periods[0]
        ranges = []
        for period in periods[1:]:
            if period == previous + 1:
                previous = period
                continue
            ranges.append((start, previous))
            start = previous = period
        ranges.append((start, previous))

        for start_period, end_period in ranges:
            start_time, _ = period_to_clock(start_period)
            _, end_time = period_to_clock(end_period)
            meetings.append({
                "day": normalized_day,
                "start_period": start_period,
                "end_period": end_period,
                "start_time": start_time,
                "end_time": end_time,
            })
    return meetings


def build_schedule_course(code, course_obj, color_index=0):
    meta = course_obj.get("metadata", {})
    return {
        "course_id": code,
        "course_name": display_course_name(meta.get("name", code)),
        "professor": meta.get("professor", "Not listed"),
        "credits": int(meta.get("credits") or 0),
        "max_capacity": meta.get("max_capacity"),
        "time": meta.get("time", ""),
        "location": meta.get("location", "Not listed"),
        "days": sorted({meeting["day"] for meeting in parse_time_to_meetings(meta.get("time"))}),
        "meetings": parse_time_to_meetings(meta.get("time")),
        "slots": sorted(extract_time_slots(meta.get("time"))),
        "color": COURSE_COLORS[color_index % len(COURSE_COLORS)],
    }


def schedule_total_credits(schedule):
    return sum(int(course.get("credits") or 0) for course in schedule.values())


def find_schedule_conflicts(schedule, candidate):
    candidate_slots = set(candidate.get("slots") or [])
    conflicts = []
    for course_id, scheduled in schedule.items():
        if course_id == candidate.get("course_id"):
            continue
        if candidate_slots & set(scheduled.get("slots") or []):
            conflicts.append(course_id)
    return conflicts


def find_course_code(course_ref, filtered_courses):
    if not course_ref:
        return None
    ref = str(course_ref).strip()
    if ref in filtered_courses:
        return ref

    normalized_ref = ref.lower()
    for code, data in filtered_courses.items():
        meta = data.get("metadata", {})
        if normalized_ref == code.lower() or normalized_ref in str(meta.get("name", "")).lower():
            return code
    return None


def course_label(course_code, course_obj=None, schedule_course=None):
    if schedule_course and schedule_course.get("course_name"):
        return f"{schedule_course.get('course_name')} ({course_code})"
    if course_obj:
        return f"{display_course_name(course_obj.get('metadata', {}).get('name', course_code))} ({course_code})"
    return str(course_code)


def extract_schedule_actions(reply, user_prompt=None):
    """Extract action dicts from JSON objects/arrays embedded in the assistant reply."""
    if not reply:
        return []
    if user_prompt is not None and not is_schedule_edit_request(user_prompt):
        return []

    decoder = json.JSONDecoder()
    actions = []
    seen = set()

    def add_action(action):
        key = json.dumps(action, sort_keys=True)
        if key not in seen:
            seen.add(key)
            actions.append(action)

    for match in re.finditer(r"[\[{]", reply):
        try:
            parsed, _ = decoder.raw_decode(reply[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            if "actions" in parsed and isinstance(parsed["actions"], list):
                for item in parsed["actions"]:
                    if isinstance(item, dict):
                        add_action(item)
            elif "action" in parsed:
                add_action(parsed)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "action" in item:
                    add_action(item)
    return actions


def apply_schedule_actions(actions, schedule, filtered_courses):
    """Apply model actions to schedule state while rejecting overlaps."""
    updated_schedule = dict(schedule)
    results = []

    for action in actions:
        action_type = normalize_action_name(action.get("action"))
        course_ref = course_id_from_action(action)
        course_code = find_course_code(course_ref, filtered_courses)

        if action_type == "add_course":
            if not course_code:
                results.append(f"Could not add {course_ref or 'the requested course'} because it is not in the filtered candidate set.")
                continue
            candidate = build_schedule_course(course_code, filtered_courses[course_code], len(updated_schedule))
            conflicts = find_schedule_conflicts(updated_schedule, candidate)
            if conflicts:
                conflict_labels = [
                    course_label(conflict_code, schedule_course=updated_schedule.get(conflict_code))
                    for conflict_code in conflicts
                ]
                results.append(
                    f"Could not add {course_label(course_code, schedule_course=candidate)} because it overlaps with "
                    + ", ".join(conflict_labels)
                    + "."
                )
                continue
            updated_schedule[course_code] = candidate
            results.append(f"Added {course_label(course_code, schedule_course=candidate)}.")
        elif action_type == "remove_course":
            if not course_code:
                results.append(f"Could not remove {course_ref or 'the requested course'} because it was not recognized.")
                continue
            if course_code in updated_schedule:
                removed_course = updated_schedule[course_code]
                updated_schedule.pop(course_code)
                results.append(f"Removed {course_label(course_code, schedule_course=removed_course)}.")
            else:
                results.append(f"{course_label(course_code, course_obj=filtered_courses.get(course_code))} is not currently in the timetable.")
        elif action_type == "optimize_courses":
            results.append("Optimization action received; add/remove course actions are needed to update the timetable.")

    return updated_schedule, results


def course_summary_for_prompt(schedule):
    if not schedule:
        return "No courses are currently selected."
    rows = []
    for index, (code, course) in enumerate(schedule.items(), start=1):
        rows.append(
            f"{index}. {course.get('course_name')} ({code}) | "
            f"{course.get('credits', 0)} credits | "
            f"capacity {course.get('max_capacity') or 'Not listed'} | "
            f"{course.get('time') or 'Not listed'}"
        )
    return "\n".join(rows)
