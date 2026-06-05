"""Shared helpers for course text normalization and display formatting."""


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value)
    return str(value).strip().lower()


def clean_focus_area(area):
    return normalize_text(str(area).split(" (")[0])


def course_search_blob(code, course_obj):
    meta = course_obj.get("metadata", {})
    chunks = course_obj.get("text_chunks", {})
    parts = [code]
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
