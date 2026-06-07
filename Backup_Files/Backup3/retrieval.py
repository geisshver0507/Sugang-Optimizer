"""Per-turn retrieval ranking for filtered course candidates."""

import re

from course_utils import course_search_blob, normalize_text


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "course", "courses",
    "do", "for", "from", "give", "have", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "please", "recommend", "should", "take", "tell", "that", "the",
    "them", "this", "to", "what", "which", "with", "you",
}

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\d{4}(?:-\d{2}){0,2}\b")

FOLLOWUP_MATCH_WORDS = {
    "match", "matches", "matched", "subject", "subjects", "course", "courses",
    "list", "show", "those", "all",
}


def tokenize_query(text):
    tokens = re.findall(r"[A-Za-z0-9?-?]+", normalize_text(text))
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def extract_course_codes(text):
    """Return course codes mentioned in text, preserving first-seen order."""
    seen = set()
    codes = []
    for code in COURSE_CODE_RE.findall(str(text or "")):
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def requested_match_count(query):
    match = re.search(r"\b(\d{1,2})\b", str(query or ""))
    return int(match.group(1)) if match else None


def is_match_followup(query):
    tokens = set(tokenize_query(query))
    return bool(tokens & FOLLOWUP_MATCH_WORDS) and any(
        word in normalize_text(query)
        for word in ("match", "matches", "matched", "subject", "subjects", "course", "courses")
    )


def score_course_for_prompt(code, course_obj, query):
    meta = course_obj.get("metadata", {})
    blob = course_search_blob(code, course_obj)
    tokens = tokenize_query(query)
    lowered_query = normalize_text(query)
    code_l = normalize_text(code)
    name = normalize_text(meta.get("name"))
    professor = normalize_text(meta.get("professor"))
    keywords = normalize_text(meta.get("keywords", []))

    score = 0
    if code_l and code_l in lowered_query:
        score += 50
    for token in tokens:
        if token in code_l:
            score += 20
        elif token in name or token in professor:
            score += 8
        elif token in keywords:
            score += 5
        elif token in blob:
            score += 1

    workload = normalize_text(meta.get("workload"))
    difficulty = normalize_text(meta.get("difficulty"))
    if any(word in lowered_query for word in ("easy", "light", "manageable", "low workload")):
        if workload == "light" or difficulty == "low":
            score += 5
    if any(word in lowered_query for word in ("hard", "difficult", "challenging", "advanced")):
        if workload == "heavy" or difficulty == "high":
            score += 4
    if any(word in lowered_query for word in ("mileage", "competitive", "bid", "bidding")):
        score += 2
    return score


def fallback_course_rank(item):
    code, course_obj = item
    meta = course_obj.get("metadata", {})
    workload_rank = {"light": 0, "medium": 1, "heavy": 2, "high": 2}
    difficulty_rank = {"low": 0, "medium": 1, "high": 2}
    return (
        workload_rank.get(normalize_text(meta.get("workload")), 1),
        difficulty_rank.get(normalize_text(meta.get("difficulty")), 1),
        int(meta.get("mileage_historical_eta") or 999),
        code,
    )


def select_relevant_courses(filtered_courses, query, limit=10, prior_codes=None):
    if not filtered_courses:
        return {}

    prior_codes = [code for code in (prior_codes or []) if code in filtered_courses]
    explicit_codes = [code for code in extract_course_codes(query) if code in filtered_courses]
    requested_count = requested_match_count(query)

    if explicit_codes:
        return {code: filtered_courses[code] for code in explicit_codes[: max(limit, len(explicit_codes))]}

    if is_match_followup(query):
        if requested_count and requested_count >= len(filtered_courses):
            return dict(filtered_courses)
        if prior_codes:
            return {code: filtered_courses[code] for code in prior_codes[: max(limit, requested_count or len(prior_codes))]}
        if requested_count:
            return dict(list(filtered_courses.items())[:requested_count])

    scored = [
        (score_course_for_prompt(code, data, query), code, data)
        for code, data in filtered_courses.items()
    ]
    scored.sort(key=lambda row: (-row[0], fallback_course_rank((row[1], row[2]))))

    if scored and scored[0][0] > 0:
        selected = [(code, data) for score, code, data in scored if score > 0][:limit]
    else:
        selected = sorted(filtered_courses.items(), key=fallback_course_rank)[:limit]
    return dict(selected)
