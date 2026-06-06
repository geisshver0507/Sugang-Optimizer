"""Per-turn retrieval ranking for filtered course candidates."""

import re

from course_utils import course_search_blob, normalize_text


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "course", "courses",
    "do", "for", "from", "give", "have", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "please", "recommend", "should", "take", "tell", "that", "the",
    "them", "this", "to", "what", "which", "with", "you",
}


def tokenize_query(text):
    tokens = re.findall(r"[A-Za-z0-9?-?]+", normalize_text(text))
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


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


def select_relevant_courses(filtered_courses, query, limit=10):
    if not filtered_courses:
        return {}

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
