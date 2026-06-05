"""Deterministic preference filters for the course tree."""

from course_utils import clean_focus_area, course_search_blob, normalize_text


YEAR_MAP = {"1st": "major_year_1", "2nd": "major_year_2", "3rd": "major_year_3", "4th": "major_year_4"}

FOCUS_KEYWORD_MAP = {
    "theory & mathematics": {
        "math", "discrete", "logic", "proofs", "linear algebra", "probability", "statistics",
        "calculus", "graphs", "random variables", "eigenvalues", "matrices",
    },
    "software engineering": {
        "software engineering", "programming", "oop", "object-oriented", "algorithms", "data structures",
        "c++", "java", "python", "classes", "sorting", "hashing", "system design",
    },
    "systems & performance": {
        "operating systems", "computer networks", "computer architecture", "systems", "performance",
        "gpu", "cuda", "parallel", "distributed", "runtime", "optimization",
    },
    "ai & data": {
        "ai", "machine learning", "deep learning", "computer vision", "data", "nlp", "llm",
        "large language models", "reinforcement learning", "visualization", "neural", "transformer",
    },
    "human & security": {
        "human", "human-ai interaction", "hci", "security", "privacy", "cryptography",
        "user modeling", "authentication", "differential privacy",
    },
}


def language_matches(selected_language, course_language):
    if selected_language == "Any":
        return True
    selected = normalize_text(selected_language)
    actual = normalize_text(course_language)
    if selected == "korean":
        return "korean" in actual
    if selected == "english":
        return "english" in actual
    return selected == actual


def lecture_type_matches(selected_type, course_type):
    if selected_type == "Both":
        return True
    selected = normalize_text(selected_type)
    actual = normalize_text(course_type)
    blended_markers = ("blended", "on-off", "video", "pre-recorded", "flipped", "online")
    if selected == "blended":
        return any(marker in actual for marker in blended_markers)
    if selected == "offline":
        is_offline = "in-person" in actual or "offline" in actual
        is_blended = any(marker in actual for marker in blended_markers)
        return is_offline and not is_blended
    return selected in actual


def focus_matches(focus_areas, course_obj):
    if not focus_areas:
        return True

    blob = course_search_blob("", course_obj)
    course_keywords = set(
        normalize_text(keyword)
        for keyword in course_obj.get("metadata", {}).get("keywords", [])
    )

    target_terms = set()
    for area in focus_areas:
        label = clean_focus_area(area)
        target_terms.add(label)
        target_terms.update(FOCUS_KEYWORD_MAP.get(label, set()))

    for term in target_terms:
        if not term:
            continue
        if term in blob or any(term in keyword or keyword in term for keyword in course_keywords):
            return True
    return False


def selected_year_keys(tree_db, prefs):
    selected_years = prefs.get("major_years")
    if selected_years is None:
        single_year = prefs.get("major_year", "Any")
        selected_years = [] if single_year == "Any" else [single_year]

    selected_years = [year for year in selected_years if year and year != "Any"]
    if not selected_years:
        return list(tree_db.keys())

    return [YEAR_MAP[year] for year in selected_years if year in YEAR_MAP]


def selected_categories(prefs):
    categories = []
    if prefs["cat_req"]:
        categories.append("major_requirement")
    if prefs["cat_basic"]:
        categories.append("major_basic")
    if prefs["cat_elec"]:
        categories.append("major_elective")
    if not categories:
        categories = ["major_requirement", "major_basic", "major_elective", "general_elective"]
    return categories


def filter_tree_courses(tree_db, prefs):
    years_to_scan = selected_year_keys(tree_db, prefs)
    categories_to_scan = selected_categories(prefs)

    matched_results = {}
    for y_key in years_to_scan:
        if y_key not in tree_db:
            continue
        for cat_key in categories_to_scan:
            if cat_key not in tree_db[y_key]:
                continue
            for code, course_obj in tree_db[y_key][cat_key].items():
                meta = course_obj.get("metadata", {})
                if not language_matches(prefs["language"], meta.get("language_medium")):
                    continue
                if not lecture_type_matches(prefs["lecture_type"], meta.get("lecture_type")):
                    continue
                if not (prefs["min_credits"] <= meta.get("credits", 0) <= prefs["max_credits"]):
                    continue
                if not focus_matches(prefs["focus_areas"], course_obj):
                    continue
                matched_results[code] = course_obj
    return matched_results
