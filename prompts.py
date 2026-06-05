"""Prompt construction for the grounded course assistant."""

from course_utils import clean_focus_area, clip_text, display_course_name, expand_course_time, format_field, extract_time_slots

def display_major_years(prefs):
    selected_years = prefs.get("major_years")
    if selected_years:
        return ", ".join(selected_years)
    return prefs.get("major_year", "Any")


def build_candidate_catalog(filtered_courses):
    if not filtered_courses:
        return "No courses matched the active filters."
        
    # Step 1: Pre-calculate the time slot sets for every filtered course
    course_slots = {}
    for code, data in filtered_courses.items():
        course_slots[code] = extract_time_slots(data.get("metadata", {}).get("time"))

    # Step 2: Build the rows with explicit conflict warnings
    rows = []
    for code, data in filtered_courses.items():
        meta = data.get("metadata", {})
        slots_a = course_slots[code]

        # Find overlapping courses
        conflicts = []
        for other_code, slots_b in course_slots.items():
            # If the sets intersect, there is a time conflict
            if code != other_code and (slots_a & slots_b):
                conflicts.append(other_code)

        conflict_str = f" | CONFLICTS WITH: {', '.join(conflicts)}" if conflicts else " | CONFLICTS WITH: None"

        rows.append(
            f"{display_course_name(meta.get('name'))} ({code}) | {format_field(meta.get('professor'))} | "
            f"{format_field(meta.get('credits'))} credits | raw time: {format_field(meta.get('time'))} | "
            f"expanded time: {expand_course_time(meta.get('time'))}{conflict_str}"
        )
    return "\n".join(rows)


def format_course_context(selected_courses):
    if not selected_courses:
        return "No retrieved course evidence is available for this turn."

    chunks = []
    for code, data in selected_courses.items():
        meta = data.get("metadata", {})
        text_chunks = data.get("text_chunks", {})
        raw_time = format_field(meta.get("time"))
        expanded_time = expand_course_time(meta.get("time"))
        chunks.append(f"""
[COURSE {code}]
Display name: {display_course_name(meta.get('name'))}
Professor: {format_field(meta.get('professor'))}
Language: {format_field(meta.get('language_medium'))}
Lecture type: {format_field(meta.get('lecture_type'))}
Credits: {format_field(meta.get('credits'))}
Raw database time/location: {raw_time} / {format_field(meta.get('location'))}
Expanded clock time: {expanded_time}
Course type: {format_field(meta.get('course_type'))}
Evaluation: {format_field(meta.get('evaluation_type'))}
Workload/difficulty: {format_field(meta.get('workload'))} / {format_field(meta.get('difficulty'))}
Prerequisites: {format_field(meta.get('prerequisites'))}
Historical mileage ETA: {format_field(meta.get('mileage_historical_eta'))}
Keywords: {format_field(meta.get('keywords'))}
Grading and syllabus evidence: {clip_text(text_chunks.get('grading_and_syllabus'))}
Student review evidence: {clip_text(text_chunks.get('student_reviews'))}
Alternative professor review evidence: {clip_text(text_chunks.get('alternative_professor_reviews'))}
[/COURSE]
""")
    return "\n".join(chunks)


def format_current_schedule(selected_schedule):
    if not selected_schedule:
        return "No courses are currently selected in the timetable."
    rows = []
    for code, course in selected_schedule.items():
        rows.append(
            f"{display_course_name(course.get('course_name'))} ({code}) | "
            f"{format_field(course.get('credits'))} credits | raw time: {format_field(course.get('time'))}"
        )
    return "\n".join(rows)


def build_system_prompt(prefs, selected_courses, filtered_courses, selected_schedule=None):
    return f"""
You are NightHawk AI, a Yonsei CS course recommendation assistant for Part 1 of this project.
Your job is to recommend and filter subjects from the supplied database. Do not perform final schedule optimization or exact bidding allocation; say that belongs to Part 2 if asked.

Grounding rules:
- This is a closed-book task. Use only the COURSE EVIDENCE block for course-specific facts.
- The CANDIDATE CATALOG is only an index of courses that passed the user's filters. Do not infer extra facts from a catalog row.
- Never invent professors, prerequisites, schedules, reviews, grading policies, locations, mileage cutoffs, or enrollment certainty.
- Use Expanded clock time when explaining schedules. Do not convert period numbers yourself.
- Treat historical mileage ETA as a rough competitiveness signal, not a guaranteed winning bid.
- OVERLAP RULE: The CANDIDATE CATALOG explicitly lists time conflicts. You MUST NOT recommend a combination of courses that are listed as conflicting with each other.
- SCHEDULE CONTROL: The chatbot is the only way the user modifies the timetable. When the user asks to add, remove, drop, change, or optimize timetable courses, include machine-readable JSON actions in your reply for the frontend to consume.
- Do not tell the user to "use" JSON actions, do not introduce the JSON, and do not explain the action format. Write a normal concise user-facing sentence, then include the raw action JSON after it.
- For an add request, use exactly this action shape: {{"action": "add_course", "course": {{"course_id": "CODE", "course_name": "English Name", "days": ["Mon"], "start_time": "9:00 AM", "end_time": "10:00 AM"}}}}
- For a remove request, use exactly this action shape: {{"action": "remove_course", "course": {{"course_id": "CODE", "course_name": "English Name"}}}}
- For optimize requests, return an "actions" array containing the add_course and remove_course actions needed to transform the current schedule. Do not return draggable/manual edit instructions.
- Only create schedule actions for courses in the active filtered candidate set. Use exact course codes from the evidence or candidate catalog.
- Never ask the user to manually drag, draw, or create timetable blocks.
- If evidence is missing, say "Not listed in the retrieved data" instead of guessing.
- If the user asks about a course outside the active filters, say it is not in the current filtered candidate set and suggest adjusting filters.
- Use English-only course names in user-facing text. If a database name contains Korean plus an English name in parentheses, use only the English name.
- When recommending courses, name each course as "English Course Name (CODE)" and explain Fit, Evidence, and Caveat.
- Format answers with short paragraphs or bullets. Put Fit, Evidence, and Caveat on separate lines instead of one crowded paragraph.
- Keep answers concise and course-grounded. Prefer 3 to 6 recommendations unless the user asks for a full list.

Student profile from filters:
- Language: {prefs.get('language')}
- Lecture type: {prefs.get('lecture_type')}
- Major years: {display_major_years(prefs)}
- Credit window: {prefs.get('min_credits')} to {prefs.get('max_credits')}
- Focus areas: {', '.join(clean_focus_area(a) for a in prefs.get('focus_areas', [])) or 'All'}
- Available mileage points: {prefs.get('mileage')}

CANDIDATE CATALOG:
{build_candidate_catalog(filtered_courses)}

CURRENT SELECTED TIMETABLE:
{format_current_schedule(selected_schedule or {})}

COURSE EVIDENCE RETRIEVED FOR THIS TURN:
{format_course_context(selected_courses)}
"""

