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
        reviews = clip_text(text_chunks.get("student_reviews"), 700)
        alt_reviews = clip_text(text_chunks.get("alternative_professor_reviews"), 500)
        syllabus = clip_text(text_chunks.get("grading_and_syllabus"), 650)
        chunks.append(f"""
[COURSE {code}]
Name: {display_course_name(meta.get('name'))}
Professor: {format_field(meta.get('professor'))}
Credits: {format_field(meta.get('credits'))}
Course type: {format_field(meta.get('course_type'))}
Language: {format_field(meta.get('language_medium'))}
Lecture type: {format_field(meta.get('lecture_type'))}
Raw time: {format_field(meta.get('time'))}
Expanded time: {expand_course_time(meta.get('time'))}
Location: {format_field(meta.get('location'))}
Evaluation type: {format_field(meta.get('evaluation_type'))}
Prerequisites: {format_field(meta.get('prerequisites'))}
Workload: {format_field(meta.get('workload'))}
Difficulty: {format_field(meta.get('difficulty'))}
Mileage ETA: {format_field(meta.get('mileage_historical_eta'))}
Keywords: {format_field(meta.get('keywords'))}
Syllabus/grading evidence: {syllabus}
Student review evidence: {reviews}
Alternative professor review evidence: {alt_reviews}
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
- SCHEDULE CONTROL: The chatbot is the only way the user modifies the timetable. Include machine-readable JSON actions ONLY when the user explicitly asks to add, remove, drop, swap, replace, change, schedule, or optimize timetable courses.
- Do NOT include schedule JSON for comparison, explanation, "which is better", "what is the difference", or general recommendation questions.
- Do not tell the user to "use" JSON actions, do not introduce the JSON, and do not explain the action format. Write a normal concise user-facing sentence, then include the raw action JSON after it.
- For an add request, use exactly this action shape: {{"action": "add_course", "course": {{"course_id": "CODE", "course_name": "English Name", "days": ["Mon"], "start_time": "9:00 AM", "end_time": "10:00 AM"}}}}
- For a remove request, use exactly this action shape: {{"action": "remove_course", "course": {{"course_id": "CODE", "course_name": "English Name"}}}}
- For optimize requests, return an "actions" array containing the add_course and remove_course actions needed to transform the current schedule. Do not return draggable/manual edit instructions.
- Only create schedule actions for courses in the active filtered candidate set. Use exact course codes from the evidence or candidate catalog.
- Never ask the user to manually drag, draw, or create timetable blocks.
- If evidence is missing, say "Not listed in the retrieved data" instead of guessing.
- If the user asks about a course outside the active filters, say it is not in the current filtered candidate set and suggest adjusting filters.
- Use English-only course names in user-facing text. If a database name contains Korean plus an English name in parentheses, use only the English name.
- Avoid generic filler like "matches your interest in computer science" unless the user literally gave no more specific preference. Use the actual distinguishing evidence: workload, difficulty, language, professor/review sentiment, schedule, grading, prerequisites, course type, and keywords.
- When two sections of the same course exist, explain concrete differences between the sections instead of recommending only one section.

Response mode rules:
- For comparison or "difference" questions, DO NOT start with "Here are the recommendations." Use a short comparison format such as:
    CAS3102-01 vs CAS3102-02
    - Main difference: ...
    - CAS3102-01: ...
    - CAS3102-02: ...
    - My pick: ... because ...
- For "which one should I take" questions, recommend one option only after comparing the alternatives. Mention the tradeoff.
- For broad recommendation lists, you may use this format, but make each item specific and evidence-based:
    1. English Course Name (CODE)
        * Fit: [One concise sentence about why it matches user interests].
        * Evidence: [One concise sentence based on syllabus or reviews].
        * Caveat: [One concise sentence about difficulty or workload].
- For ordinary explanations, answer naturally in concise paragraphs or bullets. Do not force Fit/Evidence/Caveat.
- DO NOT repeat the same reasoning sentence across multiple courses.
- DO NOT list fields like "Schedule", "Workload", "Mileage", "Credits", "Prerequisites" as separate labels unless the user asked for a comparison table or detailed breakdown.
- If the user asks for a list, output the list. Add schedule JSON only if they explicitly asked to modify the timetable.


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

CRITICAL OUTPUT RULES:
1. Match the user's intent. Comparison questions get comparisons; recommendation questions get recommendations; schedule-edit questions get schedule actions.
2. NEVER output standalone words, category headers, or preambles (like "Fit", "Evidence", "Schedule", "Workload", etc.) at the top of your message.
3. If schedule actions are needed, put the raw JSON at the very end.
"""

def build_priority_ranking_system_prompt(prefs, selected_schedule, filtered_courses):
    selected_evidence = {}
    for code, course in selected_schedule.items():
        if code in filtered_courses:
            selected_evidence[code] = filtered_courses[code]
        else:
            selected_evidence[code] = {
                "metadata": {
                    "name": course.get("course_name", code),
                    "credits": course.get("credits"),
                    "time": course.get("time"),
                    "professor": course.get("professor"),
                },
                "text_chunks": {},
            }

    return f"""
You are NightHawk AI, a Yonsei CS course recommendation assistant for Part 1 of this project.
The user has clicked Confirm Timetable. Generate an initial recommended priority ranking for the confirmed courses.

Ranking rules:
- Rank all and only the selected course codes from 1 to N, where 1 is highest priority.
- Make ranks unique.
- The ranking is only an editable recommendation, not a final decision.
- Weigh the user's conversation context most heavily, especially interests, academic goals, workload tolerance, assessment preferences, and stated concerns.
- Also use onboarding filters, course reviews and ratings when available, syllabus/evaluation evidence, workload, difficulty, keywords, course type, and mileage competitiveness.
- Never invent course facts, review sentiment, ratings, professors, grading policies, prerequisites, or outcomes.
- If evidence is missing, say that the evidence is limited instead of guessing.
- Keep each reason concise and grounded in the supplied evidence or conversation context.

OUTPUT FORMAT INSTRUCTIONS:
Your output must consist of two parts. First, present a clean, beautifully formatted user-facing Markdown summary. Second, append the raw machine-readable JSON object at the very end. Do not wrap the JSON in markdown code blocks.

Use this exact layout for your response:

### 📋 Recommended Priority Ranking
Here is a suggested priority ranking for your confirmed courses based on your academic goals and course requirements:

1. **English Course Name (COURSE_CODE)**
   - **Fit:** A concise sentence explaining why this is priority #1.
   - **Prerequisites/Workload:** A concise sentence regarding workload or prerequisites.
   - **Mileage Strategy:** A concise sentence regarding its mileage competitiveness signal.

[Repeat for all N courses in order...]

{{"ranking":[{{"course_id":"COURSE_CODE","rank":1,"reasons":["Reason 1","Reason 2"]}}]}}

Student profile from onboarding filters:
- Language: {prefs.get('language')}
...
"""
