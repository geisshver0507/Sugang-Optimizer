from html import escape

import streamlit as st
from groq import Groq

from course_repository import load_tree_database
from filters import filter_tree_courses
from guardrails import recent_conversation, validate_grounding
from prompts import build_system_prompt
from retrieval import select_relevant_courses
from schedule_utils import (
    apply_schedule_actions,
    course_summary_for_prompt,
    extract_schedule_actions,
    schedule_total_credits,
)

# import base64

# ── 0. Page config (must be first Streamlit call) ───────────────────────────
st.set_page_config(page_title="NightHawk AI - Yonsei Course Assistant", layout="wide")

# def set_background(image_file):
#     """Encodes a local image and injects it as the Streamlit app background."""
#     with open(image_file, "rb") as file:
#         encoded_string = base64.b64encode(file.read()).decode()
    
#     css = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/jpeg;base64,{encoded_string}");
#         background-size: cover;
#         background-position: center;
#         background-attachment: fixed;
#     }}
#     /* Optional: Adds a slight dark overlay so your text remains readable */
#     .stApp > header {{
#         background-color: transparent;
#     }}
#     </style>
#     """
#     st.markdown(css, unsafe_allow_html=True)

# Call this right after your Session State init (Section 3)
# if os.path.exists("Night_Sky.jpg"):
#     set_background("Night_Sky.jpg")
# else:
#     st.warning("Background image 'Night_Sky.jpg' not found in directory.")

# ── 1. Groq client ──────────────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---- 2. LLM bridge ----------------------------------------------------
def call_llm(system, messages):
    groq_messages = [{"role": "system", "content": system}]
    for m in recent_conversation(messages):
        groq_messages.append({"role": m["role"], "content": m["content"]})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=groq_messages,
        max_tokens=800,
        temperature=0.1,
    )
    return response.choices[0].message.content

# ── 3. Session state init ────────────────────────────────────────────────────
for key, default in [
    ("intake_done", False),
    ("messages", []),
    ("filtered_courses", {}),
    ("prefs", {}),
    ("retrieved_course_codes", []),
    ("selected_schedule", {}),
    ("schedule_action_log", []),
    ("timetable_confirmed", False),
    ("priority_rankings", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Load static tree database representation asset
try:
    CS_TREE = load_tree_database()
except FileNotFoundError as exc:
    st.error(str(exc))
    CS_TREE = {}


def render_weekly_timetable(schedule):
    """Render a read-only weekly timetable controlled by chatbot actions."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    row_height = 44
    start_period = 1
    end_period = 13
    total_rows = end_period - start_period + 1
    grid_height = total_rows * row_height
    time_labels = [
        "9 AM", "10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM",
        "4 PM", "5 PM", "6 PM", "7 PM", "8 PM", "9 PM",
    ]

    time_cells = "".join(
        f"<div class='time-cell' style='top:{idx * row_height}px'>{label}</div>"
        for idx, label in enumerate(time_labels)
    )
    grid_lines = "".join(
        f"<div class='grid-line' style='top:{idx * row_height}px'></div>"
        for idx in range(total_rows + 1)
    )

    blocks = []
    for course in schedule.values():
        meetings = course.get("meetings") or []
        for meeting in meetings:
            day = meeting.get("day")
            if day not in days:
                continue
            top = (int(meeting.get("start_period", 1)) - start_period) * row_height + 2
            span = int(meeting.get("end_period", 1)) - int(meeting.get("start_period", 1)) + 1
            height = max(28, span * row_height - 4)
            left = days.index(day) * 20
            name = escape(str(course.get("course_name", "Course")))
            code = escape(str(course.get("course_id", "")))
            time_text = escape(f"{meeting.get('start_time')} - {meeting.get('end_time')}")
            color = escape(str(course.get("color", "#2563eb")))
            blocks.append(
                f"""
                <div class='course-block' style='top:{top}px; left:{left}%; width:calc(20% - 8px); height:{height}px; background:{color};'>
                    <div class='course-code'>{code}</div>
                    <div class='course-name'>{name}</div>
                    <div class='course-time'>{time_text}</div>
                </div>
                """
            )

    empty_state = ""
    if not schedule:
        empty_state = "<div class='empty-state'>Ask the chatbot to add courses.</div>"

    timetable_html = f"""
    <style>
    .schedule-shell {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #ffffff;
        border: 1px solid #d8dee9;
        border-radius: 8px;
        overflow: hidden;
    }}
    .schedule-head {{
        display: grid;
        grid-template-columns: 58px repeat(5, 1fr);
        background: #0f172a;
        color: white;
        font-size: 12px;
        font-weight: 700;
        text-align: center;
    }}
    .schedule-head div {{
        padding: 10px 4px;
        border-left: 1px solid rgba(255,255,255,0.12);
    }}
    .schedule-body {{
        display: grid;
        grid-template-columns: 58px 1fr;
        height: {grid_height}px;
        position: relative;
        background: #f8fafc;
    }}
    .time-rail {{
        position: relative;
        height: {grid_height}px;
        background: #f1f5f9;
        border-right: 1px solid #cbd5e1;
    }}
    .time-cell {{
        position: absolute;
        left: 0;
        width: 100%;
        height: {row_height}px;
        padding-top: 4px;
        color: #475569;
        font-size: 11px;
        text-align: center;
        border-bottom: 1px solid #dbe3ef;
        box-sizing: border-box;
    }}
    .days-grid {{
        position: relative;
        height: {grid_height}px;
        background:
            linear-gradient(to right, transparent 19.8%, #dbe3ef 20%, transparent 20.2%),
            linear-gradient(to right, transparent 39.8%, #dbe3ef 40%, transparent 40.2%),
            linear-gradient(to right, transparent 59.8%, #dbe3ef 60%, transparent 60.2%),
            linear-gradient(to right, transparent 79.8%, #dbe3ef 80%, transparent 80.2%);
    }}
    .grid-line {{
        position: absolute;
        left: 0;
        right: 0;
        border-top: 1px solid #e2e8f0;
    }}
    .course-block {{
        position: absolute;
        border-radius: 6px;
        color: #fff;
        padding: 6px 7px;
        box-sizing: border-box;
        overflow: hidden;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.18);
    }}
    .course-code {{
        font-size: 11px;
        font-weight: 800;
        line-height: 1.1;
    }}
    .course-name {{
        margin-top: 3px;
        font-size: 11px;
        line-height: 1.15;
        max-height: 38px;
        overflow: hidden;
    }}
    .course-time {{
        margin-top: 3px;
        font-size: 10px;
        opacity: 0.88;
    }}
    .empty-state {{
        position: absolute;
        inset: 45% 8px auto 8px;
        text-align: center;
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
    }}
    </style>
    <div class='schedule-shell'>
        <div class='schedule-head'>
            <div>Time</div>
            <div>Mon</div><div>Tue</div><div>Wed</div><div>Thu</div><div>Fri</div>
        </div>
        <div class='schedule-body'>
            <div class='time-rail'>{time_cells}</div>
            <div class='days-grid'>{grid_lines}{''.join(blocks)}{empty_state}</div>
        </div>
    </div>
    """
    st.components.v1.html(timetable_html, height=grid_height + 48, scrolling=False)


def strip_schedule_action_text(reply):
    """Hide machine-readable schedule actions from the visible chat transcript."""
    if not reply:
        return ""

    visible_lines = []
    skipping_actions = False
    action_triggers = (
        "json action",
        "json actions",
        "following action",
        "following actions",
        "schedule_actions",
        '"action"',
        "'action'",
    )
    resume_prefixes = (
        "grounding note:",
        "grounding check:",
        "schedule update:",
    )

    for line in str(reply).splitlines():
        stripped = line.strip()
        lowered = stripped.lower()

        if skipping_actions:
            if any(lowered.startswith(prefix) for prefix in resume_prefixes):
                skipping_actions = False
                visible_lines.append(line)
            continue

        if any(trigger in lowered for trigger in action_triggers) or stripped in ("[", "]", "{", "}"):
            skipping_actions = True
            continue

        visible_lines.append(line)

    cleaned = "\n".join(visible_lines).strip()
    if not cleaned:
        return "I updated the timetable based on your request."
    return cleaned

# ── 4. INTAKE SCREEN ─────────────────────────────────────────────────────────
if not st.session_state.intake_done:
    
    st.markdown("""
        <style>
        /* Override Streamlit's default top/bottom padding */
        .block-container {
            padding-top: 5rem !important;
            padding-bottom: 1rem !important;
            max-width: 80vw; /* Keep it tight so the text and form stay close together */
        }

        /* Hide default Streamlit footer to save space */
        footer {
            visibility: hidden;
        }
        
        /* 🚨 NEW: Style the Form to look like a floating, glowing glass card 🚨 */
        [data-testid="stForm"] {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
        }
        
        /* Tweak multiselect if you decide to bring it back later */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #2b5c8f !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── SPLIT LAYOUT: Centered Title Left, Single-Column Form Right ──
    col_title, col_filters = st.columns([1.1, 1], gap="large")

    with col_title:
        # Using Flexbox to vertically and horizontally center the massive text
        st.markdown("""
            <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; height: 100%; min-height: 500px;'>
                <h1 style='margin-bottom: 0px; font-size: 5.5rem; font-weight: 800; line-height: 1.1;'>Hi!    I'm NightHawk AI🦅</h1>
                <h5 style='color: #94a3b8; margin-top: 15px; font-weight: 400; font-size: 1.3rem; line-height: 1.6; max-width: 85%;'>
                    Yonsei Course Assistant &mdash; Customize your targets to extract your optimal course alignment.
                </h5>
            </div>
        """, unsafe_allow_html=True)

    with col_filters:
        # Push the form down slightly to align with the middle of the text
        st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
        
        with st.form("intake_form"):
            st.markdown("<h3 style='text-align: center; margin-bottom: 25px;'>📋 Search Parameters</h3>", unsafe_allow_html=True)
            
            left, right = st.columns(2)

            with left: language = st.radio("Language Medium", ["Any", "English", "Korean"], horizontal=True)
            with right: lecture_type = st.radio("Lecture Type", ["Both", "Offline", "Blended"], horizontal=True)
            
            st.markdown("<div style='margin-top: 15px; margin-bottom: 5px;'><b>Course Categories</b></div>", unsafe_allow_html=True)
            # Putting checkboxes in a neat horizontal row to save vertical space while staying in one main column
            cat_col1, cat_col2, cat_col3 = st.columns(3)
            with cat_col1: cat_req = st.checkbox("Major Requirement", value=True)
            with cat_col2: cat_elec = st.checkbox("Major Elective", value=True)
            with cat_col3: cat_basic = st.checkbox("Major Basic", value=True)

            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            major_year = st.selectbox("Target Major Year", ["Any", "2nd", "3rd", "4th"], index=2)
            max_credits = st.number_input("Course Credit Taking", min_value=0, max_value=21, value=19, step=1)

            st.markdown("<br>", unsafe_allow_html=True) # Spacer before the button
            
            # ── BOTTOM: Submit Button ──
            submitted = st.form_submit_button("Generate Strategic Schedule Matching Preferences →", use_container_width=True)

            if submitted:
                prefs = {
                    "language": language,
                    "lecture_type": lecture_type,
                    "major_year": major_year,
                    "major_years": [] if major_year == "Any" else [major_year],
                    "cat_req": cat_req,
                    "cat_elec": cat_elec,
                    "cat_basic": cat_basic,
                    "max_credits": max_credits,
                    "focus_areas": [],  # Passed as empty so backend logic doesn't break
                }
                
                # Execute RAG structural pruning lookups
                filtered = filter_tree_courses(CS_TREE, prefs)

                # Do not silently fall back to the full database; that makes the assistant hallucinate.
                if not filtered:
                    st.warning("No courses matched those filters. Please loosen one or more filters to get grounded recommendations.")
                else:
                    st.session_state.filtered_courses = filtered
                    st.session_state.prefs = prefs
                    st.session_state.intake_done = True
                    st.rerun()

# ── 5. CHAT SCREEN ───────────────────────────────────────────────────────────
else:
    # ==========================================
    # 🌟 CSS: WIDEN THE NATIVE SIDEBAR
    # ==========================================
    st.markdown("""
        <style>
        /* Force the native sidebar to take up ~45% of the screen width */
        [data-testid="stSidebar"] {
            min-width: 45vw !important;
            max-width: 50vw !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # 🌟 LEFT SIDEBAR: DUAL-COLUMN LAYOUT
    # ==========================================
    with st.sidebar:
        # Split the wide sidebar into two internal columns
        # Ratio [1, 1.4] gives the timetable slightly more breathing room
        filter_col, time_col = st.columns([1, 1.4], gap="medium")

        with filter_col:
            p = st.session_state.prefs
            year_options = ["1st", "2nd", "3rd", "4th"]
            language_options = ["Any", "English", "Korean"]

            current_years = p.get("major_years")
            if current_years is None:
                current_years = [] if p.get("major_year") == "Any" else [p.get("major_year", "3rd")]
            current_years = [year for year in current_years if year in year_options]
            default_years = current_years if current_years else year_options

            current_language = p.get("language", "Any")
            if current_language not in language_options:
                current_language = "Any"

            st.markdown("### Dynamic Filters")
            selected_years = st.multiselect(
                "Major Years",
                year_options,
                default=default_years,
                key="chat_major_years",
            )
            selected_language = st.radio(
                "Language Medium",
                language_options,
                index=language_options.index(current_language),
                key="chat_language",
            )

            stored_years = current_years if current_years else year_options
            filters_changed = (
                selected_language != p.get("language", "Any")
                or selected_years != stored_years
            )

            if filters_changed:
                updated_p = dict(p)
                updated_p["language"] = selected_language
                updated_p["major_years"] = selected_years
                if not selected_years or set(selected_years) == set(year_options):
                    updated_p["major_year"] = "Any"
                elif len(selected_years) == 1:
                    updated_p["major_year"] = selected_years[0]
                else:
                    updated_p["major_year"] = ", ".join(selected_years)

                updated_filtered = filter_tree_courses(CS_TREE, updated_p)
                st.session_state.prefs = updated_p
                st.session_state.filtered_courses = updated_filtered
                st.session_state.retrieved_course_codes = []
                p = updated_p

                if not updated_filtered:
                    st.warning("No courses match the current sidebar filters.")

            st.divider()
            st.markdown("### Active Constraints")
            display_years = p.get("major_years") or year_options
            st.write(f"Language: **{p['language']}**")
            st.write(f"Format: **{p['lecture_type']}**")
            st.write(f"Target Years: **{', '.join(display_years)}**")
            st.write(f"Credits: **{p['max_credits']} pts**")
            st.metric("Filtered Candidates", len(st.session_state.filtered_courses))
            selected_schedule = st.session_state.selected_schedule
            total_credits = schedule_total_credits(selected_schedule)
            st.metric("Total Credits", total_credits)
            
            if st.button("Reset Filters & Availability", use_container_width=True):
                st.session_state.intake_done = False
                st.session_state.messages = []
                st.session_state.retrieved_course_codes = []
                st.session_state.selected_schedule = {}
                st.session_state.schedule_action_log = []
                st.session_state.timetable_confirmed = False
                st.session_state.priority_rankings = {}
                st.session_state.pop("chat_major_years", None)
                st.session_state.pop("chat_language", None)
                st.rerun()

            if st.session_state.timetable_confirmed and selected_schedule:
                st.markdown("### Priority Ranking")
                course_count = len(selected_schedule)
                with st.form("priority_ranking_form"):
                    ranking_values = {}
                    for code, course in selected_schedule.items():
                        label = f"{code} - {course.get('course_name')}"
                        ranking_values[code] = st.selectbox(
                            label,
                            list(range(1, course_count + 1)),
                            key=f"priority_{code}",
                        )
                    submitted_rankings = st.form_submit_button("Save Priority Ranking", use_container_width=True)
                    if submitted_rankings:
                        if len(set(ranking_values.values())) != course_count:
                            st.warning("Each course needs a unique priority number.")
                        else:
                            st.session_state.priority_rankings = ranking_values
                            st.success("Priority ranking saved.")

        with time_col:
            st.markdown("### 📅 Weekly Timetable")
            render_weekly_timetable(selected_schedule)

            if selected_schedule:
                with st.expander("Selected Courses", expanded=True):
                    for code, course in selected_schedule.items():
                        st.write(f"**{code}** - {course.get('course_name')}")
                        st.caption(f"{course.get('credits', 0)} credits | {course.get('time') or 'Time not listed'}")
            else:
                st.info("The timetable is empty. Ask the assistant to add a course.")

            if st.session_state.schedule_action_log:
                with st.expander("Schedule Action Log"):
                    for item in st.session_state.schedule_action_log[-8:]:
                        st.write(item)

            if st.button("Confirm Timetable", disabled=not selected_schedule, use_container_width=True):
                final_list = course_summary_for_prompt(selected_schedule)
                confirmation_prompt = (
                    "The user has confirmed this timetable. Ask the user to rank the selected courses "
                    "from 1 to N, where 1 is highest priority and N is lowest priority. "
                    "Here is the final selected course list:\n"
                    f"{final_list}"
                )
                llm_messages = st.session_state.messages + [{"role": "user", "content": confirmation_prompt}]
                ranking_reply = call_llm(
                    "You are NightHawk AI. The timetable is confirmed. Ask for priority rankings only; do not add or remove courses.",
                    llm_messages,
                )
                st.session_state.messages.append({"role": "user", "content": "Confirmed timetable."})
                st.session_state.messages.append({"role": "assistant", "content": strip_schedule_action_text(ranking_reply)})
                st.session_state.timetable_confirmed = True
                st.rerun()

                submitted_rankings = st.form_submit_button("Save Priority Ranking", use_container_width=True)
                if submitted_rankings:
                    if len(set(ranking_values.values())) != course_count:
                        st.warning("Each course needs a unique priority number.")
                    else:
                        st.session_state.priority_rankings = ranking_values
                        st.success("Priority ranking saved.")

    # ==========================================
    # 🌟 MAIN AREA: CHAT INTERFACE
    # ==========================================
    st.title("NightHawk AI - Yonsei Course Assistant 🎓")
    
    with st.container(height=700, border=False):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                content = msg["content"]
                if msg["role"] == "assistant":
                    content = strip_schedule_action_text(content)
                st.write(content)

        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.write(
                    f"Hello! I have matched **{len(st.session_state.filtered_courses)} system courses** filtering exactly along your desired categories and keywords. "
                    "Tell me which courses you want to add, remove, or optimize, and I will update the timetable automatically."
                )

    if prompt := st.chat_input("Ask to add, remove, optimize, or compare courses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing targeted data chunks..."):
                selected_courses = select_relevant_courses(st.session_state.filtered_courses, prompt)
                st.session_state.retrieved_course_codes = list(selected_courses.keys())
                system_prompt = build_system_prompt(
                    p,
                    selected_courses,
                    st.session_state.filtered_courses,
                    st.session_state.selected_schedule,
                )
                reply = call_llm(system_prompt, st.session_state.messages)
                reply = validate_grounding(reply, selected_courses, st.session_state.filtered_courses)

                actions = extract_schedule_actions(reply)
                action_results = []
                if actions:
                    updated_schedule, action_results = apply_schedule_actions(
                        actions,
                        st.session_state.selected_schedule,
                        st.session_state.filtered_courses,
                    )
                    st.session_state.selected_schedule = updated_schedule
                    st.session_state.schedule_action_log.extend(action_results)
                    st.session_state.timetable_confirmed = False
                    st.session_state.priority_rankings = {}

            if st.session_state.retrieved_course_codes:
                st.caption("Evidence used: " + ", ".join(st.session_state.retrieved_course_codes))

            display_reply = strip_schedule_action_text(reply)
            if action_results:
                display_reply += "\n\nSchedule update:\n" + "\n".join(f"- {result}" for result in action_results)
            st.write(display_reply)

        st.session_state.messages.append({"role": "assistant", "content": display_reply})
        st.rerun()
