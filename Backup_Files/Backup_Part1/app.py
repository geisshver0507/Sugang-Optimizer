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
st.set_page_config(page_title="Yonsei Course Assistant", layout="wide")

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

# ── 4. INTAKE SCREEN ─────────────────────────────────────────────────────────
if not st.session_state.intake_done:
    
    st.markdown("""
        <div style='text-align: center; margin-bottom: 25px;'>
            <h1 style='margin-bottom: 0px; font-size: 3rem;'> Hi! I'm NightHawk AI 🦅</h1>
            <h5 style='color: #cbd5e1; margin-top: 4px; font-weight: 500;'>Yonsei Course Assistant - Customize your targets to extract your optimal course alignment.</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* Your existing multiselect styling */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #2b5c8f !important;
            color: white !important;
        }
        
        /* 🚨 NEW: Override Streamlit's default top/bottom padding 🚨 */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 1rem !important;
            max-width: 90vw;
        }

        /* 🚨 NEW: Hide default Streamlit footer to save space 🚨 */
        footer {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.form("intake_form"):
        col_inputs = st.container()

        # ── LEFT SIDE: All Parameters Compressed ──
        with col_inputs:
            st.subheader("📋 Search Parameters")
            
            # Nested columns to save vertical space
            inner_left, inner_right = st.columns(2)
            
            with inner_left:
                language = st.radio("Language Medium", ["Any", "English", "Korean"], horizontal=True)
                lecture_type = st.radio("Lecture Type", ["Both", "Offline", "Blended"], horizontal=True)
                
            with inner_right:
                st.markdown("**Course Categories**")
                cat_req = st.checkbox("Major Requirement", value=True)
                cat_elec = st.checkbox("Major Elective", value=True)
                cat_basic = st.checkbox("Major Basic", value=True)

            # Target Major Year and Mileage side-by-side
            year_mil_col1, year_mil_col2 = st.columns(2)
            with year_mil_col1:
                major_year = st.selectbox("Target Major Year", ["Any", "2nd", "3rd", "4th"], index=2)
            with year_mil_col2:
                max_credits = st.number_input("Course Credit Taking", min_value=0, max_value=21, value=19, step=1)

            focus_areas = st.multiselect(
                "Areas of Interest (Focus Keywords)",
                [
                    "Theory & Mathematics (Discrete Math, Linear Algebra, Probability)", 
                    "Software Engineering (Object-Oriented Programming, Algorithms)", 
                    "Systems & Performance (Operating Systems, Computer Networks, Computer Architecture)", 
                    "AI & Data (Machine Learning, Computer Vision)", 
                    "Human & Security (Human-Computer Interaction, Computer Security)"
                ],
            )

        # The timetable now lives on the chatbot page and is updated only by chatbot actions.
        if False:
            st.markdown("<h3 style='text-align: center; margin-bottom: 5px;'>Preferred Class Time Slots</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #666; font-size: 14px;'>Click and drag across time slots to map out your ideal schedule window.</p>", unsafe_allow_html=True)

            timetable_html = """
            <div id="timetable-container" style="
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                user-select: none; 
                max-width: 900px; 
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.85); /* Slightly more opaque for readability against the starry sky */
                backdrop-filter: blur(10px);
                border-radius: 12px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                padding: 12px; /* Reduced from 20px */
                overflow-x: auto;
            ">
                <table style="width: 100%; border-collapse: separate; border-spacing: 3px; text-align: center; font-size: 11px;"> <!-- Reduced font size & spacing -->
                    <thead>
                        <tr>
                            <th style="padding: 6px; width: 60px; color: #495057; font-weight: 600;">Time</th>
                            <th style="padding: 6px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 4px;">Mon <span id="count-Mon" style="font-size:9px; display:block; color:#868e96; font-weight:400;">0h</span></th>
                            <th style="padding: 6px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 4px;">Tue <span id="count-Tue" style="font-size:9px; display:block; color:#868e96; font-weight:400;">0h</span></th>
                            <th style="padding: 6px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 4px;">Wed <span id="count-Wed" style="font-size:9px; display:block; color:#868e96; font-weight:400;">0h</span></th>
                            <th style="padding: 6px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 4px;">Thu <span id="count-Thu" style="font-size:9px; display:block; color:#868e96; font-weight:400;">0h</span></th>
                            <th style="padding: 6px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 4px;">Fri <span id="count-Fri" style="font-size:9px; display:block; color:#868e96; font-weight:400;">0h</span></th>
                        </tr>
                    </thead>
                    <tbody id="timetable-body">
                    </tbody>
                </table>
                <input type="hidden" id="timetable-output" name="timetable_output" value="{}">
            </div>

            <script>
            const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
            const hours = ['09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00'];
            const tbody = document.getElementById('timetable-body');
            let isMouseDown = false;
            let isSelecting = true;
            let selectedSlots = {};

            days.forEach(day => selectedSlots[day] = []);

            hours.forEach((hour) => {
                const tr = document.createElement('tr');
                const tdTime = document.createElement('td');
                tdTime.innerText = hour;
                tdTime.style.padding = '4px'; /* Reduced padding */
                tdTime.style.fontWeight = '600';
                tdTime.style.color = '#495057';
                tdTime.style.background = '#f8f9fa';
                tdTime.style.borderRadius = '4px';
                tr.appendChild(tdTime);

                days.forEach(day => {
                    const td = document.createElement('td');
                    td.style.height = '24px'; /* Reduced height from 32px to save ~80px total vertical space */
                    td.style.cursor = 'pointer';
                    td.style.backgroundColor = '#f1f3f5'; 
                    td.style.borderRadius = '4px';
                    td.style.transition = 'all 0.15s ease';
                    td.dataset.day = day;
                    td.dataset.hour = hour;

                    td.addEventListener('mouseenter', () => {
                        if(!td.classList.contains('active')) {
                            td.style.backgroundColor = '#e3fafc';
                        }
                    });
                    td.addEventListener('mouseleave', () => {
                        if(!td.classList.contains('active')) {
                            td.style.backgroundColor = '#f1f3f5';
                        }
                    });

                    td.addEventListener('mousedown', (e) => {
                        isMouseDown = true;
                        isSelecting = !td.classList.contains('active');
                        executeToggle(td, isSelecting);
                        e.preventDefault();
                    });
                    
                    td.addEventListener('mouseover', () => {
                        if (isMouseDown) {
                            executeToggle(td, isSelecting);
                        }
                    });

                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });

            window.addEventListener('mouseup', () => {
                if (isMouseDown) {
                    isMouseDown = false;
                    updateOutput();
                }
            });

            function executeToggle(cell, forceSelect) {
                const day = cell.dataset.day;
                const hour = cell.dataset.hour;
                
                if (forceSelect) {
                    cell.classList.add('active');
                    cell.style.backgroundColor = '#1c3879';
                    cell.style.boxShadow = 'inset 0 0 8px rgba(0,0,0,0.2)';
                    if (!selectedSlots[day].includes(hour)) {
                        selectedSlots[day].push(hour);
                    }
                } else {
                    cell.classList.remove('active');
                    cell.style.backgroundColor = '#f1f3f5';
                    cell.style.boxShadow = 'none';
                    selectedSlots[day] = selectedSlots[day].filter(h => h !== hour);
                }
                document.getElementById(`count-${day}`).innerText = `${selectedSlots[day].length}h`;
            }

            function updateOutput() {
                const output = document.getElementById('timetable-output');
                output.value = JSON.stringify(selectedSlots);
                output.dispatchEvent(new Event('change', { bubbles: true }));
            }
            </script>
            """
            st.components.v1.html(timetable_html, height=410, scrolling=False)

        # ── BOTTOM: Submit Button ──
        # Putting it back in the main form context so it spans the entire width beneath the columns
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
                "focus_areas": focus_areas,
            }
            
            # Execute RAG structural pruning lookups
            filtered = filter_tree_courses(CS_TREE, prefs)

            # Do not silently fall back to the full database; that makes the assistant hallucinate.
            if not filtered:
                st.warning("No courses matched those filters. Please loosen one or more filters to get grounded recommendations.")

            st.session_state.filtered_courses = filtered
            st.session_state.prefs = prefs
            st.session_state.intake_done = True
            st.rerun()

# ── 5. CHAT SCREEN ───────────────────────────────────────────────────────────
else:
    with st.sidebar:
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
        st.write(f"Interests Specified: **{', '.join([a.split(' (')[0] for a in p['focus_areas']]) if p['focus_areas'] else 'All'}**")
        st.metric("Filtered Candidates", len(st.session_state.filtered_courses))
        
        if st.button("Reset Filters & Availability"):
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

    st.title("🎓 Yonsei Course Assistant")

    chat_col, timetable_col = st.columns([1.25, 0.95], gap="large")

    with chat_col:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

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

                display_reply = reply
                if action_results:
                    display_reply += "\n\nSchedule update:\n" + "\n".join(f"- {result}" for result in action_results)
                st.write(display_reply)

            st.session_state.messages.append({"role": "assistant", "content": display_reply})
            st.rerun()

    with timetable_col:
        st.subheader("Weekly Timetable")
        selected_schedule = st.session_state.selected_schedule
        total_credits = schedule_total_credits(selected_schedule)
        st.metric("Total Credits", total_credits)
        render_weekly_timetable(selected_schedule)

        if selected_schedule:
            with st.expander("Selected Courses", expanded=True):
                for code, course in selected_schedule.items():
                    st.write(f"**{code}** - {course.get('course_name')}")
                    st.caption(f"{course.get('credits', 0)} credits | {course.get('time') or 'Time not listed'}")
        else:
            st.info("The timetable is empty. Ask the chatbot to add a course.")

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
            st.session_state.messages.append({"role": "assistant", "content": ranking_reply})
            st.session_state.timetable_confirmed = True
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
