import streamlit as st
from groq import Groq
from database import CS_COURSES  # make sure this export exists
import json

# ── 0. Page config (must be first Streamlit call) ───────────────────────────
st.set_page_config(page_title="Yonsei Course Assistant", layout="wide")

# ── 1. Groq client ──────────────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── 2. Helpers ───────────────────────────────────────────────────────────────
def slim_course(c):
    return {k: v for k, v in c.items()
            if k not in ["keywords", "location", "time", "review from other courses"]}

def filter_courses(courses, prefs):
    filtered = {}
    for code, c in courses.items():
        if prefs["difficulty"] != "Any" and c["difficulty"] != prefs["difficulty"]:
            continue
        if prefs["workload"] != "Any" and c["workload"] != prefs["workload"]:
            continue
        if prefs["credits"] != "Any" and c["credits"] != int(prefs["credits"]):
            continue
        if prefs["language"] != "Any" and c["language medium"] != prefs["language"]:
            continue
        if prefs["major_req"] and not c["major requirement"]:
            continue
        filtered[code] = c
    return filtered

def call_llm(system, messages):
    # Groq expects messages without the system key inline —
    # pass system as the first message with role "system"
    groq_messages = [{"role": "system", "content": system}]
    for m in messages:
        groq_messages.append({"role": m["role"], "content": m["content"]})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # or "mixtral-8x7b-32768"
        messages=groq_messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content

# ── 3. Session state init ────────────────────────────────────────────────────
for key, default in [
    ("intake_done", False),
    ("messages", []),
    ("filtered_courses", {}),
    ("prefs", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── 4. INTAKE SCREEN ─────────────────────────────────────────────────────────
if not st.session_state.intake_done:
    st.title("🎓 Yonsei Course Assistant")
    st.markdown("Answer a few questions and I'll narrow down the right courses for you.")

    with st.form("intake_form"):
        col1, col2 = st.columns(2)

        with col1:
            difficulty = st.radio("Preferred difficulty", ["Any", "Easy", "Medium", "Hard"], horizontal=True)
            workload   = st.radio("Preferred workload",   ["Any", "Low", "Medium", "High"],  horizontal=True)
            credits    = st.radio("Credits",              ["Any", "1", "2", "3"],             horizontal=True)

        with col2:
            language  = st.radio("Language of instruction", ["Any", "English", "Korean"], horizontal=True)
            major_req = st.toggle("Only show major requirements")
            year      = st.selectbox("Your year", ["1st year", "2nd year", "3rd year", "4th year"])
            mileage   = st.number_input("Your available mileage points", min_value=0, max_value=500, value=100, step=10)

        submitted = st.form_submit_button("Find my courses →", use_container_width=True)

        if submitted:
            prefs = {
                "difficulty": difficulty,
                "workload":   workload,
                "credits":    credits,
                "language":   language,
                "major_req":  major_req,
                "year":       year,
                "mileage":    mileage,
            }
            filtered = filter_courses(CS_COURSES, prefs)

            # Fallback: if filters are too strict, warn but don't block
            if not filtered:
                st.warning("No courses matched your filters — showing all courses instead.")
                filtered = CS_COURSES

            st.session_state.filtered_courses = {
                code: slim_course(c) for code, c in filtered.items()
            }
            st.session_state.prefs = prefs
            st.session_state.intake_done = True
            st.rerun()

# ── 5. CHAT SCREEN ───────────────────────────────────────────────────────────
else:
    with st.sidebar:
        st.markdown("### Your filters")
        p = st.session_state.prefs
        st.write(f"Difficulty: **{p['difficulty']}**")
        st.write(f"Workload: **{p['workload']}**")
        st.write(f"Credits: **{p['credits']}**")
        st.write(f"Language: **{p['language']}**")
        st.write(f"Major req only: **{p['major_req']}**")
        st.write(f"Mileage: **{p['mileage']} pts**")
        st.divider()
        st.metric("Courses matched", len(st.session_state.filtered_courses))
        if st.button("← Change filters"):
            st.session_state.intake_done = False
            st.session_state.messages = []
            st.rerun()

    st.title("🎓 Yonsei Course Assistant")

    # System prompt built from filtered (slimmed) courses only
    course_context = json.dumps(st.session_state.filtered_courses, ensure_ascii=False, indent=2)
    system_prompt = f"""
You are a Yonsei University course registration consultant helping a student
allocate mileage points strategically. Be concise and specific.

Student profile:
- Year: {p['year']}
- Available mileage: {p['mileage']} points
- Preferences: {p['difficulty']} difficulty, {p['workload']} workload, {p['credits']} credits, {p['language']} instruction

The 'added by in ETA' field shows how many students added the course at a given
mileage bid in previous semesters — use it to assess competitiveness and advise
how many points to bid. Warn about overbidding (wasting points) and underbidding
(not getting in). Suggest alternative professors when relevant.

Courses matching the student's filters:
{course_context}
"""

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # First-message prompt so the chat doesn't start blank
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write(
                f"Hi! I found **{len(st.session_state.filtered_courses)} courses** matching your preferences. "
                "Ask me which to bid on, how to split your points, or anything about specific courses or professors."
            )

    if prompt := st.chat_input("Ask about courses, mileage strategy, professors…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = call_llm(system_prompt, st.session_state.messages)
            st.write(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})