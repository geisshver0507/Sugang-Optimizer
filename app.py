import streamlit as st
from groq import Groq
import json
import os

# ── 0. Page config (must be first Streamlit call) ───────────────────────────
st.set_page_config(page_title="Yonsei Course Assistant", layout="wide")

# ── 1. Groq client ──────────────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── 2. Optimized Tree Retrieval Engine ───────────────────────────────────────
def load_tree_database():
    """Safely loads our newly optimized layered tree structure file."""
    filename = "segmented_cs_courses.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.error(f"Missing data tracking asset: '{filename}'. Please run your migration script first.")
        return {}

def filter_tree_courses(tree_db, prefs):
    """
    Traverses the tree branches structurally first, then executes metadata filters
    to extract matching items while remaining lightweight.
    """
    # Step 1: Identify Year Node Branch to prune unnecessary scans
    year_map = {"2nd": "major_year_2", "3rd": "major_year_3", "4th": "major_year_4"}
    target_year_key = year_map.get(prefs["major_year"])
    
    years_to_scan = [target_year_key] if target_year_key else list(tree_db.keys())
    
    # Step 2: Identify Category Node Branches based on checkbox states
    categories_to_scan = []
    if prefs["cat_req"]: categories_to_scan.append("major_requirement")
    if prefs["cat_basic"]: categories_to_scan.append("major_basic")
    if prefs["cat_elec"]: categories_to_scan.append("major_elective")
    
    # Fallback default if zero boxes checked
    if not categories_to_scan:
        categories_to_scan = ["major_requirement", "major_basic", "major_elective", "general_elective"]

    matched_results = {}

    # Step 3: Fast Structural Lookup & Secondary Verification
    for y_key in years_to_scan:
        if y_key not in tree_db:
            continue
        for cat_key in categories_to_scan:
            if cat_key not in tree_db[y_key]:
                continue
            
            # Scan elements inside this precise folder bucket path
            for code, course_obj in tree_db[y_key][cat_key].items():
                meta = course_obj["metadata"]
                
                # Dynamic Filter: Language Medium
                if prefs["language"] != "Any" and meta.get("language_medium") != prefs["language"]:
                    continue
                    
                # Dynamic Filter: Lecture Format Type Mapping
                if prefs["lecture_type"] != "Both":
                    # Maps UI 'Offline' to data 'In-person' if needed
                    match_type = "In-person" if prefs["lecture_type"] == "Offline" else prefs["lecture_type"]
                    if meta.get("lecture_type") != match_type:
                        continue
                        
                # Dynamic Filter: Course Credits Range
                if not (prefs["min_credits"] <= meta.get("credits", 0) <= prefs["max_credits"]):
                    continue
                    
                # Dynamic Filter: Focus Keyword Match Lookups
                if prefs["focus_areas"]:
                    # Lowercase user inputs by cleaning up the parenthetical explanations
                    cleaned_user_areas = [area.split(" (")[0].lower() for area in prefs["focus_areas"]]
                    course_keywords = [k.lower() for k in meta.get("keywords", [])]
                    
                    match_found = any(
                        user_area in course_keywords or any(user_area in kw for kw in course_keywords)
                        for user_area in cleaned_user_areas
                    )
                    if not match_found:
                        continue
                
                # Course passed all evaluations -> Add to selection cache
                # We save the full block so the chat screen can access the text chunks later
                matched_results[code] = course_obj

    return matched_results

def call_llm(system, messages):
    groq_messages = [{"role": "system", "content": system}]
    for m in messages:
        groq_messages.append({"role": m["role"], "content": m["content"]})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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

# Load static tree database representation asset
CS_TREE = load_tree_database()

# ── 4. INTAKE SCREEN ─────────────────────────────────────────────────────────
if not st.session_state.intake_done:
    st.title("🎓 Yonsei Course Assistant")
    st.markdown("Customize your targets to extract your optimal course alignment.")

    st.markdown("""
        <style>
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #2b5c8f !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.form("intake_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Core Parameters")
            language = st.radio("Language Medium", ["Any", "English", "Korean"], horizontal=True)
            lecture_type = st.radio("Lecture Type", ["Both", "Offline", "Blended"], horizontal=True)
            major_year = st.selectbox("Target Major Year", ["Any", "2nd", "3rd", "4th"])
            
            st.markdown("**Course Categories** *(Select all that apply)*")
            cat_req = st.checkbox("Major Requirement", value=True)
            cat_elec = st.checkbox("Major Elective", value=True)
            cat_basic = st.checkbox("Major Basic", value=True)

        with col2:
            st.subheader("🎯 Context & Interests")
            
            st.markdown("**Planned Credit Range**")
            cred_col1, cred_col2 = st.columns(2)
            with cred_col1:
                min_credits = st.number_input("Minimum Course Credits", min_value=1, max_value=21, value=3)
            with cred_col2:
                max_credits = st.number_input("Maximum Course Credits", min_value=1, max_value=21, value=19)

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
            
            mileage = st.number_input("Your Available Mileage Points", min_value=0, max_value=500, value=72, step=1)

        # ── 4b. INTERACTIVE TIMETABLE UI ──────────────────────────────────────
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; margin-bottom: 5px;'>Preferred Class Time Slots</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; font-size: 14px;'>Click and drag across time slots to map out your ideal schedule window.</p>", unsafe_allow_html=True)

        timetable_html = """
        <div id="timetable-container" style="
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
            user-select: none; 
            max-width: 900px; 
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(8px);
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 20px;
            overflow-x: auto;
        ">
            <table style="width: 100%; border-collapse: separate; border-spacing: 4px; text-align: center; font-size: 13px;">
                <thead>
                    <tr>
                        <th style="padding: 10px; width: 80px; color: #495057; font-weight: 600;">Time</th>
                        <th style="padding: 10px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 6px;">Mon <span id="count-Mon" style="font-size:10px; display:block; color:#868e96; font-weight:400;">0h</span></th>
                        <th style="padding: 10px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 6px;">Tue <span id="count-Tue" style="font-size:10px; display:block; color:#868e96; font-weight:400;">0h</span></th>
                        <th style="padding: 10px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 6px;">Wed <span id="count-Wed" style="font-size:10px; display:block; color:#868e96; font-weight:400;">0h</span></th>
                        <th style="padding: 10px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 6px;">Thu <span id="count-Thu" style="font-size:10px; display:block; color:#868e96; font-weight:400;">0h</span></th>
                        <th style="padding: 10px; color: #1c3879; font-weight: 700; background: rgba(28, 56, 121, 0.05); border-radius: 6px;">Fri <span id="count-Fri" style="font-size:10px; display:block; color:#868e96; font-weight:400;">0h</span></th>
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
            tdTime.style.padding = '8px';
            tdTime.style.fontWeight = '600';
            tdTime.style.color = '#495057';
            tdTime.style.background = '#f8f9fa';
            tdTime.style.borderRadius = '6px';
            tr.appendChild(tdTime);

            days.forEach(day => {
                const td = document.createElement('td');
                td.style.height = '32px';
                td.style.cursor = 'pointer';
                td.style.backgroundColor = '#f1f3f5'; 
                td.style.borderRadius = '6px';
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
        st.components.v1.html(timetable_html, height=480, scrolling=False)

        submitted = st.form_submit_button("Generate Strategic Schedule Matching Preferences →", use_container_width=True)

        if submitted:
            prefs = {
                "language": language,
                "lecture_type": lecture_type,
                "major_year": major_year,
                "cat_req": cat_req,
                "cat_elec": cat_elec,
                "cat_basic": cat_basic,
                "min_credits": min_credits,
                "max_credits": max_credits,
                "focus_areas": focus_areas,
                "mileage": mileage,
            }
            
            # Execute RAG structural pruning lookups
            filtered = filter_tree_courses(CS_TREE, prefs)

            # Fallback if filters match nothing
            if not filtered:
                st.warning("No precise matches found. Displaying all tree elements to prevent blank states.")
                # Flattens database for fallback viewing
                filtered = {}
                for year in CS_TREE:
                    for cat in CS_TREE[year]:
                        filtered.update(CS_TREE[year][cat])

            st.session_state.filtered_courses = filtered
            st.session_state.prefs = prefs
            st.session_state.intake_done = True
            st.rerun()

# ── 5. CHAT SCREEN ───────────────────────────────────────────────────────────
else:
    with st.sidebar:
        st.markdown("### Active Constraints")
        p = st.session_state.prefs
        st.write(f"Language: **{p['language']}**")
        st.write(f"Format: **{p['lecture_type']}**")
        st.write(f"Target Year: **{p['major_year']}**")
        st.write(f"Allowed Credits: **{p['min_credits']} - {p['max_credits']} pts**")
        st.write(f"Interests Specified: **{', '.join([a.split(' (')[0] for a in p['focus_areas']]) if p['focus_areas'] else 'All'}**")
        st.write(f"Bidding Capacity: **{p['mileage']} Max Points**")
        st.divider()
        st.metric("Filtered Candidates", len(st.session_state.filtered_courses))
        
        if st.button("← Reset Filters & Availability"):
            st.session_state.intake_done = False
            st.session_state.messages = []
            st.rerun()

    st.title("🎓 Yonsei Course Assistant")

    # Build an ultra-lean metadata map for UI presentation/display logic
    slimmed_display_context = {}
    for code, data in st.session_state.filtered_courses.items():
        m = data["metadata"]
        slimmed_display_context[code] = {
            "name": m["name"],
            "professor": m["professor"],
            "time": m["time"],
            "location": m["location"],
            "mileage_historical_eta": m["mileage_historical_eta"]
        }

    # Construct the RAG Payload Context injection for the LLM
    # Contains the metadata along with the rich, heavy text review chunks
    rag_context_list = []
    for code, data in st.session_state.filtered_courses.items():
        m = data["metadata"]
        t = data["text_chunks"]
        chunk = f"""
        COURSE: {code} - {m['name']}
        Professor: {m['professor']} | Evaluation: {m['evaluation_type']} | Workload: {m['workload']}
        Historical Competitive Bids (ETA): {m['mileage_historical_eta']} points
        Syllabus Context: {t['grading_and_syllabus']}
        Student Feedback Reviews: {t['student_reviews']}
        ---
        """
        rag_context_list.append(chunk)
    
    llm_rag_payload = "\n".join(rag_context_list)

    system_prompt = f"""
You are an expert Yonsei University course registration advisor assisting a computer science student with allocating mileage pool points strategically based on historic enrollment pressure profiles.

Student Profile & Targets:
- Major Year Preference Tier: {p['major_year']}
- Target Credit Windows: {p['min_credits']} to {p['max_credits']} Total Units
- Available Mileage Capital: {p['mileage']} Points

Evaluate historical competition indexes shown within 'Historical Competitive Bids (ETA)' keys to build specific advice. Flag both overbidding trends and underbidding edge conditions dynamically.

Courses matching the customized parameters (RAG Context Injection Chunks):
{llm_rag_payload}
"""

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write(
                f"Hello! I have matched **{len(st.session_state.filtered_courses)} system courses** filtering exactly along your desired categories, keywords, and time profiles. "
                "How should we allocate your available mileage balances across your target distribution?"
            )

    if prompt := st.chat_input("Inquire about allocation strategies, workload balancing, or competitiveness metrics..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing targeted data chunks..."):
                reply = call_llm(system_prompt, st.session_state.messages)
            st.write(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})