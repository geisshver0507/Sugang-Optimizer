"""
app.py
------
NightHawk AI — Yonsei Course Assistant
Part 1 (course selection chatbot) + Part 2 (mileage strategy engine) fully integrated.

Flow:
  Intake → Chat/Timetable → Confirm → Priority Ranking → Mileage Strategy
"""

import base64
from html import escape
import json
import re

import streamlit as st
from groq import Groq

from course_repository import load_tree_database
from course_utils import display_course_name
from filters import filter_tree_courses
from guardrails import recent_conversation, validate_grounding
from prompts import build_system_prompt
from retrieval import extract_course_codes, select_relevant_courses
from schedule_utils import (
    apply_schedule_actions,
    course_summary_for_prompt,
    extract_schedule_actions,
    schedule_total_credits,
)

# Strategy engine imports (Part 2)
from feature_extractor import load_features, flatten_json
from model import load_model, predict_threshold, explain_threshold, FEATURE_LABELS, feature_importance_df
from optimizer import (
    CourseInput, allocate_bids, strategy_summary,
    TOTAL_MILEAGE, MAX_BID_PER_COURSE,
)
from strategy_engine import get_strategy_for_ranked_list, format_strategy_for_chat

import pandas as pd

# ── 0. Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NightHawk AI - Yonsei Course Assistant",
    page_icon="🦅",
    layout="wide",
)

JSON_PATH = "segmented_cs_courses.json"

# ── Cached loaders (Part 2) ─────────────────────────────────────────────────
@st.cache_resource
def get_strategy_model():
    return load_model()

@st.cache_data
def get_df_features(json_path):
    return load_features(json_path)

@st.cache_data
def get_flat_courses(json_path):
    return flatten_json(json_path)

# ── 1. Groq client ──────────────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── 2. LLM bridge ───────────────────────────────────────────────────────────
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
    ("intake_done",           False),
    ("messages",              []),
    ("filtered_courses",      {}),
    ("prefs",                 {}),
    ("retrieved_course_codes",[]),
    ("last_recommended_course_codes", []),
    ("selected_schedule",     {}),
    ("schedule_action_log",   []),
    ("timetable_confirmed",   False),
    ("priority_rankings",     {}),
    ("ai_suggested_rankings", {}),
    # Part 2 state
    ("strategy_results",      None),   # list[BidResult] | None
    ("strategy_generated",    False),
    ("student_year",          2),
]:
    if key not in st.session_state:
        st.session_state[key] = default

try:
    CS_TREE = load_tree_database()
except FileNotFoundError as exc:
    st.error(str(exc))
    CS_TREE = {}

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar width ── */
[data-testid="stSidebar"] {
    min-width: 45vw !important;
    max-width: 50vw !important;
}

/* ── Part 2 bid cards ── */
.bid-card {
    background: #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #3b82f6;
}
.bid-card.safe     { border-left-color: #22c55e; }
.bid-card.moderate { border-left-color: #f59e0b; }
.bid-card.risky    { border-left-color: #ef4444; }

.rule-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}

.strategy-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border: 1px solid #3b82f6;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_weekly_timetable(schedule):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    row_height = 44
    start_period = 1
    end_period = 13
    total_rows = end_period - start_period + 1
    grid_height = total_rows * row_height
    time_labels = [
        "9 AM","10 AM","11 AM","12 PM","1 PM","2 PM","3 PM",
        "4 PM","5 PM","6 PM","7 PM","8 PM","9 PM",
    ]

    time_cells = "".join(
        f"<div class='time-cell' style='top:{idx*row_height}px'>{label}</div>"
        for idx, label in enumerate(time_labels)
    )
    grid_lines = "".join(
        f"<div class='grid-line' style='top:{idx*row_height}px'></div>"
        for idx in range(total_rows + 1)
    )

    blocks = []
    for course in schedule.values():
        meetings = course.get("meetings") or []
        for meeting in meetings:
            day = meeting.get("day")
            if day not in days:
                continue
            top    = (int(meeting.get("start_period", 1)) - start_period) * row_height + 2
            span   = int(meeting.get("end_period", 1)) - int(meeting.get("start_period", 1)) + 1
            height = max(28, span * row_height - 4)
            left   = days.index(day) * 20
            name   = escape(display_course_name(course.get("course_name", "Course")))
            code   = escape(str(course.get("course_id", "")))
            time_text = escape(f"{meeting.get('start_time')} - {meeting.get('end_time')}")
            color  = escape(str(course.get("color", "#2563eb")))
            blocks.append(
                f"<div class='course-block' style='top:{top}px;left:{left}%;width:calc(20% - 8px);height:{height}px;background:{color};'>"
                f"<div class='course-code'>{code}</div>"
                f"<div class='course-name'>{name}</div>"
                f"<div class='course-time'>{time_text}</div>"
                f"</div>"
            )

    empty_state = "" if schedule else "<div class='empty-state'>Ask the chatbot to add courses.</div>"

    html = f"""
    <style>
    .schedule-shell{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#ffffff;border:1px solid #d8dee9;border-radius:8px;overflow:hidden;}}
    .schedule-head{{display:grid;grid-template-columns:58px repeat(5,1fr);background:#0f172a;color:white;font-size:12px;font-weight:700;text-align:center;}}
    .schedule-head div{{padding:10px 4px;border-left:1px solid rgba(255,255,255,0.12);}}
    .schedule-body{{display:grid;grid-template-columns:58px 1fr;height:{grid_height}px;position:relative;background:#f8fafc;}}
    .time-rail{{position:relative;height:{grid_height}px;background:#f1f5f9;border-right:1px solid #cbd5e1;}}
    .time-cell{{position:absolute;left:0;width:100%;height:{row_height}px;padding-top:4px;color:#475569;font-size:11px;text-align:center;border-bottom:1px solid #dbe3ef;box-sizing:border-box;}}
    .days-grid{{position:relative;height:{grid_height}px;background:linear-gradient(to right,transparent 19.8%,#dbe3ef 20%,transparent 20.2%),linear-gradient(to right,transparent 39.8%,#dbe3ef 40%,transparent 40.2%),linear-gradient(to right,transparent 59.8%,#dbe3ef 60%,transparent 60.2%),linear-gradient(to right,transparent 79.8%,#dbe3ef 80%,transparent 80.2%);}}
    .grid-line{{position:absolute;left:0;right:0;border-top:1px solid #e2e8f0;}}
    .course-block{{position:absolute;border-radius:6px;color:#fff;padding:6px 7px;box-sizing:border-box;overflow:hidden;box-shadow:0 6px 14px rgba(15,23,42,0.18);}}
    .course-code{{font-size:11px;font-weight:800;line-height:1.1;}}
    .course-name{{margin-top:3px;font-size:11px;line-height:1.15;max-height:38px;overflow:hidden;}}
    .course-time{{margin-top:3px;font-size:10px;opacity:0.88;}}
    .empty-state{{position:absolute;inset:45% 8px auto 8px;text-align:center;color:#64748b;font-size:13px;font-weight:600;}}
    </style>
    <div class='schedule-shell'>
        <div class='schedule-head'>
            <div>Time</div><div>Mon</div><div>Tue</div><div>Wed</div><div>Thu</div><div>Fri</div>
        </div>
        <div class='schedule-body'>
            <div class='time-rail'>{time_cells}</div>
            <div class='days-grid'>{grid_lines}{''.join(blocks)}{empty_state}</div>
        </div>
    </div>
    """
    st.components.v1.html(html, height=grid_height + 48, scrolling=False)


def strip_schedule_action_text(reply):
    if not reply:
        return ""
    visible_lines = []
    skipping_actions = False
    action_triggers = (
        "json action","json actions","following action","following actions",
        "schedule_actions",'"action',"'action'",
    )
    resume_prefixes = ("grounding check:","schedule update:")
    for line in str(reply).splitlines():
        stripped = line.strip()
        lowered  = stripped.lower()
        if skipping_actions:
            if any(lowered.startswith(p) for p in resume_prefixes):
                skipping_actions = False
                visible_lines.append(line)
            continue
        if any(t in lowered for t in action_triggers) or stripped in ("[","]","{","}"):
            skipping_actions = True
            continue
        visible_lines.append(line)
    cleaned = "\n".join(visible_lines).strip()
    return cleaned or "I updated the timetable based on your request."


def format_assistant_reply(reply):
    text = strip_schedule_action_text(reply)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[^\x00-\x7F]+\s*\(([^()]*[A-Za-z][^()]*)\)", r"\1", text)
    for label in ("Fit","Evidence","Caveat","Schedule","Workload","Mileage","Credits","Prerequisites","Why it fits"):
        text = re.sub(rf"(?<!^)(?<!\n)\s+({re.escape(label)}:)", rf"\n- **\1**", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\n)\s+(\d+\.\s+)", r"\n\n\1", text)
    cleaned_lines = []
    for line in text.splitlines():
        cleaned = line.strip()
        if re.fullmatch(r"[-*•]+\s*", cleaned):
            continue
        cleaned = re.sub(r"(?<!\*)\*\s*$", "", cleaned).strip()
        cleaned_lines.append(cleaned)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def course_label_from_filtered(code):
    course_obj = st.session_state.filtered_courses.get(code, {})
    meta = course_obj.get("metadata", {})
    return f"{display_course_name(meta.get('name', code))} ({code})"


def iter_json_values(text):
    decoder = json.JSONDecoder()
    source  = str(text or "")
    for match in re.finditer(r"[\[{]", source):
        try:
            parsed, _ = decoder.raw_decode(source[match.start():])
        except json.JSONDecodeError:
            continue
        yield parsed


def normalize_priority_recommendation(reply, selected_schedule):
    course_codes = list(selected_schedule.keys())
    course_set   = set(course_codes)
    ranking_items = []
    for parsed in iter_json_values(reply):
        if isinstance(parsed, dict):
            candidate = (
                parsed.get("ranking") or parsed.get("rankings") or
                parsed.get("priorities") or parsed.get("courses")
            )
        elif isinstance(parsed, list):
            candidate = parsed
        else:
            candidate = None
        if isinstance(candidate, list):
            ranking_items = candidate
            break

    normalized = []
    seen = set()
    for fb_idx, item in enumerate(ranking_items, 1):
        if not isinstance(item, dict):
            continue
        code = str(item.get("course_id") or item.get("course_code") or item.get("code") or "").strip()
        if code not in course_set or code in seen:
            continue
        try:
            model_rank = int(item.get("rank") or item.get("priority") or fb_idx)
        except (TypeError, ValueError):
            model_rank = fb_idx
        reasons = item.get("reasons") or item.get("reason") or item.get("rationale") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        reasons = [str(r).strip() for r in reasons if str(r).strip()]
        if not reasons:
            reasons = ["The model selected this position from the confirmed timetable evidence."]
        normalized.append({"course_id": code, "model_rank": model_rank, "reasons": reasons[:3]})
        seen.add(code)

    normalized.sort(key=lambda x: (x["model_rank"], course_codes.index(x["course_id"])))
    for code in course_codes:
        if code not in seen:
            normalized.append({"course_id": code, "model_rank": len(normalized)+1,
                                "reasons": ["Included in confirmed timetable; limited ranking evidence."]})
    for rank, item in enumerate(normalized, 1):
        item["rank"] = rank
    return normalized


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: STRATEGY ENGINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_strategy_engine(priority_rankings: dict, selected_schedule: dict, student_year: int):
    """
    Convert confirmed schedule + priority rankings into BidResult list.
    priority_rankings: {course_code: rank_number}
    """
    try:
        model, explainer = get_strategy_model()
        df_features      = get_df_features(JSON_PATH)
        flat_courses     = get_flat_courses(JSON_PATH)
    except Exception as e:
        st.error(f"Strategy model unavailable: {e}")
        st.info("Run: python3 model.py --train")
        return None

    # Build ranked list for strategy_engine API
    ranked_list = []
    for code, rank in sorted(priority_rankings.items(), key=lambda x: x[1]):
        course_obj = selected_schedule.get(code, {})
        name = display_course_name(course_obj.get("course_name", code))
        ranked_list.append({"code": code, "name": name, "rank": rank})

    results = get_strategy_for_ranked_list(ranked_list, {"year": student_year})
    return results


def render_strategy_panel(results, student_year: int):
    """Render the full mileage strategy UI inline (adapted from strategy_app.py)."""

    RISK_CSS   = {"Safe": "safe", "Moderate": "moderate", "Risky": "risky"}
    RISK_EMOJI = {"Safe": "🟢",   "Moderate": "🟡",       "Risky": "🔴"}

    # Banner
    st.markdown(
        f"""
        <div class="strategy-banner">
            <h2 style="margin:0;color:#f8fafc;">🎯 Mileage Betting Strategy</h2>
            <p style="margin:0.4rem 0 0 0;color:#94a3b8;">
                Budget: <b style="color:#38bdf8;">{TOTAL_MILEAGE} pts</b> &nbsp;|&nbsp;
                Max per course: <b style="color:#38bdf8;">{MAX_BID_PER_COURSE} pts</b> &nbsp;|&nbsp;
                Year: <b style="color:#38bdf8;">{student_year}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_used = sum(r.recommended_bid for r in results)
    remaining  = TOTAL_MILEAGE - total_used

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Budget",     f"{TOTAL_MILEAGE} pts")
    c2.metric("Used",       f"{total_used} pts")
    c3.metric("Remaining",  f"{remaining} pts")
    c4.metric("Max/course", f"{MAX_BID_PER_COURSE} pts")

    st.divider()
    st.subheader("📋 Recommended Bids")

    for r in results:
        css = RISK_CSS.get(r.risk_level, "")
        em  = RISK_EMOJI.get(r.risk_level, "⚪")

        st.markdown(f'<div class="bid-card {css}">', unsafe_allow_html=True)

        ca, cb, cc, cd = st.columns([1, 5, 3, 3])
        ca.markdown(f"**#{r.rank}**")
        cb.markdown(f"**{r.name}**  \n`{r.code}`")
        cc.metric(
            "Recommended Bid",
            f"{r.recommended_bid} pts",
            delta=f"est. threshold: ~{int(r.predicted_threshold)} pts",
            delta_color="off",
        )
        cd.metric(
            "Risk",
            f"{em} {r.risk_level}",
            delta=f"{r.confidence_pct:.0f}% confidence",
            delta_color="off",
        )

        if r.note:
            st.caption(r.note)

        with st.expander("Why this bid?"):
            st.markdown(explain_threshold(r.recommended_bid, r.shap_breakdown, r.name))

            shap_df = pd.DataFrame([
                {"factor": FEATURE_LABELS.get(k, k), "impact": v}
                for k, v in r.shap_breakdown.items()
                if abs(v) > 0.3
            ]).sort_values("impact", key=abs, ascending=False).head(8)

            if not shap_df.empty:
                st.bar_chart(shap_df.set_index("factor")["impact"], height=200)
                st.caption(
                    "Positive = increases competition (bid more needed) | "
                    "Negative = decreases competition"
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📄 Raw strategy text"):
        st.code(strategy_summary(results, TOTAL_MILEAGE))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INTAKE SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state.intake_done:
    st.markdown("""
        <style>
        .block-container { padding-top: 5rem !important; padding-bottom: 1rem !important; max-width: 80vw; }
        footer { visibility: hidden; }
        [data-testid="stForm"] {
            background: linear-gradient(145deg, rgba(30,41,59,0.6), rgba(15,23,42,0.8));
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 10px 40px 0 rgba(0,0,0,0.5);
        }
        </style>
    """, unsafe_allow_html=True)

    col_title, col_filters = st.columns([1.1, 1], gap="large")

    def gif_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    try:
        gif_b64 = gif_to_base64("sonic.gif")
        gif_html = f"<div class='nh-gif-wrap'><img src='data:image/gif;base64,{gif_b64}' alt='NightHawk animation'/></div>"
    except FileNotFoundError:
        gif_html = ""

    with col_title:
        st.markdown(f"""
        <style>
        @keyframes fadeSlideUp {{
            from {{ opacity:0; transform:translateY(28px); }}
            to   {{ opacity:1; transform:translateY(0);    }}
        }}
        .nh-hero-wrap {{ display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;height:100%;min-height:500px;gap:0; }}
        .nh-title     {{ margin:0;font-size:5.5rem;font-weight:800;line-height:1.1;opacity:0;animation:fadeSlideUp 0.8s cubic-bezier(0.22,1,0.36,1) 0.1s forwards; }}
        .nh-subtitle  {{ color:#94a3b8;margin-top:15px;font-weight:400;font-size:1.3rem;line-height:1.6;max-width:85%;opacity:0;animation:fadeSlideUp 0.8s cubic-bezier(0.22,1,0.36,1) 0.4s forwards; }}
        .nh-gif-wrap  {{ margin-top:28px;opacity:0;animation:fadeSlideUp 0.8s cubic-bezier(0.22,1,0.36,1) 0.7s forwards; }}
        .nh-gif-wrap img {{ width:220px;border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,0.45); }}
        </style>
        <div class='nh-hero-wrap'>
            <h1 class='nh-title'>Hi! I'm NightHawk AI🦅</h1>
            <h5 class='nh-subtitle'>
                Yonsei Course Assistant &mdash; Customize your targets to extract your optimal
                course alignment, then get a mileage betting strategy automatically.
            </h5>
            {gif_html}
        </div>
        """, unsafe_allow_html=True)

    with col_filters:
        st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)

        with st.form("intake_form"):
            st.markdown("<h3 style='text-align:center;margin-bottom:25px;'>📋 Search Parameters</h3>", unsafe_allow_html=True)

            left, right = st.columns(2)
            with left:  language     = st.radio("Language Medium", ["Any","English","Korean"], horizontal=True)
            with right: lecture_type = st.radio("Lecture Type",    ["Both","Offline","Blended"], horizontal=True)

            st.markdown("<div style='margin-top:15px;margin-bottom:5px;'><b>Course Categories</b></div>", unsafe_allow_html=True)
            cat_col1, cat_col2, cat_col3 = st.columns(3)
            with cat_col1: cat_req   = st.checkbox("Major Requirement", value=True)
            with cat_col2: cat_elec  = st.checkbox("Major Elective",    value=True)
            with cat_col3: cat_basic = st.checkbox("Major Basic",        value=True)

            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
            major_year  = st.selectbox("Target Major Year", ["Any","2nd","3rd","4th"], index=2)
            max_credits = st.number_input("Course Credit Taking", min_value=0, max_value=21, value=19, step=1)

            # ── NEW: student year for mileage strategy ──────────────────
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            student_year_input = st.selectbox(
                "Your University Year (for mileage strategy)",
                [1, 2, 3, 4], index=1,
                help="Used later by the mileage betting engine to predict competition levels.",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Generate Strategic Schedule Matching Preferences →",
                use_container_width=True,
            )

            if submitted:
                prefs = {
                    "language":     language,
                    "lecture_type": lecture_type,
                    "major_year":   major_year,
                    "major_years":  [] if major_year == "Any" else [major_year],
                    "cat_req":      cat_req,
                    "cat_elec":     cat_elec,
                    "cat_basic":    cat_basic,
                    "max_credits":  max_credits,
                    "focus_areas":  [],
                }
                filtered = filter_tree_courses(CS_TREE, prefs)
                if not filtered:
                    st.warning("No courses matched those filters. Please loosen one or more filters.")
                else:
                    st.session_state.filtered_courses = filtered
                    st.session_state.prefs            = prefs
                    st.session_state.student_year     = student_year_input
                    st.session_state.intake_done      = True
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN CHAT SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
else:
    selected_schedule = st.session_state.selected_schedule
    p                 = st.session_state.prefs
    student_year      = st.session_state.student_year

    # ── SIDEBAR ─────────────────────────────────────────────────────────────
    with st.sidebar:
        filter_col, time_col = st.columns([1, 1.4], gap="medium")

        # ── Filter column ──────────────────────────────────────────────────
        with filter_col:
            year_options     = ["1st","2nd","3rd","4th"]
            language_options = ["Any","English","Korean"]

            current_years = p.get("major_years") or []
            if not current_years and p.get("major_year") != "Any":
                current_years = [p.get("major_year","3rd")]
            current_years  = [y for y in current_years if y in year_options]
            default_years  = current_years or year_options

            current_language = p.get("language","Any")
            if current_language not in language_options:
                current_language = "Any"

            st.markdown("### Dynamic Filters")
            selected_years    = st.multiselect("Major Years", year_options, default=default_years, key="chat_major_years")
            selected_language = st.radio("Language Medium", language_options,
                                         index=language_options.index(current_language), key="chat_language")

            stored_years    = current_years or year_options
            filters_changed = (selected_language != p.get("language","Any") or selected_years != stored_years)

            if filters_changed:
                updated_p = dict(p)
                updated_p["language"]    = selected_language
                updated_p["major_years"] = selected_years
                if not selected_years or set(selected_years) == set(year_options):
                    updated_p["major_year"] = "Any"
                elif len(selected_years) == 1:
                    updated_p["major_year"] = selected_years[0]
                else:
                    updated_p["major_year"] = ", ".join(selected_years)

                updated_filtered = filter_tree_courses(CS_TREE, updated_p)
                st.session_state.prefs            = updated_p
                st.session_state.filtered_courses = updated_filtered
                st.session_state.retrieved_course_codes = []
                st.session_state.last_recommended_course_codes = []
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
            st.write(f"Student Year: **Year {student_year}**")
            st.metric("Filtered Candidates", len(st.session_state.filtered_courses))
            st.metric("Total Credits", schedule_total_credits(selected_schedule))

            if st.button("Reset Filters & Availability", use_container_width=True):
                for k in ("intake_done","messages","retrieved_course_codes","selected_schedule",
                          "schedule_action_log","timetable_confirmed","priority_rankings",
                          "ai_suggested_rankings","strategy_results","strategy_generated",
                          "last_recommended_course_codes","chat_major_years","chat_language"):
                    st.session_state.pop(k, None)
                st.session_state.intake_done = False
                st.rerun()

            # ── Priority Ranking (shown after timetable confirmed) ─────────
            if st.session_state.timetable_confirmed and selected_schedule:
                st.markdown("### Priority Ranking")
                st.caption("AI suggested rankings below — adjust if needed.")
                course_count  = len(selected_schedule)
                ai_suggestions = st.session_state.get("ai_suggested_rankings", {})

                with st.form("priority_ranking_form"):
                    ranking_values = {}
                    for code, course in selected_schedule.items():
                        label      = f"{display_course_name(course.get('course_name'))} ({code})"
                        ai_default = max(1, min(ai_suggestions.get(code, 1), course_count))
                        ranking_values[code] = st.selectbox(
                            label,
                            list(range(1, course_count + 1)),
                            index=ai_default - 1,
                            key=f"priority_{code}",
                        )

                    save_col, gen_col = st.columns(2)
                    submitted_rankings = save_col.form_submit_button("💾 Save Rankings", use_container_width=True)
                    gen_strategy_btn   = gen_col.form_submit_button(
                        "🎯 Generate Mileage Strategy",
                        use_container_width=True,
                        type="primary",
                    )

                    if submitted_rankings:
                        if len(set(ranking_values.values())) != course_count:
                            st.warning("Each course needs a unique priority number.")
                        else:
                            st.session_state.priority_rankings = ranking_values
                            st.success("Priority ranking saved.")

                    if gen_strategy_btn:
                        # Validate uniqueness first
                        if len(set(ranking_values.values())) != course_count:
                            st.warning("Each course needs a unique priority number before generating the strategy.")
                        else:
                            st.session_state.priority_rankings = ranking_values
                            with st.spinner("Running mileage strategy engine..."):
                                results = run_strategy_engine(
                                    ranking_values,
                                    selected_schedule,
                                    student_year,
                                )
                            if results:
                                st.session_state.strategy_results  = results
                                st.session_state.strategy_generated = True
                                st.rerun()

        # ── Timetable column ───────────────────────────────────────────────
        with time_col:
            st.markdown("### 📅 Weekly Timetable")
            render_weekly_timetable(selected_schedule)

            if selected_schedule:
                with st.expander("Selected Courses", expanded=True):
                    for code, course in selected_schedule.items():
                        st.write(f"**{display_course_name(course.get('course_name'))}** ({code})")
                        st.caption(f"{course.get('credits',0)} credits | {course.get('time') or 'Time not listed'}")
            else:
                st.info("The timetable is empty. Ask the assistant to add a course.")

            if st.session_state.schedule_action_log:
                with st.expander("Schedule Action Log"):
                    for item in st.session_state.schedule_action_log[-8:]:
                        st.write(item)

            # ── Confirm Timetable button ───────────────────────────────────
            if not st.session_state.timetable_confirmed:
                if st.button("Confirm Timetable", disabled=not selected_schedule, use_container_width=True):
                    final_list = course_summary_for_prompt(selected_schedule)
                    confirmation_prompt = (
                        "The user has confirmed this timetable. "
                        "First, explain your overall strategic reasoning in a short, friendly paragraph. "
                        "Second, list the courses in priority order using a numbered list. "
                        "Finally, append the raw JSON object at the very end. "
                        f"Here is the final selected course list:\n{final_list}"
                    )
                    llm_messages   = st.session_state.messages + [{"role":"user","content":confirmation_prompt}]
                    ranking_reply  = call_llm(
                        "You are NightHawk AI. The timetable is confirmed. "
                        "Explain your reasoning first, provide a numbered list, "
                        "and place the JSON dictionary {\"COURSE_CODE\": rank_number, ...} at the VERY END.",
                        llm_messages,
                    )
                    suggested_rankings = {}
                    try:
                        json_match = re.search(r'\{[^{}]+\}', ranking_reply, re.DOTALL)
                        if json_match:
                            suggested_rankings = json.loads(json_match.group())
                            suggested_rankings = {k: int(v) for k, v in suggested_rankings.items()}
                    except Exception:
                        pass

                    used_ranks = set(suggested_rankings.values())
                    next_rank  = 1
                    for code in selected_schedule.keys():
                        if code not in suggested_rankings:
                            while next_rank in used_ranks:
                                next_rank += 1
                            suggested_rankings[code] = next_rank
                            used_ranks.add(next_rank)

                    st.session_state.messages.append({"role":"user","content":"Confirmed timetable."})
                    display_text = re.sub(r'\{[^{}]+\}', '', ranking_reply).strip()
                    st.session_state.messages.append({"role":"assistant","content":format_assistant_reply(display_text)})
                    st.session_state.timetable_confirmed    = True
                    st.session_state.ai_suggested_rankings  = suggested_rankings
                    st.rerun()
            else:
                # Already confirmed — show a small status label
                st.success("✅ Timetable confirmed")
                if st.session_state.strategy_generated:
                    st.info("🎯 Strategy generated — see main area below")

    # ── MAIN AREA ────────────────────────────────────────────────────────────

    # ── Strategy panel (shown ABOVE chat once generated) ────────────────────
    if st.session_state.strategy_generated and st.session_state.strategy_results:
        st.markdown("---")
        render_strategy_panel(st.session_state.strategy_results, student_year)
        st.markdown("---")

        # Option to regenerate with a different safety margin
        with st.expander("⚙️ Regenerate with custom safety buffer"):
            safety_pct = st.slider("Safety buffer (%)", 0, 30, 15, key="regenerate_safety")
            if st.button("Regenerate Strategy", key="regen_btn"):
                try:
                    model, explainer = get_strategy_model()
                    df_features      = get_df_features(JSON_PATH)
                    flat_courses     = get_flat_courses(JSON_PATH)

                    course_inputs = []
                    rankings = st.session_state.priority_rankings
                    rank_data = sorted(rankings.items(), key=lambda x: x[1])

                    for code, rank in rank_data:
                        if code not in df_features.index:
                            continue
                        feat = df_features.loc[code].to_dict()
                        feat.update({
                            "student_year":       float(student_year),
                            "student_mileage":    float(TOTAL_MILEAGE),
                            "num_courses_wanted": float(len(rank_data)),
                            "rank_in_list":       float(rank),
                            "priority_ratio":     (len(rank_data) - rank + 1) / len(rank_data),
                            "budget_ratio":       1.0,
                        })
                        threshold, shap = predict_threshold(model, explainer, feat)
                        c_info = flat_courses.get(code, {})
                        course_obj = selected_schedule.get(code, {})
                        course_inputs.append(CourseInput(
                            code                = code,
                            name                = display_course_name(course_obj.get("course_name", code)),
                            rank                = rank,
                            predicted_threshold = threshold,
                            is_major_req        = (c_info.get("category") == "major_requirement"),
                            shap_breakdown      = shap,
                        ))

                    new_results = allocate_bids(
                        course_inputs,
                        total_mileage  = TOTAL_MILEAGE,
                        max_per_course = MAX_BID_PER_COURSE,
                        safety_margin  = safety_pct / 100.0,
                    )
                    st.session_state.strategy_results = new_results
                    st.rerun()
                except Exception as e:
                    st.error(f"Regeneration failed: {e}")

        st.markdown("---")

    # ── Chat interface ───────────────────────────────────────────────────────
    st.title("NightHawk AI - Yonsei Course Assistant 🎓")

    # Show prompt nudge if strategy not yet generated but timetable confirmed
    if st.session_state.timetable_confirmed and not st.session_state.strategy_generated:
        st.info(
            "✅ Timetable confirmed! Set your priority rankings in the sidebar, "
            "then click **🎯 Generate Mileage Strategy** to get your bidding plan."
        )

    with st.container(height=700, border=False):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                content = msg["content"]
                if msg["role"] == "assistant":
                    content = format_assistant_reply(content)
                st.write(content)

        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.write(
                    f"Hello! I have matched **{len(st.session_state.filtered_courses)} system courses** "
                    "filtering exactly along your desired categories and keywords. "
                    "Tell me which courses you want to add, remove, or optimize, and I will update the timetable automatically."
                )

    if prompt := st.chat_input("Ask to add, remove, optimize, or compare courses..."):
        st.session_state.messages.append({"role":"user","content":prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing targeted data chunks..."):
                prior_course_codes = (
                    st.session_state.get("last_recommended_course_codes")
                    or st.session_state.get("retrieved_course_codes")
                    or []
                )
                selected_courses = select_relevant_courses(
                    st.session_state.filtered_courses,
                    prompt,
                    limit=max(10, len(prior_course_codes)),
                    prior_codes=prior_course_codes,
                )
                st.session_state.retrieved_course_codes = list(selected_courses.keys())
                system_prompt = build_system_prompt(
                    p,
                    selected_courses,
                    st.session_state.filtered_courses,
                    st.session_state.selected_schedule,
                )
                reply  = call_llm(system_prompt, st.session_state.messages)
                reply  = validate_grounding(reply, selected_courses, st.session_state.filtered_courses)

                actions = extract_schedule_actions(reply, user_prompt=prompt)
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
                    st.session_state.priority_rankings   = {}
                    # Invalidate strategy when schedule changes
                    st.session_state.strategy_generated  = False
                    st.session_state.strategy_results    = None

            if st.session_state.retrieved_course_codes:
                evidence_labels = [
                    course_label_from_filtered(code)
                    for code in st.session_state.retrieved_course_codes
                ]
                st.caption("Evidence used: " + ", ".join(evidence_labels))

            display_reply = format_assistant_reply(reply)
            if action_results:
                display_reply += "\n\nSchedule update:\n" + "\n".join(action_results)
            st.write(display_reply)

        mentioned_codes = [
            code for code in extract_course_codes(display_reply)
            if code in st.session_state.filtered_courses
        ]
        if mentioned_codes:
            st.session_state.last_recommended_course_codes = mentioned_codes

        st.session_state.messages.append({"role":"assistant","content":display_reply})
        st.rerun()
