"""
integrated_app.py
-----------------
NightHawk AI — Yonsei Course Assistant
Part 1 (course selection chatbot) + Part 2 (mileage strategy engine) fully integrated.

Flow:
  Intake → Chat/Timetable → Confirm → Priority Ranking → Mileage Strategy

Part 1 modules live in  ./part1/
Part 2 modules live in  ./part2/
Data files live in      ./database/
"""

import base64
from pathlib import Path
import sys
from html import escape
import json
import re

ROOT         = Path(__file__).resolve().parent
DATABASE_DIR = ROOT / "database"
ASSETS_DIR   = ROOT / "assets"

# ── Add both sub-packages to path ───────────────────────────────────────────
for module_dir in (ROOT / "part1", ROOT / "part2"):
    p = str(module_dir)
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
from groq import Groq

# ── Part 1 imports ───────────────────────────────────────────────────────────
from course_repository import load_tree_database
from course_utils      import display_course_name
from filters           import filter_tree_courses
from guardrails        import recent_conversation, validate_grounding
from prompts           import build_system_prompt
from retrieval         import extract_course_codes, select_relevant_courses
from schedule_utils    import (
    apply_schedule_actions,
    course_summary_for_prompt,
    extract_schedule_actions,
    schedule_total_credits,
)

# ── Part 2 imports ───────────────────────────────────────────────────────────
from survival_model    import load_applicant_data, load_course_meta, build_curve, confidence_label
from allocator         import TOTAL_MILEAGE, MAX_BID
from strategy_engine_v2 import get_strategy_for_ranked_list, format_strategy_for_chat

# ── Data paths ───────────────────────────────────────────────────────────────
JSON_PATH    = str(DATABASE_DIR / "segmented_cs_courses.json")
RAW_CSV_PATH = str(DATABASE_DIR / "mileage_history_all.csv")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NightHawk AI - Yonsei Course Assistant",
    page_icon="🦅",
    layout="wide",
)

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── LLM bridge ───────────────────────────────────────────────────────────────
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

# ── Session state init ───────────────────────────────────────────────────────
for key, default in [
    ("intake_done",                    False),
    ("messages",                       []),
    ("filtered_courses",               {}),
    ("prefs",                          {}),
    ("retrieved_course_codes",         []),
    ("last_recommended_course_codes",  []),
    ("selected_schedule",              {}),
    ("schedule_action_log",            []),
    ("timetable_confirmed",            False),
    ("priority_rankings",              {}),
    ("ai_suggested_rankings",          {}),
    ("strategy_results",               None),
    ("strategy_generated",             False),
    ("student_year",                   2),
]:
    if key not in st.session_state:
        st.session_state[key] = default

try:
    CS_TREE = load_tree_database()
except FileNotFoundError as exc:
    st.error(str(exc))
    CS_TREE = {}

# ── Cached strategy data loaders ─────────────────────────────────────────────
@st.cache_data
def _load_strategy_data():
    try:
        df   = load_applicant_data(RAW_CSV_PATH)
        meta = load_course_meta(JSON_PATH, RAW_CSV_PATH)
        return df, meta
    except Exception:
        return None, {}

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# Two layers:
#   A) Original NightHawk CSS — intake form, timetable, chat layout (unchanged)
#   B) New Yonsei strategy card CSS — replaces old .bid-card system
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@300;400;500&display=swap');

/* ── A. STRUCTURAL / LAYOUT (unchanged from original) ── */

[data-testid="stSidebar"] {
    min-width: 45vw !important;
    max-width: 50vw !important;
}

/* ── B. YONSEI STRATEGY CARD SYSTEM ── */

/* Scoped inside .ys-scope so it doesn't affect Part 1 UI */
.ys-scope {
    font-family: 'Outfit', 'Noto Sans KR', sans-serif;
}

/* Strategy banner */
.ys-banner {
    background: linear-gradient(135deg, #07101F 0%, #0F1E35 60%, #0E2040 100%);
    border: 1px solid rgba(27,79,216,0.3);
    border-radius: 12px;
    padding: 1.3rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
}
.ys-banner-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #EDF2F8;
    letter-spacing: -0.01em;
}
.ys-banner-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4A6080;
    margin-top: 0.2rem;
    letter-spacing: 0.06em;
}
.ys-banner-badge {
    background: rgba(27,79,216,0.15);
    border: 1px solid rgba(27,79,216,0.3);
    color: #4D80E4;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.3rem 0.75rem;
    border-radius: 4px;
    letter-spacing: 0.06em;
}

/* Budget bar */
.ys-budget {
    background: #0A1525;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.9rem 1.3rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.ys-budget-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3D566E;
    margin-bottom: 0.2rem;
    font-family: 'Outfit', sans-serif;
}
.ys-budget-val {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    color: #EDF2F8;
    font-weight: 500;
}
.ys-budget-bar-wrap { flex: 1; min-width: 120px; }
.ys-budget-bar-track {
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
    height: 3px;
    overflow: hidden;
    margin-top: 0.35rem;
}
.ys-budget-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: #1B4FD8;
}

/* Summary row */
.ys-summary {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin-bottom: 1rem;
}
.ys-sum-card {
    background: #0A1525;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 7px;
    padding: 0.75rem 0.9rem;
}
.ys-sum-label {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3D566E;
    margin-bottom: 0.25rem;
    font-family: 'Outfit', sans-serif;
}
.ys-sum-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.25rem;
    font-weight: 500;
    color: #EDF2F8;
}
.ys-sum-sub {
    font-size: 0.62rem;
    color: #3D566E;
    margin-top: 0.1rem;
    font-family: 'DM Mono', monospace;
}

/* Bid cards */
.ys-card {
    background: #0A1525;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 9px;
    margin-bottom: 0.65rem;
    overflow: hidden;
    transition: border-color 0.2s;
}
.ys-card:hover { border-color: rgba(77,128,228,0.2); }
.ys-card-accent { height: 2px; width: 100%; }
.ys-card-body { padding: 1rem 1.25rem 0.9rem 1.25rem; }

.ys-card-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 0.85rem;
}
.ys-card-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #3D566E;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 0.18rem 0.45rem;
    margin-right: 0.65rem;
    flex-shrink: 0;
    margin-top: 0.12rem;
}
.ys-card-info { flex: 1; }
.ys-card-name {
    font-family: 'Outfit', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #EDF2F8;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.ys-card-sub {
    font-size: 0.68rem;
    color: #3D566E;
    margin-top: 0.18rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.02em;
}
.ys-status-pill {
    font-family: 'Outfit', sans-serif;
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.22rem 0.6rem;
    border-radius: 20px;
    flex-shrink: 0;
    white-space: nowrap;
}

/* Card grid */
.ys-card-grid {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr 1fr;
    gap: 0.9rem;
    padding: 0.75rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 0.75rem;
    align-items: start;
}
.ys-stat-label {
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3D566E;
    margin-bottom: 0.25rem;
    font-family: 'Outfit', sans-serif;
}
.ys-stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.2rem;
    font-weight: 500;
    color: #EDF2F8;
    line-height: 1;
}
.ys-stat-unit {
    font-size: 0.65rem;
    color: #3D566E;
    margin-left: 0.12rem;
    vertical-align: super;
}
.ys-stat-sub {
    font-size: 0.62rem;
    color: #3D566E;
    margin-top: 0.22rem;
    font-family: 'DM Mono', monospace;
}
.ys-stat-sm {
    font-family: 'Outfit', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #8FA3BC;
    line-height: 1.2;
}
.ys-prob-track {
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
    height: 3px;
    overflow: hidden;
    margin-top: 0.35rem;
}
.ys-prob-fill { height: 100%; border-radius: 2px; }
.ys-dq-tag {
    display: inline-block;
    font-size: 0.59rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
    padding: 0.12rem 0.4rem;
    border-radius: 3px;
    margin-top: 0.22rem;
}
.ys-dq-high   { background: rgba(16,185,129,0.1);  color: #10B981; border: 1px solid rgba(16,185,129,0.2); }
.ys-dq-medium { background: rgba(245,166,35,0.1);  color: #F5A623; border: 1px solid rgba(245,166,35,0.2); }
.ys-dq-low    { background: rgba(239,68,68,0.1);   color: #EF4444; border: 1px solid rgba(239,68,68,0.2); }

.ys-card-note {
    font-size: 0.72rem;
    color: #4A6080;
    line-height: 1.5;
    font-family: 'Outfit', sans-serif;
}

.ys-section-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3D566E;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.ys-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* ── C. Original intake form / timetable CSS (unchanged) ── */

.schedule-shell { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#ffffff; border:1px solid #d8dee9; border-radius:8px; overflow:hidden; }
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


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 HELPERS  (unchanged from original integrated_app.py)
# ═════════════════════════════════════════════════════════════════════════════

def render_weekly_timetable(schedule):
    days = ["Mon","Tue","Wed","Thu","Fri"]
    row_height = 44; start_period = 1; end_period = 13
    total_rows = end_period - start_period + 1
    grid_height = total_rows * row_height
    time_labels = ["9 AM","10 AM","11 AM","12 PM","1 PM","2 PM","3 PM",
                   "4 PM","5 PM","6 PM","7 PM","8 PM","9 PM"]
    time_cells = "".join(
        f"<div class='time-cell' style='top:{idx*row_height}px'>{label}</div>"
        for idx, label in enumerate(time_labels))
    grid_lines = "".join(
        f"<div class='grid-line' style='top:{idx*row_height}px'></div>"
        for idx in range(total_rows + 1))
    blocks = []
    for course in schedule.values():
        meetings = course.get("meetings") or []
        for meeting in meetings:
            day = meeting.get("day")
            if day not in days: continue
            top    = (int(meeting.get("start_period",1)) - start_period) * row_height + 2
            span   = int(meeting.get("end_period",1)) - int(meeting.get("start_period",1)) + 1
            height = max(28, span * row_height - 4)
            left   = days.index(day) * 20
            name   = escape(display_course_name(course.get("course_name","Course")))
            code   = escape(str(course.get("course_id","")))
            time_text = escape(f"{meeting.get('start_time')} - {meeting.get('end_time')}")
            color  = escape(str(course.get("color","#2563eb")))
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
    </div>"""
    st.components.v1.html(html, height=grid_height + 48, scrolling=False)


def strip_schedule_action_text(reply):
    if not reply: return ""
    visible_lines = []
    skipping_actions = False
    action_triggers = ("json action","json actions","following action","following actions",
                       "schedule_actions",'"action',"'action'")
    resume_prefixes = ("grounding check:","schedule update:")
    for line in str(reply).splitlines():
        stripped = line.strip(); lowered = stripped.lower()
        if skipping_actions:
            if any(lowered.startswith(p) for p in resume_prefixes):
                skipping_actions = False; visible_lines.append(line)
            continue
        if any(t in lowered for t in action_triggers) or stripped in ("[","]","{","}"):
            skipping_actions = True; continue
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
        if re.fullmatch(r"[-*•]+\s*", cleaned): continue
        cleaned = re.sub(r"(?<!\*)\*\s*$","",cleaned).strip()
        cleaned_lines.append(cleaned)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def course_label_from_filtered(code):
    course_obj = st.session_state.filtered_courses.get(code, {})
    meta = course_obj.get("metadata", {})
    return f"{display_course_name(meta.get('name', code))} ({code})"


def iter_json_values(text):
    decoder = json.JSONDecoder(); source = str(text or "")
    for match in re.finditer(r"[\[{]", source):
        try:
            parsed, _ = decoder.raw_decode(source[match.start():])
        except json.JSONDecodeError:
            continue
        yield parsed


def normalize_priority_recommendation(reply, selected_schedule):
    course_codes = list(selected_schedule.keys()); course_set = set(course_codes)
    ranking_items = []
    for parsed in iter_json_values(reply):
        if isinstance(parsed, dict):
            candidate = (parsed.get("ranking") or parsed.get("rankings") or
                         parsed.get("priorities") or parsed.get("courses"))
        elif isinstance(parsed, list):
            candidate = parsed
        else:
            candidate = None
        if isinstance(candidate, list):
            ranking_items = candidate; break
    normalized = []; seen = set()
    for fb_idx, item in enumerate(ranking_items, 1):
        if not isinstance(item, dict): continue
        code = str(item.get("course_id") or item.get("course_code") or item.get("code") or "").strip()
        if code not in course_set or code in seen: continue
        try:
            model_rank = int(item.get("rank") or item.get("priority") or fb_idx)
        except (TypeError, ValueError):
            model_rank = fb_idx
        reasons = item.get("reasons") or item.get("reason") or item.get("rationale") or []
        if isinstance(reasons, str): reasons = [reasons]
        reasons = [str(r).strip() for r in reasons if str(r).strip()]
        if not reasons: reasons = ["The model selected this position from the confirmed timetable evidence."]
        normalized.append({"course_id": code, "model_rank": model_rank, "reasons": reasons[:3]}); seen.add(code)
    normalized.sort(key=lambda x: (x["model_rank"], course_codes.index(x["course_id"])))
    for code in course_codes:
        if code not in seen:
            normalized.append({"course_id": code, "model_rank": len(normalized)+1,
                                "reasons": ["Included in confirmed timetable; limited ranking evidence."]})
    for rank, item in enumerate(normalized, 1):
        item["rank"] = rank
    return normalized


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 HELPERS — new Yonsei strategy card system
# ═════════════════════════════════════════════════════════════════════════════

STATUS_COLOR = {
    "Secured":  "#10D9A0",
    "Likely":   "#4D80E4",
    "Risky":    "#F5A623",
    "Stretch":  "#F5A623",
    "Drop":     "#EF4444",
    "Unfunded": "#EF4444",
}
STATUS_LABEL = {
    "Secured":  "SECURED",
    "Likely":   "LIKELY",
    "Risky":    "UNCERTAIN",
    "Stretch":  "UNCERTAIN",
    "Drop":     "UNLIKELY",
    "Unfunded": "UNFUNDED",
}

def _source_label(source: str) -> str:
    s = source.lower()
    if "same_year" in s and "same_prof" in s: return "Year + Prof history"
    if "same_year" in s:                       return "Year-specific history"
    if "same_professor" in s:                  return "Prof. history"
    if "same_course" in s:                     return "Course history"
    return "Demand proxy"


def render_bid_card(r, student_year: int, df_raw, meta):
    color   = STATUS_COLOR.get(r.status, "#6B7280")
    label   = STATUS_LABEL.get(r.status, r.status.upper())
    prob    = int(r.win_prob * 100)
    src     = _source_label(r.data_source)
    dq_cls  = {"High":"ys-dq-high","Medium":"ys-dq-medium","Low":"ys-dq-low"}.get(r.confidence,"ys-dq-low")
    note_first = r.note.split("\n")[0].strip()[:240] if r.note else ""

    st.markdown(f"""
<div class="ys-card">
  <div class="ys-card-accent" style="background:{color}"></div>
  <div class="ys-card-body">
    <div class="ys-card-top">
      <div class="ys-card-rank">#{r.rank}</div>
      <div class="ys-card-info">
        <div class="ys-card-name">{r.name}</div>
        <div class="ys-card-sub">{r.code}&nbsp;&nbsp;·&nbsp;&nbsp;Prof.&nbsp;{r.professor or '—'}</div>
      </div>
      <div class="ys-status-pill" style="background:{color}18;color:{color};border:1px solid {color}35">
        {label}
      </div>
    </div>
    <div class="ys-card-grid">
      <div>
        <div class="ys-stat-label">Recommended Bid</div>
        <div class="ys-stat-value">{r.bid}<span class="ys-stat-unit">pt</span></div>
        <div class="ys-stat-sub">safe min: {r.min_safe_bid}pt</div>
      </div>
      <div>
        <div class="ys-stat-label">Win Probability</div>
        <div class="ys-stat-value" style="color:{color}">{prob}<span class="ys-stat-unit">%</span></div>
        <div class="ys-prob-track">
          <div class="ys-prob-fill" style="width:{prob}%;background:{color}"></div>
        </div>
        <div class="ys-stat-sub" style="margin-top:0.3rem">calibrated from historical bids</div>
      </div>
      <div>
        <div class="ys-stat-label">Competition</div>
        <div class="ys-stat-sm">{r.competition}</div>
        <div class="ys-stat-sub">[{r.min_safe_bid}pt safe]</div>
      </div>
      <div>
        <div class="ys-stat-label">Data Quality</div>
        <div class="ys-stat-sm">{r.confidence}</div>
        <div class="ys-dq-tag {dq_cls}">{src}</div>
        <div class="ys-stat-sub" style="margin-top:0.3rem">{r.confidence_pct:.0f}% est. confidence</div>
      </div>
    </div>
    <div class="ys-card-note">{note_first}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Extra note lines + curve in expander
    extra_notes = [l.strip() for l in r.note.split("\n")[1:] if l.strip()]
    with st.expander("↗ Survival curve & full notes"):
        if extra_notes:
            for line in extra_notes:
                st.caption(line)
            st.divider()
        if df_raw is not None:
            try:
                import pandas as _pd
                curve = build_curve(df_raw, r.code, r.professor,
                                    student_year=student_year, meta=meta)
                if len(curve.bids) > 0:
                    cdf = _pd.DataFrame({"bid": curve.bids, "P(win)": curve.p_enroll}).set_index("bid")
                    st.line_chart(cdf, height=160)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Your bid",   f"{r.bid}pt → {prob}%")
                    c2.metric("Ceiling",    f"{curve.reachable_max():.0%}")
                    c3.metric("Data yrs",   str(curve.data_years or "—"))
                    st.caption(f"Source: `{curve.source}` · {curve.n_effective:.0f} weighted rows")
                else:
                    st.caption("No historical curve — demand proxy used.")
            except Exception as e:
                st.caption(f"Curve unavailable: {e}")
        else:
            st.caption("Historical data not loaded — curve display unavailable.")


def render_strategy_summary(results):
    """Budget bar + 4-column summary grid."""
    total_used = sum(r.bid for r in results)
    pct        = int(total_used / TOTAL_MILEAGE * 100)
    n_good     = sum(1 for r in results if r.status in ("Secured","Likely"))
    n_risky    = sum(1 for r in results if r.status in ("Risky","Stretch"))
    n_drop     = sum(1 for r in results if r.status in ("Drop","Unfunded"))
    remaining  = TOTAL_MILEAGE - total_used

    st.markdown(f"""
<div class="ys-budget">
  <div>
    <div class="ys-budget-label">Points Used</div>
    <div class="ys-budget-val">{total_used} / {TOTAL_MILEAGE}</div>
  </div>
  <div class="ys-budget-bar-wrap">
    <div class="ys-budget-label">{pct}% of budget allocated</div>
    <div class="ys-budget-bar-track">
      <div class="ys-budget-bar-fill" style="width:{pct}%"></div>
    </div>
  </div>
  <div>
    <div class="ys-budget-label">Remaining</div>
    <div class="ys-budget-val">{remaining}pt</div>
  </div>
</div>
<div class="ys-summary">
  <div class="ys-sum-card">
    <div class="ys-sum-label">Courses</div>
    <div class="ys-sum-value">{len(results)}</div>
    <div class="ys-sum-sub">in strategy</div>
  </div>
  <div class="ys-sum-card" style="border-top:2px solid #10D9A0">
    <div class="ys-sum-label">Secured / Likely</div>
    <div class="ys-sum-value" style="color:#10D9A0">{n_good}</div>
    <div class="ys-sum-sub">≥ 85% odds</div>
  </div>
  <div class="ys-sum-card" style="border-top:2px solid #F5A623">
    <div class="ys-sum-label">Uncertain</div>
    <div class="ys-sum-value" style="color:#F5A623">{n_risky}</div>
    <div class="ys-sum-sub">60–84% odds</div>
  </div>
  <div class="ys-sum-card" style="border-top:2px solid #EF4444">
    <div class="ys-sum-label">Unlikely / Drop</div>
    <div class="ys-sum-value" style="color:#EF4444">{n_drop}</div>
    <div class="ys-sum-sub">< 60% odds</div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_strategy_panel(results, student_year: int):
    """Full strategy panel rendered in the main area."""
    df_raw, meta = _load_strategy_data()

    # Banner
    st.markdown(f"""
<div class="ys-banner ys-scope">
  <div>
    <div class="ys-banner-title">🎯 Mileage Betting Strategy</div>
    <div class="ys-banner-sub">연세대학교 · Survival-curve engine · Year {student_year} personalised</div>
  </div>
  <div class="ys-banner-badge">{TOTAL_MILEAGE}PT BUDGET · MAX {MAX_BID}PT/COURSE</div>
</div>
""", unsafe_allow_html=True)

    render_strategy_summary(results)

    st.markdown('<div class="ys-section-label">Bid Recommendations</div>', unsafe_allow_html=True)

    for r in results:
        render_bid_card(r, student_year, df_raw, meta)

    with st.expander("Export strategy text (for chatbot / sharing)"):
        st.code(format_strategy_for_chat(results), language="markdown")


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY ENGINE RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_strategy_engine(
    priority_rankings: dict,
    selected_schedule: dict,
    student_year: int,
    target_conf: float = 0.90,
):
    ranked_list = []
    for code, rank in sorted(priority_rankings.items(), key=lambda x: x[1]):
        course_obj = selected_schedule.get(code, {})
        name = display_course_name(course_obj.get("course_name", code))
        ranked_list.append({"code": code, "name": name, "rank": rank})
    try:
        return get_strategy_for_ranked_list(
            ranked_list,
            {"year": student_year},
            raw_csv=RAW_CSV_PATH,
            json_path=JSON_PATH,
            target_confidence=target_conf,
        )
    except Exception as e:
        st.error(f"Strategy engine failed: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 4. INTAKE SCREEN  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

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
        gif_b64 = gif_to_base64("eagle2.gif")
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

            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            student_year_input = st.selectbox(
                "Your University Year (for mileage strategy)",
                [1, 2, 3, 4], index=1,
                help="Used by the mileage betting engine to predict competition levels.",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Generate Strategic Schedule Matching Preferences →",
                use_container_width=True,
            )

            if submitted:
                prefs = {
                    "language":    language,
                    "lecture_type":lecture_type,
                    "major_year":  major_year,
                    "major_years": [] if major_year == "Any" else [major_year],
                    "cat_req":     cat_req,
                    "cat_elec":    cat_elec,
                    "cat_basic":   cat_basic,
                    "max_credits": max_credits,
                    "focus_areas": [],
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


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN CHAT SCREEN  (unchanged structure, updated strategy rendering)
# ═════════════════════════════════════════════════════════════════════════════

else:
    selected_schedule = st.session_state.selected_schedule
    p                 = st.session_state.prefs
    student_year      = st.session_state.student_year

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        filter_col, time_col = st.columns([1, 1.4], gap="medium")

        # ── Filter column ────────────────────────────────────────────────────
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

            # ── Priority Ranking (after timetable confirmed) ─────────────────
            if st.session_state.timetable_confirmed and selected_schedule:
                st.markdown("### Priority Ranking")
                st.caption("AI suggested rankings below — adjust if needed.")
                course_count   = len(selected_schedule)
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
                        use_container_width=True, type="primary",
                    )

                    if submitted_rankings:
                        if len(set(ranking_values.values())) != course_count:
                            st.warning("Each course needs a unique priority number.")
                        else:
                            st.session_state.priority_rankings = ranking_values
                            st.success("Priority ranking saved.")

                    if gen_strategy_btn:
                        if len(set(ranking_values.values())) != course_count:
                            st.warning("Each course needs a unique priority number before generating the strategy.")
                        else:
                            st.session_state.priority_rankings = ranking_values
                            with st.spinner("Running mileage strategy engine…"):
                                results = run_strategy_engine(
                                    ranking_values, selected_schedule, student_year)
                            if results:
                                st.session_state.strategy_results   = results
                                st.session_state.strategy_generated = True
                                st.rerun()
            else:
                if st.session_state.timetable_confirmed:
                    st.success("✅ Timetable confirmed")
                    if st.session_state.strategy_generated:
                        st.info("🎯 Strategy generated — see main area below")

        # ── Timetable column ──────────────────────────────────────────────────
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

            # ── Confirm Timetable button ──────────────────────────────────────
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
                    llm_messages  = st.session_state.messages + [{"role":"user","content":confirmation_prompt}]
                    ranking_reply = call_llm(
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

                    # Bulletproof LLM output scrubber
                    clean_rankings = {}
                    valid_codes    = set(selected_schedule.keys())
                    for k, v in suggested_rankings.items():
                        if isinstance(v, str) and v.strip() in valid_codes:
                            course, rank_val = v.strip(), k
                        else:
                            course, rank_val = str(k).strip(), v
                        if course not in valid_codes: continue
                        extracted_rank = None
                        if isinstance(rank_val, (list, tuple)):
                            for item in rank_val:
                                try: extracted_rank = int(item); break
                                except (ValueError, TypeError): pass
                        else:
                            try:
                                extracted_rank = int(rank_val)
                            except (ValueError, TypeError):
                                match = re.search(r'\d+', str(rank_val))
                                if match: extracted_rank = int(match.group())
                        if extracted_rank is not None:
                            clean_rankings[course] = extracted_rank
                    suggested_rankings = clean_rankings

                    used_ranks = set(suggested_rankings.values())
                    next_rank  = 1
                    for c_code in selected_schedule.keys():
                        if c_code not in suggested_rankings:
                            while next_rank in used_ranks: next_rank += 1
                            suggested_rankings[c_code] = next_rank
                            used_ranks.add(next_rank)

                    st.session_state.messages.append({"role":"user","content":"Confirmed timetable."})
                    display_text = re.sub(r'\{[^{}]+\}', '', ranking_reply).strip()
                    st.session_state.messages.append({"role":"assistant","content":format_assistant_reply(display_text)})
                    st.session_state.timetable_confirmed   = True
                    st.session_state.ai_suggested_rankings = suggested_rankings
                    st.rerun()

    # ── MAIN AREA ─────────────────────────────────────────────────────────────

    # Strategy panel shown above chat once generated
    if st.session_state.strategy_generated and st.session_state.strategy_results:
        st.markdown("---")
        render_strategy_panel(st.session_state.strategy_results, student_year)
        st.markdown("---")

        with st.expander("⚙️ Regenerate with custom confidence target"):
            api_conf = st.slider("Target Win Probability (%)", 70, 99, 90, key="regen_conf")
            if st.button("Regenerate Strategy", key="regen_btn"):
                with st.spinner("Re-calculating survival curves…"):
                    new_results = run_strategy_engine(
                        st.session_state.priority_rankings,
                        selected_schedule,
                        student_year,
                        target_conf=api_conf / 100.0,
                    )
                    if new_results:
                        st.session_state.strategy_results = new_results
                        st.rerun()

    # ── Chat interface ────────────────────────────────────────────────────────
    st.title("NightHawk AI - Yonsei Course Assistant 🎓")

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

    if prompt := st.chat_input("Ask to add, remove, optimize, or compare courses…"):
        st.session_state.messages.append({"role":"user","content":prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing targeted data chunks…"):
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
                    p, selected_courses,
                    st.session_state.filtered_courses,
                    st.session_state.selected_schedule,
                )
                reply = call_llm(system_prompt, st.session_state.messages)
                reply = validate_grounding(reply, selected_courses, st.session_state.filtered_courses)

                actions = extract_schedule_actions(reply, user_prompt=prompt)
                action_results = []
                if actions:
                    updated_schedule, action_results = apply_schedule_actions(
                        actions,
                        st.session_state.selected_schedule,
                        st.session_state.filtered_courses,
                    )
                    st.session_state.selected_schedule   = updated_schedule
                    st.session_state.schedule_action_log.extend(action_results)
                    st.session_state.timetable_confirmed = False
                    st.session_state.priority_rankings   = {}
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