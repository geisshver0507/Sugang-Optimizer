"""
strategy_app.py
---------------
Standalone Streamlit UI for Part 2: Mileage Betting Strategy.

Yonsei fixed rules (no user input needed):
  - Total budget: 72 pts per semester
  - Max per course: 36 pts

Run: streamlit run strategy_app.py
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Mileage Strategy Engine",
    page_icon="🎯",
    layout="wide"
)

from feature_extractor import load_features, flatten_json
from model import load_model, predict_threshold, explain_threshold, FEATURE_LABELS
from optimizer import (
    CourseInput, allocate_bids, strategy_summary,
    TOTAL_MILEAGE, MAX_BID_PER_COURSE
)
from strategy_engine_model import get_strategy_for_ranked_list, format_strategy_for_chat

ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = ROOT / "database"
JSON_PATH = str(DATABASE_DIR / "segmented_cs_courses.json")

# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_features(json_path):
    return load_features(json_path)

@st.cache_data
def get_flat(json_path):
    return flatten_json(json_path)

# ── Styling ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
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
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────

try:
    model, explainer = get_model()
    df_features      = get_features(JSON_PATH)
    flat_courses     = get_flat(JSON_PATH)
except Exception as e:
    st.error(f"Failed to load model or data: {e}")
    st.info("Run: python3 model.py --train")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🎯 Mileage Strategy Engine")

# Fixed rules banner — always visible
st.markdown(
    f"""
    <div class="rule-box">
        <b>📋 Yonsei Mileage Rules (Fixed)</b> &nbsp;|&nbsp;
        Total budget: <b>{TOTAL_MILEAGE} pts</b> per semester &nbsp;|&nbsp;
        Maximum per course: <b>{MAX_BID_PER_COURSE} pts</b>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("👤 Student Profile")

    student_year = st.selectbox(
        "Your year", [1, 2, 3, 4],
        index=1,
        help="Affects which courses you're eligible for and competition patterns"
    )

    st.divider()

    # Fixed budget display — not an input anymore
    st.markdown("### 💰 Mileage Budget")
    st.metric("Total available", f"{TOTAL_MILEAGE} pts")
    st.metric("Max per course",  f"{MAX_BID_PER_COURSE} pts")
    st.caption("These are fixed Yonsei university rules.")

    st.divider()

    safety_margin = st.slider(
        "Safety buffer (%)",
        min_value=0, max_value=30, value=15,
        help="Extra % above predicted threshold. Higher = safer but spreads budget thinner."
    ) / 100.0

    st.divider()

    if st.button("🔄 Retrain model", use_container_width=True):
        from model import train
        with st.spinner("Training..."):
            train(save=True)
        st.cache_resource.clear()
        st.success("Model retrained!")
        st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_select, tab_api, tab_insights = st.tabs([
    "🖱️ Select Courses",
    "🔗 Integration Mode",
    "📊 Model Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Course Selection
# ══════════════════════════════════════════════════════════════════════════════

with tab_select:
    st.subheader("Select and rank your courses")
    st.caption(
        "In the full system this list comes from the Part 1 chatbot. "
        "Use this tab to test Part 2 independently."
    )

    # Build display options
    course_options = {}
    for code, c in flat_courses.items():
        prof = c.get("professor") or c.get("metadata", {}).get("professor", "")
        name = c.get("name", code)
        label = f"{code} — {name}"
        if prof:
            label += f" (Prof. {prof})"
        course_options[label] = code

    selected_labels = st.multiselect(
        "Choose courses you want to register for",
        options=list(course_options.keys()),
        max_selections=10,
        help="Select 2–7 courses for a realistic strategy"
    )

    if not selected_labels:
        st.info("Select at least 2 courses to see a strategy.")
        st.stop()

    selected_codes = [course_options[l] for l in selected_labels]

    # Rank assignment
    st.markdown("#### Set priority rank for each course")
    st.caption("Rank 1 = most important to you.")

    rank_data = []
    for label, code in zip(selected_labels, selected_codes):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.text(label)
        with c2:
            rank = st.number_input(
                "Rank", min_value=1, max_value=len(selected_codes),
                value=len(rank_data) + 1,
                key=f"rank_{code}",
                label_visibility="collapsed"
            )
        rank_data.append((code, rank))

    st.divider()

    if st.button("🚀 Generate Strategy", type="primary", use_container_width=True):

        with st.spinner("Predicting competition and allocating bids..."):

            course_inputs = []
            for code, rank in rank_data:
                if code not in df_features.index:
                    st.warning(f"Course {code} not in feature set — skipping")
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

                course_inputs.append(CourseInput(
                    code                = code,
                    name                = c_info.get("name", code),
                    rank                = rank,
                    predicted_threshold = threshold,
                    is_major_req        = (c_info.get("category") == "major_requirement"),
                    shap_breakdown      = shap,
                ))

            results = allocate_bids(
                course_inputs,
                total_mileage  = TOTAL_MILEAGE,
                max_per_course = MAX_BID_PER_COURSE,
                safety_margin  = safety_margin,
            )

        # ── Summary metrics ────────────────────────────────────────────────
        total_used  = sum(r.recommended_bid for r in results)
        remaining   = TOTAL_MILEAGE - total_used

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Budget",       f"{TOTAL_MILEAGE} pts")
        c2.metric("Used",         f"{total_used} pts")
        c3.metric("Remaining",    f"{remaining} pts")
        c4.metric("Max/course",   f"{MAX_BID_PER_COURSE} pts")

        st.divider()
        st.subheader("📋 Recommended Bids")

        RISK_CSS   = {"Safe": "safe", "Moderate": "moderate", "Risky": "risky"}
        RISK_EMOJI = {"Safe": "🟢",   "Moderate": "🟡",       "Risky": "🔴"}

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
                delta_color="off"
            )
            cd.metric(
                "Risk",
                f"{em} {r.risk_level}",
                delta=f"{r.confidence_pct:.0f}% confidence",
                delta_color="off"
            )

            if r.note:
                st.caption(r.note)

            with st.expander("Why this bid?"):
                st.markdown(explain_threshold(
                    r.recommended_bid, r.shap_breakdown, r.name
                ))

                shap_df = pd.DataFrame([
                    {"factor": FEATURE_LABELS.get(k, k), "impact": float(v)}
                    for k, v in r.shap_breakdown.items()
                    if isinstance(v, (int, float)) and abs(v) > 0.3
                ]).sort_values("impact", key=abs, ascending=False).head(8)

                if not shap_df.empty:
                    st.bar_chart(
                        shap_df.set_index("factor")["impact"],
                        height=200
                    )
                    st.caption(
                        "Positive = increases competition (bid more needed) | "
                        "Negative = decreases competition"
                    )

            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📄 Raw strategy text"):
            st.code(strategy_summary(results, TOTAL_MILEAGE))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Integration Mode
# ══════════════════════════════════════════════════════════════════════════════

with tab_api:
    st.subheader("🔗 How Part 1 chatbot calls Part 2")
    st.markdown(f"""
    Part 1 passes the student's ranked list and year.
    Mileage is **not** passed — it's always {TOTAL_MILEAGE}pts.

    ```python
    from strategy_engine import get_strategy_for_ranked_list, format_strategy_for_chat

    ranked_list = [
        {{"code": "CAS3120-02",    "name": "Machine Learning",       "rank": 1}},
        {{"code": "CAS4160-01-00", "name": "Reinforcement Learning", "rank": 2}},
        {{"code": "CAS3205-01",    "name": "Computer Graphics",      "rank": 3}},
    ]

    # No "mileage" key — it's always {TOTAL_MILEAGE}pts
    strategy = get_strategy_for_ranked_list(ranked_list, {{"year": 3}})
    print(format_strategy_for_chat(strategy))
    ```
    """)

    st.divider()
    st.subheader("Test the API")

    json_input = st.text_area(
        "Paste ranked list JSON",
        value=json.dumps([
            {"code": "CAS3120-02",    "name": "Machine Learning",       "rank": 1},
            {"code": "CAS4160-01-00", "name": "Reinforcement Learning", "rank": 2},
            {"code": "CAS3205-01",    "name": "Computer Graphics",      "rank": 3},
        ], indent=2),
        height=200
    )
    api_year = st.number_input("Student year", 1, 4, 3)

    if st.button("Run strategy engine", type="primary"):
        try:
            ranked = json.loads(json_input)
            results = get_strategy_for_ranked_list(ranked, {"year": api_year})
            total   = sum(r.recommended_bid for r in results)

            st.success(f"Total: {total}/{TOTAL_MILEAGE} pts | "
                       f"Max per course: {MAX_BID_PER_COURSE} pts")

            for r in results:
                em = {"Safe":"🟢","Moderate":"🟡","Risky":"🔴"}.get(r.risk_level,"⚪")
                st.markdown(
                    f"**#{r.rank} {r.name}** — Bid: `{r.recommended_bid} pts` | "
                    f"{em} {r.risk_level} ({r.confidence_pct:.0f}%) | "
                    f"est. threshold: ~{int(r.predicted_threshold)} pts"
                )
                if r.note:
                    st.caption(r.note)
        except Exception as e:
            st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Insights
# ══════════════════════════════════════════════════════════════════════════════

with tab_insights:
    st.subheader("📊 What the model discovered")

    from model import feature_importance_df
    fi = feature_importance_df(model)

    st.bar_chart(
        fi.set_index("label")["importance"].head(12),
        height=400
    )

    st.dataframe(
        fi[["label", "importance"]].rename(
            columns={"label": "Factor", "importance": "Importance Score"}
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(f"""
    ---
    **Data summary**
    - Training rows: check `training_final.csv`
    - Yonsei budget: **{TOTAL_MILEAGE} pts** per semester (fixed)
    - Per-course cap: **{MAX_BID_PER_COURSE} pts** (fixed)
    - Most courses have low competition (threshold ≈ 1pt)
    - Model quality improves as more semesters are scraped
    """)


