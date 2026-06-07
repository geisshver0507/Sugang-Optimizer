"""
strategy_app.py
---------------
Standalone Streamlit UI for Part 2: Mileage Betting Strategy.
Uses the survival-curve based system (no training required).

Run: streamlit run strategy_app.py
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Mileage Strategy Engine",
    page_icon="🎯",
    layout="wide"
)

from survival_model import load_applicant_data, load_course_meta, build_curve, confidence_label
from allocator import build_strategy, TOTAL_MILEAGE, MAX_BID
from strategy_engine_v2 import get_strategy_for_ranked_list, format_strategy_for_chat

JSON_PATH = "segmented_cs_courses.json"
RAW_CSV   = "mileage_history_all.csv"

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
    .bid-card.secured  { border-left-color: #22c55e; }
    .bid-card.likely   { border-left-color: #f59e0b; }
    .bid-card.risky    { border-left-color: #f97316; }
    .bid-card.stretch  { border-left-color: #f97316; }
    .bid-card.drop     { border-left-color: #ef4444; }
    .bid-card.unfunded { border-left-color: #ef4444; }
    .rule-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def get_course_list():
    """All courses from JSON + their most recent professor from raw CSV."""
    import json as _json
    with open(JSON_PATH) as f:
        data = _json.load(f)

    courses = {}
    for _, yr_data in data.items():
        for _, cat_data in yr_data.items():
            for code, course in cat_data.items():
                if code not in courses:
                    m = course.get("metadata", {})
                    courses[code] = {"name": m.get("name", code), "professor": ""}

    # Overlay most recent professor from raw CSV
    if Path(RAW_CSV).exists():
        df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
        df = df.dropna(subset=["professor"])
        latest = (df.sort_values(["year", "semester"], ascending=False)
                    .groupby("course_code")["professor"].first())
        for code, prof in latest.items():
            if code in courses:
                courses[code]["professor"] = prof

    return courses

@st.cache_data
def get_survival_data():
    return load_applicant_data(RAW_CSV)

@st.cache_data
def get_meta():
    return load_course_meta(JSON_PATH, RAW_CSV)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🎯 Mileage Strategy Engine")
st.markdown(
    f'<div class="rule-box">'
    f'<b>📋 Yonsei Mileage Rules (Fixed)</b> &nbsp;|&nbsp;'
    f'Total budget: <b>{TOTAL_MILEAGE} pts</b> per semester &nbsp;|&nbsp;'
    f'Maximum per course: <b>{MAX_BID} pts</b>'
    f'</div>',
    unsafe_allow_html=True
)

# ── Check data ─────────────────────────────────────────────────────────────────

if not Path(RAW_CSV).exists():
    st.error(f"**{RAW_CSV} not found.**")
    st.info("Run `merge_scraped_csvs.py` in your mileage_csvs folder first, "
            "then copy the output here.")
    st.stop()

courses = get_course_list()
df_raw  = get_survival_data()
meta    = get_meta()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("👤 Student Profile")

    student_year = st.selectbox(
        "Your year", [1, 2, 3, 4], index=2,
        help="Affects year-quota dynamics (some courses favour certain years)"
    )

    st.divider()
    st.markdown("### 💰 Budget")
    st.metric("Total",       f"{TOTAL_MILEAGE} pts")
    st.metric("Max/course",  f"{MAX_BID} pts")
    st.caption("Fixed Yonsei rules — not adjustable.")

    st.divider()
    st.markdown("### 📊 Data loaded")
    years = sorted(df_raw["year"].unique())
    sems  = df_raw.groupby(["year","semester"]).size()
    for (yr, sem), cnt in sems.items():
        st.caption(f"{yr}-{sem}: {cnt:,} applicant rows")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_select, tab_api, tab_explore = st.tabs([
    "🖱️ Build Strategy",
    "🔗 Integration Mode",
    "📊 Explore Courses"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Build Strategy
# ══════════════════════════════════════════════════════════════════════════════

with tab_select:
    st.subheader("Select and rank your courses")
    st.caption("In the full system this list comes from the Part 1 chatbot. "
               "Use this tab to test Part 2 independently.")

    # Build dropdown labels
    course_options = {}
    for code, c in courses.items():
        prof  = c.get("professor", "")
        name  = c.get("name", code)
        label = f"{code} — {name}"
        if prof:
            label += f" ({prof})"
        course_options[label] = code

    selected_labels = st.multiselect(
        "Choose courses you want to register for",
        options=list(course_options.keys()),
        max_selections=10,
        help="Select 2–7 courses for a realistic strategy"
    )

    if not selected_labels:
        st.info("Select at least 2 courses above to see a strategy.")
        st.stop()

    selected_codes = [course_options[l] for l in selected_labels]

    # Rank assignment
    st.markdown("#### Set priority rank")
    st.caption("Rank 1 = most important to you. Lower-ranked courses get funded last.")

    rank_data = []
    for i, (label, code) in enumerate(zip(selected_labels, selected_codes)):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.text(label)
        with c2:
            rank = st.number_input(
                "Rank", min_value=1, max_value=len(selected_codes),
                value=i + 1, key=f"rank_{code}",
                label_visibility="collapsed"
            )
        rank_data.append({"code": code, "name": courses[code]["name"],
                           "rank": rank, "professor": courses[code]["professor"]})

    st.divider()

    if st.button("🚀 Generate Strategy", type="primary", use_container_width=True):

        with st.spinner("Building survival curves and allocating bids..."):
            results = build_strategy(
                rank_data,
                student_year=student_year,
                target_conf=0.90,
                raw_csv=RAW_CSV,
                json_path=JSON_PATH,
            )

        # ── Summary metrics ────────────────────────────────────────────────
        total_used = sum(r.bid for r in results)
        n_good     = sum(1 for r in results if r.status in ("Secured", "Likely"))
        n_risky    = sum(1 for r in results if r.status == "Risky")
        n_drop     = sum(1 for r in results if r.status in ("Drop", "Unfunded"))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Budget used",   f"{total_used}/{TOTAL_MILEAGE} pts")
        c2.metric("Good+ odds",    f"{n_good}/{len(results)}")
        c3.metric("Risky",         f"{n_risky}",
                  delta="needs caution" if n_risky else "none",
                  delta_color="inverse" if n_risky else "normal")
        c4.metric("Drop/Unfunded", f"{n_drop}",
                  delta="reprioritize" if n_drop else "none",
                  delta_color="inverse" if n_drop else "normal")

        st.divider()
        st.subheader("📋 Recommended Bids")

        STATUS_CSS   = {"Secured": "secured", "Likely": "likely",
                        "Risky": "risky", "Stretch": "stretch",
                        "Drop": "drop", "Unfunded": "unfunded"}
        STATUS_EMOJI = {"Secured": "🟢", "Likely": "🟡",
                        "Risky": "🟠", "Stretch": "🟠",
                        "Drop": "⛔", "Unfunded": "⛔"}
        CONF_EMOJI   = {"High": "📊", "Medium": "🔍", "Low": "❓"}

        for r in results:
            css = STATUS_CSS.get(r.status, "")
            em  = STATUS_EMOJI.get(r.status, "⚪")
            ce  = CONF_EMOJI.get(r.confidence, "")

            st.markdown(f'<div class="bid-card {css}">', unsafe_allow_html=True)

            ca, cb, cc, cd, ce_col = st.columns([1, 4, 2, 2, 2])
            ca.markdown(f"**#{r.rank}**")
            cb.markdown(f"**{r.name}**  \n`{r.code}` · {r.professor}")
            cc.metric("Bid",         f"{r.bid} pts",
                      delta=f"min safe: {r.min_safe_bid} pts", delta_color="off")
            cd.metric("Calibrated odds", f"{r.win_prob:.0%}",
                      delta=r.competition, delta_color="off")
            ce_col.metric("Status",  f"{em} {r.status}",
                          delta=f"{ce} {r.confidence} ({r.confidence_pct:.0f}%)",
                          delta_color="off")

            for line in r.note.split("\n"):
                if line.strip():
                    st.caption(line.strip())

            # Survival curve mini-chart
            with st.expander("📈 Bid vs win probability curve"):
                curve = build_curve(df_raw, r.code, r.professor,
                                    student_year=student_year, meta=meta)
                if len(curve.bids) > 0:
                    chart_df = pd.DataFrame({
                        "bid":     curve.bids,
                        "P(win)":  curve.p_enroll,
                    }).set_index("bid")
                    st.line_chart(chart_df, height=200)
                    st.caption(
                        f"Data source: **{curve.source}** · "
                        f"Years: {curve.data_years} · "
                        f"Effective sample: {curve.n_effective:.0f} weighted rows"
                    )
                    st.caption(
                        f"The recommended bid of **{r.bid}pt** sits at "
                        f"**{r.win_prob:.0%} calibrated odds**. "
                        f"Min safe bid (90% target): **{r.min_safe_bid}pt**."
                    )
                else:
                    st.caption("No curve data — demand proxy used.")

            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📄 Copy strategy text (for chatbot)"):
            st.code(format_strategy_for_chat(results))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Integration Mode
# ══════════════════════════════════════════════════════════════════════════════

with tab_api:
    st.subheader("🔗 How Part 1 chatbot calls Part 2")
    st.markdown(f"""
Part 1 passes the student's ranked list and year. Mileage is **not** passed — always {TOTAL_MILEAGE}pts.

```python
from strategy_engine_v2 import get_strategy_for_ranked_list, format_strategy_for_chat

ranked_list = [
    {{"code": "CAS3120-02", "name": "Machine Learning",       "rank": 1}},
    {{"code": "CAS4160-01", "name": "Reinforcement Learning", "rank": 2}},
    {{"code": "CAS3205-01", "name": "Computer Graphics",      "rank": 3}},
]

results = get_strategy_for_ranked_list(ranked_list, {{"year": 3}})
print(format_strategy_for_chat(results))
```

Professor is **optional** — auto-resolved from the most recent scraped data if not supplied.
""")

    st.divider()
    st.subheader("Test the integration")

    json_input = st.text_area(
        "Paste ranked list JSON",
        value=json.dumps([
            {"code": "CAS3101-02", "name": "Operating System",         "rank": 1},
            {"code": "CAS3116-01", "name": "Computer Vision",          "rank": 2},
            {"code": "CAS4127-01", "name": "Human-AI Interaction",     "rank": 3},
        ], indent=2),
        height=220
    )
    api_year = st.number_input("Student year", 1, 4, 3)
    api_conf = st.slider("Target confidence", 80, 97, 90) / 100.0

    if st.button("Run strategy engine", type="primary"):
        try:
            ranked = json.loads(json_input)
            results = get_strategy_for_ranked_list(
                ranked, {"year": api_year},
                target_confidence=api_conf
            )
            total = sum(r.bid for r in results)

            STATUS_EMOJI = {"Secured":"🟢","Likely":"🟡","Risky":"🟠","Stretch":"🟠","Drop":"⛔","Unfunded":"⛔"}
            st.success(f"Total: {total}/{TOTAL_MILEAGE} pts used")

            for r in results:
                em = STATUS_EMOJI.get(r.status, "⚪")
                st.markdown(
                    f"**#{r.rank} {r.name}** ({r.professor}) — "
                    f"Bid: `{r.bid}pt` | {em} {r.status} | "
                    f"calibrated odds: {r.win_prob:.0%} | "
                    f"safe bid: {r.min_safe_bid}pt [{r.competition}]"
                )
                for line in r.note.split("\n"):
                    if line.strip():
                        st.caption(line.strip())

            st.divider()
            st.code(format_strategy_for_chat(results), language="markdown")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Explore Courses
# ══════════════════════════════════════════════════════════════════════════════

with tab_explore:
    st.subheader("📊 Explore competition curves by course")
    st.caption("See exactly what the model knows about each course — "
               "the raw survival curve from historical data.")

    explore_label = st.selectbox(
        "Select a course",
        options=list(course_options.keys())
    )
    explore_code = course_options[explore_label]
    explore_prof = courses[explore_code]["professor"]
    explore_year = st.selectbox("Your year (for personalisation)", [1,2,3,4], index=2)

    curve = build_curve(df_raw, explore_code, explore_prof,
                        student_year=explore_year, meta=meta)
    conf, conf_pct = confidence_label(curve)

    col1, col2, col3 = st.columns(3)
    col1.metric("Data source",     curve.source)
    col2.metric("Confidence",      f"{conf} ({conf_pct:.0f}%)")
    col3.metric("Years of data",   str(curve.data_years))

    if len(curve.bids) > 0:
        chart_df = pd.DataFrame({
            "bid":    curve.bids,
            "P(win)": curve.p_enroll,
        }).set_index("bid")
        st.line_chart(chart_df, height=300)

        st.markdown("**Bid thresholds:**")
        thresh_cols = st.columns(4)
        for i, target in enumerate([0.70, 0.80, 0.90, 0.97]):
            msb = curve.min_bid_for(target)
            thresh_cols[i].metric(f"{target:.0%} confidence", f"{msb} pt")

        st.markdown("**Max reachable win rate:** "
                    f"`{curve.reachable_max():.0%}` "
                    f"(bidding more than `{curve.min_bid_for(curve.reachable_max() - 0.01)}pt` adds nothing)")

        # Raw distribution
        with st.expander("Raw bid distribution from historical data"):
            sub = df_raw[df_raw["course_code"] == explore_code].copy()
            sub["enrolled_bool"] = (sub["enrolled"].astype(str).str.upper() == "Y").astype(int)
            if len(sub) > 0:
                dist = sub.groupby(["year","semester","professor"]).agg(
                    applicants=("mileage_bid","count"),
                    enrolled=("enrolled_bool","sum"),
                    avg_bid=("mileage_bid","mean"),
                    min_win_bid=("mileage_bid", lambda x: x[sub.loc[x.index,"enrolled_bool"]==1].min() if (sub.loc[x.index,"enrolled_bool"]==1).any() else None),
                ).reset_index()
                st.dataframe(dist, use_container_width=True, hide_index=True)
    else:
        st.warning("No historical data for this course. "
                   "Predictions use ETA demand proxy only.")
