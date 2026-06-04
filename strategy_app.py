"""
strategy_app.py
---------------
Standalone Streamlit UI for Part 2: Mileage Betting Strategy.

Run independently: streamlit run strategy_app.py

Later integration: Part 1 chatbot calls get_strategy_for_ranked_list()
directly from optimizer.py — no Streamlit needed.

The UI lets you:
1. Load courses from the shared JSON
2. Select and rank courses manually (simulating what Part 1 chatbot outputs)
3. Enter your mileage budget
4. See the recommended strategy with explanations
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mileage Strategy Engine",
    page_icon="🎯",
    layout="wide"
)

# ── Imports ───────────────────────────────────────────────────────────────────
from feature_extractor import load_features, flatten_json
from synthetic_data_generator import FEATURE_COLS
from model import (
    load_model,
    predict_threshold,
    explain_threshold,
    feature_importance_df,
)
from optimizer import CourseInput, allocate_bids, strategy_summary

# ── Constants ─────────────────────────────────────────────────────────────────
JSON_PATH = "segmented_cs_courses.json"

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_course_features(json_path):
    return load_features(json_path)

@st.cache_data
def get_flat_courses(json_path):
    return flatten_json(json_path)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-safe     { color: #22c55e; font-weight: 700; }
    .risk-moderate { color: #f59e0b; font-weight: 700; }
    .risk-risky    { color: #ef4444; font-weight: 700; }
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
    .metric-box {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Main App ──────────────────────────────────────────────────────────────────

st.title("🎯 Mileage Strategy Engine")
st.caption("Part 2 of the Yonsei Course Registration AI System — standalone mode")

# Sidebar: student profile
with st.sidebar:
    st.header("👤 Student Profile")
    student_year = st.selectbox("Your year", [1, 2, 3, 4], index=1)
    total_mileage = st.number_input(
        "Available mileage points", min_value=10, max_value=500, value=150, step=10
    )
    safety_margin = st.slider(
        "Safety buffer (%)",
        min_value=0, max_value=40, value=15,
        help="Extra % above prediction to bid — higher = safer but uses more points"
    ) / 100.0

    st.divider()
    st.markdown("### 📁 Data source")
    json_path = st.text_input("JSON path", value=JSON_PATH)
    if not Path(json_path).exists():
        st.error(f"File not found: {json_path}")
        st.stop()

    st.divider()
    if st.button("🔄 Retrain model", use_container_width=True):
        from model import train
        with st.spinner("Training..."):
            train(json_path=json_path, save=True)
        st.cache_resource.clear()
        st.success("Model retrained!")


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    model, explainer = get_model()
    df_features = get_course_features(json_path)
    flat_courses = get_flat_courses(json_path)
except Exception as e:
    st.error(f"Failed to load model or data: {e}")
    st.info("Try clicking 'Retrain model' in the sidebar.")
    st.stop()


# ── Tab layout ────────────────────────────────────────────────────────────────
tab_manual, tab_api, tab_insights = st.tabs([
    "🖱️ Manual Course Selection",
    "🔗 API / Integration Mode",
    "📊 Model Insights"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Manual selection — simulates what Part 1 chatbot outputs
# ═════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Select and rank courses")
    st.caption(
        "In the full system, this list comes automatically from the Part 1 chatbot. "
        "Here you can test Part 2 independently."
    )

    # Build display names
    course_options = {}
    for code, c in flat_courses.items():
        display = f"{code} — {c.get('name', code)} ({c.get('professor', 'Unknown')})"
        course_options[display] = code

    selected_display = st.multiselect(
        "Choose courses you want to register for",
        options=list(course_options.keys()),
        max_selections=10,
        help="Select 3-7 courses for a realistic strategy"
    )

    if not selected_display:
        st.info("Select at least 2 courses to see a strategy.")

    selected_codes = [course_options[d] for d in selected_display]

    # Ranking UI
    st.markdown("#### Rank your courses (drag to reorder)")
    st.caption("Rank 1 = most important to you. The model uses this as a priority signal.")

    rank_data = []
    for i, (disp, code) in enumerate(zip(selected_display, selected_codes)):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text(f"  {disp}")
        with col2:
            rank = st.number_input(
                "Rank", min_value=1, max_value=len(selected_codes),
                value=i + 1, key=f"rank_{code}", label_visibility="collapsed"
            )
        rank_data.append((code, rank))

    st.divider()

    if st.button("🚀 Generate Strategy", type="primary", use_container_width=True):

        with st.spinner("Analysing competition and allocating bids..."):

            # Build CourseInput list
            course_inputs = []
            for code, rank in rank_data:
                if code not in df_features.index:
                    st.warning(f"Course {code} not in features — skipping")
                    continue

                feat_row = df_features.loc[code].to_dict()
                feat_row["student_year"] = float(student_year)
                feat_row["student_mileage"] = float(total_mileage)
                feat_row["num_courses_wanted"] = float(len(rank_data))
                feat_row["rank_in_list"] = float(rank)
                feat_row["priority_ratio"] = (len(rank_data) - rank + 1) / len(rank_data)
                feat_row["budget_ratio"] = total_mileage / 500.0

                pred_bid, shap_breakdown = predict_threshold(
                    model, explainer, feat_row
                )

                c_info = flat_courses.get(code, {})
                course_inputs.append(CourseInput(
                    code=code,
                    name=c_info.get("name", code),
                    rank=rank,
                    predicted_threshold=pred_bid,
                    is_major_req=(c_info.get("category") == "major_requirement"),
                    shap_breakdown=shap_breakdown,
                ))

            results = allocate_bids(
                course_inputs,
                total_mileage=total_mileage,
                safety_margin=safety_margin,
            )

        # ── Display results ────────────────────────────────────────────────
        total_used = sum(r.recommended_bid for r in results)
        remaining = total_mileage - total_used

        col1, col2, col3 = st.columns(3)
        col1.metric("Total budget", f"{total_mileage} pts")
        col2.metric("Points allocated", f"{total_used} pts")
        col3.metric("Points remaining", f"{remaining} pts",
                    delta=f"{remaining} unspent",
                    delta_color="normal")

        st.divider()
        st.subheader("📋 Recommended Strategy")

        RISK_COLORS = {"Safe": "safe", "Moderate": "moderate", "Risky": "risky"}
        RISK_EMOJI  = {"Safe": "🟢", "Moderate": "🟡", "Risky": "🔴"}

        for r in results:
            css_class = RISK_COLORS.get(r.risk_level, "")
            with st.container():
                st.markdown(
                    f'<div class="bid-card {css_class}">',
                    unsafe_allow_html=True
                )
                col_a, col_b, col_c, col_d = st.columns([1, 5, 2, 2])
                col_a.markdown(f"**#{r.rank}**")
                col_b.markdown(f"**{r.name}**  \n`{r.code}`")
                col_c.metric("Bid", f"{r.recommended_bid} pts",
                              delta=f"predicted threshold: ~{int(r.predicted_threshold)} pts",
                              delta_color="off")
                col_d.metric(
                    "Risk",
                    f"{RISK_EMOJI[r.risk_level]} {r.risk_level}",
                    delta=f"{r.confidence_pct:.0f}% confidence",
                    delta_color="off"
                )

                if r.note:
                    st.caption(r.note)

                # SHAP explanation expandable
                with st.expander("Why this bid? (AI reasoning)"):
                    explanation = explain_threshold(
                        r.predicted_threshold, r.shap_breakdown, r.name
                    )
                    st.markdown(explanation)

                    # Mini bar chart of top SHAP drivers
                    shap_df = pd.DataFrame([
                        {"factor": k.replace("_", " "), "impact": v}
                        for k, v in r.shap_breakdown.items()
                        if abs(v) > 0.5
                    ]).sort_values("impact", key=abs, ascending=False).head(8)

                    if not shap_df.empty:
                        shap_df["color"] = shap_df["impact"].apply(
                            lambda x: "#ef4444" if x > 0 else "#22c55e"
                        )
                        st.bar_chart(
                            shap_df.set_index("factor")["impact"],
                            use_container_width=True, height=200
                        )
                        st.caption("Red = increases competition (bid higher) | "
                                   "Green = decreases competition (bid lower)")

                st.markdown("</div>", unsafe_allow_html=True)

        # Raw strategy text (for copy-pasting into Part 1)
        with st.expander("📄 Raw strategy output (for integration)"):
            st.code(strategy_summary(results, total_mileage))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: API / Integration Mode — how Part 1 will call Part 2
# ═════════════════════════════════════════════════════════════════════════════
with tab_api:
    st.subheader("🔗 How Part 1 integrates with Part 2")
    st.markdown("""
    When you're ready to connect the chatbot, Part 1 calls this function directly:
    ```python
    from strategy_engine import get_strategy_for_ranked_list
    
    # Part 1 chatbot produces this output:
    ranked_list = [
        {"code": "CAS3120-02", "name": "Machine Learning", "rank": 1},
        {"code": "CAS4160-01-00", "name": "Reinforcement Learning", "rank": 2},
        {"code": "CAS3301-01", "name": "NLP", "rank": 3},
    ]
    student_profile = {
        "year": 3,
        "mileage": 150,
    }
    
    strategy = get_strategy_for_ranked_list(ranked_list, student_profile)
    # Returns list of BidResult objects — render however you want
    ```
    """)

    st.markdown("---")
    st.subheader("Test the API directly")
    json_input = st.text_area(
        "Paste a ranked list JSON",
        value=json.dumps([
            {"code": "CAS3120-02", "name": "Machine Learning", "rank": 1},
            {"code": "CAS4160-01-00", "name": "Reinforcement Learning", "rank": 2},
            {"code": "CAS3301-01", "name": "Natural Language Processing", "rank": 3},
        ], indent=2),
        height=200
    )
    api_year = st.number_input("Student year", 1, 4, 3)
    api_mileage = st.number_input("Student mileage", 10, 500, 150)

    if st.button("Run strategy engine", type="primary"):
        try:
            ranked_list = json.loads(json_input)
            from strategy_engine import get_strategy_for_ranked_list
            results = get_strategy_for_ranked_list(
                ranked_list,
                {"year": api_year, "mileage": api_mileage},
                json_path=json_path,
                safety_margin=safety_margin,
            )
            for r in results:
                st.markdown(
                    f"**#{r.rank} {r.name}** — "
                    f"Bid: `{r.recommended_bid} pts` | "
                    f"Risk: {r.risk_level} ({r.confidence_pct:.0f}%)"
                )
                if r.note:
                    st.caption(r.note)
        except Exception as e:
            st.error(f"Error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Insights — what the model discovered
# ═════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.subheader("📊 What the model discovered")
    st.caption(
        "Feature importance shows which signals the model found most predictive "
        "of competition — NOT what you hardcoded."
    )

    from model import FEATURE_LABELS
    fi_df = feature_importance_df(model)

    st.bar_chart(
        fi_df.set_index("label")["importance"].head(12),
        use_container_width=True,
        height=400
    )

    st.dataframe(
        fi_df[["label", "importance"]].rename(
            columns={"label": "Factor", "importance": "Importance Score"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""
    ---
    **How to read this:**
    - Higher importance = the model found this feature more useful for predicting competition
    - The *direction* of each feature's effect varies per course — SHAP values (shown per course above) reveal that
    - If a feature you expected to matter scores low, it means in your data, 
      it doesn't consistently correlate with competition — that's a finding for your report
    """)
