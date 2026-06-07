import streamlit as st
import pandas as pd
import os

# --- STEP 1: DYNAMIC CSV LOADER ---
@st.cache_data
def load_courses_from_csv():
    csv_path = "course_summary_all.csv"
    
    if os.path.exists(csv_path):
        try:
            # Read CSV file
            df = pd.read_csv(csv_path, encoding="utf-8")
            
            # Drop rows with missing crucial information
            df = df.dropna(subset=["course_code_portal", "course_name", "professor"])
            
            # Create a unique list of specific course offerings (code + professor combination)
            unique_offerings = df[["course_code_portal", "course_name", "professor", "lecture_time"]].drop_duplicates()
            
            course_dict = {}
            for _, row in unique_offerings.iterrows():
                # Create a uniquely identifiable key string
                unique_key = f"{row['course_code_portal']} | {row['professor']}"
                
                # Create a readable label for display
                display_label = f"{row['course_code_portal']} - {row['course_name']} (Prof. {row['professor']} | {row['lecture_time']})"
                
                course_dict[unique_key] = {
                    "code": row['course_code_portal'],
                    "name": row['course_name'],
                    "professor": row['professor'],
                    "time": row['lecture_time'],
                    "label": display_label
                }
            return course_dict
            
        except Exception as e:
            st.sidebar.error(f"Error loading CSV file: {e}")
            
    # Minimal static fallback list if the CSV cannot be accessed
    return {
        "CAS4160-01-00 | Default": {
            "code": "CAS4160-01-00", "name": "Reinforcement Learning", "professor": "AI", "time": "Mon 1", "label": "CAS4160-01-00 - Reinforcement Learning"
        }
    }

# Load processed data dictionary
available_courses_dict = load_courses_from_csv()

# --- BACKEND PIPELINE MOCKS ---
class MockStrategyEngine:
    def predict_courses(self, requested_courses, context):
        results = []
        for c in requested_courses:
            # Generate a baseline threshold estimate simulating XGBoost pattern logic
            if "객체지향" in c["name"] or "Reinforcement" in c["name"]:
                base_thresh = 85
            elif "자료구조" in c["name"] or "Machine" in c["name"]:
                base_thresh = 55
            else:
                base_thresh = 15
                
            results.append({
                "code": c["code"],
                "name": f"{c['name']} (Prof. {c['professor']})",
                "rank": c["rank"],
                "predicted_threshold": base_thresh,
                "shap_breakdown": {"historical_demand": base_thresh // 2, "professor_popularity": 15, "year_quota_impact": 5}
            })
        return results

class MockBudgetOptimizer:
    def __init__(self, budget):
        self.budget = budget
        
    def allocate(self, predictions):
        output = []
        total_predicted = sum([p["predicted_threshold"] for p in predictions]) if predictions else 1
        leftover = self.budget
        
        for i, p in enumerate(predictions):
            if total_predicted > 0:
                share = p["predicted_threshold"] / total_predicted
                bid = int(self.budget * share)
            else:
                bid = self.budget // len(predictions)
                
            if bid > 99: bid = 99
            leftover -= bid
            
            risk = "Safe" if bid >= p["predicted_threshold"] else "Moderate Risk"
            if bid < (p["predicted_threshold"] * 0.75):
                risk = "High Risk"
                
            output.append({
                "code": p["code"],
                "name": p["name"],
                "rank": p["rank"],
                "predicted_threshold": p["predicted_threshold"],
                "allocated_bid": bid,
                "risk_level": risk,
                "shap_breakdown": p["shap_breakdown"]
            })
            
        if output and leftover > 0:
            output[0]["allocated_bid"] += leftover
        return output


# --- STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="Mileage Strategy Tester", layout="wide")

st.title("🧮 Custom Priority Mileage Strategy Tester")
st.subheader("Evaluate Part 2 Engine Using `course_summary_all.csv` Data")

# Side Parameters
st.sidebar.header("👤 Student Profile Attributes")
student_year = st.sidebar.slider("Grade Year Status", 1, 4, 4)
total_budget = st.sidebar.number_input("Total Mileage Points Budget", min_value=0, max_value=200, value=150)

st.sidebar.write("---")
st.sidebar.markdown("### 🗃️ Database Integrity Status")
if os.path.exists("course_summary_all.csv"):
    st.sidebar.success(f"Successfully synchronized {len(available_courses_dict)} unique options from CSV.")
else:
    st.sidebar.error("File `course_summary_all.csv` missing from directory root.")

# --- SELECTION AND BLOCK DUPLICATES INTERFACE ---
st.header("📋 Pick and Arrange Your Desired Schedule")
st.write("Search and select items from the database. Streamlit automatically guarantees an item can only be picked once.")

# The dynamic searchable multi-select widget
selected_unique_keys = st.multiselect(
    "Search or select courses from database:",
    options=list(available_courses_dict.keys()),
    format_func=lambda x: available_courses_dict[x]["label"]
)

# Process customized inputs and ranking parameters
requested_courses = []
if selected_unique_keys:
    st.write("### 🔢 Assign Priority Ranks")
    st.info("Assign a ranking number. 1 represents your highest preference/must-have asset.")
    
    # Render layout neatly using adaptive column blocks
    col_layout = st.columns(len(selected_unique_keys)) if len(selected_unique_keys) <= 4 else st.columns(4)
    
    for i, key in enumerate(selected_unique_keys):
        course_info = available_courses_dict[key]
        col_index = i % 4
        
        with col_layout[col_index]:
            st.markdown(f"**{course_info['name']}**")
            st.caption(f"Code: `{course_info['code']}` | Prof: {course_info['professor']}")
            
            # Input number handles ordering parameters manually
            assigned_rank = st.number_input(
                f"Priority Rank", 
                min_value=1, 
                max_value=len(selected_unique_keys), 
                value=min(i + 1, len(selected_unique_keys)), 
                key=f"rank_{key}"
            )
            
            requested_courses.append({
                "code": course_info['code'],
                "name": course_info['name'],
                "professor": course_info['professor'],
                "rank": assigned_rank
            })

    # Sort strictly based on priority numbers assigned
    requested_courses = sorted(requested_courses, key=lambda x: x["rank"])
else:
    st.warning("Your planning cart is currently empty. Use the dropdown selector above to query the database.")

st.write("---")

# --- STRATEGY COMPUTATION TRIGGER ---
if st.button("🚀 Run Optimizer Calculations", type="primary") and requested_courses:
    
    # Check for duplicate structural rankings
    ranks_checked = [c["rank"] for c in requested_courses]
    if len(ranks_checked) != len(set(ranks_checked)):
        st.error("❌ Conflict Detected: You have assigned identical rank positions to different subjects. Please ensure each item has an individual priority rank.")
    else:
        # Pass variables directly to processing mocks (swap for real files later)
        engine = MockStrategyEngine()
        optimizer = MockBudgetOptimizer(budget=total_budget)
        
        with st.spinner("Processing calculations across models..."):
            predictions = engine.predict_courses(requested_courses, {"year": student_year})
            final_strategy = optimizer.allocate(predictions)
            
        st.header("📊 Optimized Mileage Distribution Results")
        
        summary_data = []
        for item in final_strategy:
            summary_data.append({
                "Priority Rank": f"Rank {item['rank']}",
                "Course ID": item['code'],
                "Course offering details": item['name'],
                "Predicted Historical Threshold": f"{item['predicted_threshold']} pts",
                "Calculated Smart Bid Allocation": f"**{item['allocated_bid']} pts**",
                "Projected Risk Assessment": item['risk_level']
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.table(df_summary)
        
        # Balance tracking metrics
        total_used = sum([item['allocated_bid'] for item in final_strategy])
        st.metric(
            label="Total Points Accounted For", 
            value=f"{total_used} / {total_budget} pts", 
            delta=f"{total_budget - total_used} points leftover"
        )
