import json
from collections import defaultdict
from database import CS_COURSES  # Imports your original raw database

def migrate_database(old_db):
    # Use a lambda to automatically create nested dictionaries on the fly
    tree_db = defaultdict(lambda: defaultdict(dict))
    
    for code, course in old_db.items():
        # 1. Determine the Year Layer
        # Fallback to 'unknown' if missing, otherwise build 'major_year_X'
        year_val = course.get("major year")
        year_key = f"major_year_{year_val}" if year_val else "major_year_unknown"
        
    
        
        # 3. Separate Clean Metadata
        metadata = {
            "name": course.get("name"),
            "professor": course.get("professor"),
            "language_medium": course.get("language medium"),
            "lecture_type": course.get("lecture type"),
            "credits": int(course.get("credits", 0)) if course.get("credits") else 0,
            "workload": course.get("workload"),
            "difficulty": course.get("difficulty"),
            "time": course.get("time"),
            "location": course.get("location"),
            "evaluation_type": course.get("evaluation type"),
            "prerequisites": course.get("prerequisites", []),
            # Safely cast ETA to an integer if it's stored as a string numeric
            "mileage_historical_eta": int(course.get("added by in ETA", 0)) if str(course.get("added by in ETA", "")).isdigit() else course.get("added by in ETA"),
            "keywords": course.get("keywords", [])
        }
        
        # 4. Separate Heavy Text Chunks (For the LLM RAG injection context)
        text_chunks = {
            "grading_and_syllabus": course.get("grading scheme", ""),
            "student_reviews": course.get("review", ""),
            "alternative_professor_reviews": course.get("review from other courses", "")
        }
        
        payload = {
            "metadata": metadata,
            "text_chunks": text_chunks
        }

        # 4. Determine Dynamic Categorization Branches
        assigned = False
        if course.get("major requirement"):
            tree_db[year_key]["major_requirement"][code] = payload
            assigned = True
        if course.get("major basic"):
            tree_db[year_key]["major_basic"][code] = payload
            assigned = True
        if course.get("major elective"):
            tree_db[year_key]["major_elective"][code] = payload
            assigned = True
            
        # Catch-all fallback if all boolean switches are False
        if not assigned:
            tree_db[year_key]["general_elective"][code] = payload
            
    # Convert defaultdict back to standard dict objects for clean output
    return {year: dict(categories) for year, categories in tree_db.items()}

if __name__ == "__main__":
    print("🚀 Initializing database transformation pipeline...")
    
    # Run the migration function
    new_tree_structure = migrate_database(CS_COURSES)
    
    # Save it out neatly as a formatted JSON file
    output_filename = "segmented_cs_courses.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(new_tree_structure, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Migration successful! New structured asset saved to: '{output_filename}'")