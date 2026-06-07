"""Course database loading helpers."""

import json
from pathlib import Path


DEFAULT_DATABASE = "segmented_cs_courses.json"


def load_tree_database(filename=DEFAULT_DATABASE):
    db_path = Path(__file__).resolve().parent / filename
    if not db_path.exists():
        raise FileNotFoundError(f"Missing course database: {db_path}")
    with db_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_tree_database(tree_db):
    courses = {}
    for year_bucket in tree_db.values():
        for category_bucket in year_bucket.values():
            courses.update(category_bucket)
    return courses
