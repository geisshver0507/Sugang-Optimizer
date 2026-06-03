"""
scrape_mileage.py
-----------------
Scrapes mileage betting history for every course in segmented_cs_courses.json
from https://yonsei-mileage-helper.vercel.app

OUTPUTS (all in same folder as this script):
  mileage_csvs/
      CAS3205-01-00_2026_1학기_컴퓨터그래픽스.csv   ← one per course per semester
      ...
  mileage_history_all.csv     ← all student rows merged
  course_summary_all.csv      ← one row per course per semester (capacity, prof, etc.)

USAGE:
  # Fetch 2026 semester 1 only (default)
  python3 scrape_mileage.py

  # Fetch multiple years and semesters
  python3 scrape_mileage.py --years 2024 2025 2026 --semesters 1 2

  # Watch the browser while it runs
  python3 scrape_mileage.py --years 2024 2025 2026 --semesters 1 2 --visible

  # Just re-merge already-downloaded CSVs without scraping
  python3 scrape_mileage.py --merge-only

  # Slower requests if you keep getting rate-limited
  python3 scrape_mileage.py --years 2024 2025 2026 --semesters 1 2 --delay 5

RESUME: If the script crashes or you stop it, just run the same command again.
        Already-saved CSVs are skipped automatically.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_URL    = "https://yonsei-mileage-helper.vercel.app"
JSON_PATH   = "segmented_cs_courses.json"
OUT_DIR     = Path("mileage_csvs")
MERGED_CSV  = "mileage_history_all.csv"
SUMMARY_CSV = "course_summary_all.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_course_code(code: str) -> tuple:
    """
    'CAS3205-01-00' → ('CAS3205', '01', '00')
    'CAS3120-01'    → ('CAS3120',  '01', '00')
    """
    parts = code.strip().split("-")
    base    = parts[0]
    section = parts[1].zfill(2) if len(parts) > 1 else "01"
    lab     = parts[2].zfill(2) if len(parts) > 2 else "00"
    return base, section, lab


def safe_filename(code: str, name: str) -> str:
    name_clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()[:40]
    return f"{code}_{name_clean}.csv"


def load_courses(json_path: str) -> dict:
    """Returns {course_code: course_name} deduplicated from the shared JSON."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    courses = {}
    for year_data in data.values():
        for cat_data in year_data.values():
            for code, course in cat_data.items():
                if code not in courses:
                    courses[code] = course["metadata"].get("name", code)
    return courses


# ── Core scraper ───────────────────────────────────────────────────────────────

async def fetch_one(page, base_code: str, section: str, lab: str,
                    year: int, semester: int) -> tuple:
    """
    Loads the page, fills the form, clicks search, and returns:
        (summary_dict, list_of_row_dicts)
    Returns (None, []) if no data found for this course/semester.
    """

    # ── 1. Load page with retry ───────────────────────────────────────────
    for attempt in range(3):
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(800)
            break
        except Exception as e:
            err = str(e)
            if attempt == 2:
                raise
            wait = 15 * (attempt + 1)
            print(f"\n    [load error, waiting {wait}s… attempt {attempt+2}/3]",
                  end=" ", flush=True)
            await asyncio.sleep(wait)
            # Clear any stuck error page
            try:
                await page.goto("about:blank", timeout=5000)
                await asyncio.sleep(2)
            except Exception:
                pass

    # ── 2. Set year ───────────────────────────────────────────────────────
    # The hidden input[name="year"] always reflects what's shown
    current = int(await page.locator("input[name='year']").input_value())
    while current < year:
        await page.locator("button.btn.join-item").filter(has_text="»").click()
        await page.wait_for_timeout(350)
        current = int(await page.locator("input[name='year']").input_value())
    while current > year:
        await page.locator("button.btn.join-item").filter(has_text="«").click()
        await page.wait_for_timeout(350)
        current = int(await page.locator("input[name='year']").input_value())

    # ── 3. Set semester ───────────────────────────────────────────────────
    label = "1학기" if semester == 1 else "2학기"
    await page.locator(f"input[type='radio'][aria-label='{label}']").click()
    await page.wait_for_timeout(250)

    # ── 4. Fill course code fields ────────────────────────────────────────
    # Exact input names confirmed from DOM inspection:
    #   input[name='courseCode']  (max 7 chars, e.g. CAS3205)
    #   input[name='section']     (max 2 chars, e.g. 01)
    #   input[name='lab']         (max 2 chars, e.g. 00)
    for name_attr, value in [
        ("courseCode", base_code),
        ("section",    section),
        ("lab",        lab),
    ]:
        loc = page.locator(f"input[name='{name_attr}']")
        await loc.click(click_count=3)
        await loc.fill(value)
        await page.wait_for_timeout(100)

    # Print what's actually in the fields so you can see it's working
    v0 = await page.locator("input[name='courseCode']").input_value()
    v1 = await page.locator("input[name='section']").input_value()
    v2 = await page.locator("input[name='lab']").input_value()
    print(f"[{v0}|{v1}|{v2}]", end=" ", flush=True)

    # ── 5. Submit ─────────────────────────────────────────────────────────
    await page.locator("button[type='submit']").click()

    # ── 6. Wait for result ────────────────────────────────────────────────
    try:
        await page.wait_for_selector("text=정원", timeout=12000)
    except PWTimeout:
        body = await page.inner_text("body")
        if "정원" not in body and "지원자" not in body:
            return None, []
    await page.wait_for_timeout(600)

    # ── 7. Extract summary card ───────────────────────────────────────────
    body_text = await page.inner_text("body")
    lines = [l.strip() for l in body_text.split("\n") if l.strip()]

    # Every field is rendered as:  label\nvalue  (on consecutive lines)
    label_map = {
        "학정번호":       "course_code_portal",
        "과목명":         "course_name_portal",
        "교수명":         "professor",
        "강의시간":       "lecture_time",
        "강의실":         "classroom",
        "정원":           "capacity",
        "전공자정원":     "major_quota",
        "지원자":         "total_applicants",
        "평균 마일리지":  "avg_mileage",
    }
    numeric = {"capacity", "major_quota", "total_applicants", "avg_mileage"}

    summary = {}
    for i, line in enumerate(lines):
        if line in label_map and i + 1 < len(lines):
            key = label_map[line]
            val = lines[i + 1]
            if key in numeric:
                # Strip suffix like "80(N)" → 80
                clean = re.sub(r"\([^)]*\)", "", val).strip()
                try:
                    summary[key] = float(clean)
                except ValueError:
                    summary[key] = clean
            else:
                summary[key] = val

    # ── 8. Extract student rows table ─────────────────────────────────────
    rows = []

    # Try table element first
    tables = await page.query_selector_all("table")
    target = None
    for t in tables:
        txt = await t.inner_text()
        if "순위" in txt and "마일리지" in txt:
            target = t
            break

    if target:
        # Get headers from <th> elements
        headers_raw = [
            (await th.inner_text()).replace("\n", " ").strip()
            for th in await target.query_selector_all("th")
        ]
        # Map Korean headers to English keys
        col_map = {
            "No": "no",
            "순위": "rank",
            "마일리지": "mileage_bid",
            "전공자(전공자정원포함여부)": "is_major_incl_quota",
            "신청과목수": "num_courses_applied",
            "졸업신청": "graduation_applicant",
            "초수강": "first_time_enrollment",
            "총이수": "total_credit_ratio",
            "직전이수": "prev_semester_credit_ratio",
            "학년": "grade_year",
            "수강여부": "enrolled",
            "비고": "note",
        }
        headers = [col_map.get(h, h.lower().replace(" ", "_")) for h in headers_raw]

        for tr in await target.query_selector_all("tr"):
            cells = [
                (await td.inner_text()).strip()
                for td in await tr.query_selector_all("td")
            ]
            if not cells:
                continue
            if headers and len(headers) == len(cells):
                row = dict(zip(headers, cells))
            else:
                # Positional fallback
                fallback = ["no", "rank", "mileage_bid", "is_major_incl_quota",
                            "num_courses_applied", "graduation_applicant",
                            "first_time_enrollment", "total_credit_ratio",
                            "prev_semester_credit_ratio", "grade_year", "enrolled", "note"]
                row = {fallback[i] if i < len(fallback) else f"col_{i}": v
                       for i, v in enumerate(cells)}
            rows.append(row)

    else:
        # Fallback: parse rows directly from page text
        # Pattern matches lines like: "1 1 35 Y(Y) 6 Y Y 1 0.1578 4 Y"
        pat = re.compile(
            r"^(\d+)\s+(\d+)\s+(\d+)\s+([YN]\([YN]\))\s+(\d+)"
            r"\s+([YN])\s+([YN])\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([YN])$"
        )
        for line in lines:
            m = pat.match(line)
            if m:
                rows.append({
                    "no":                         int(m.group(1)),
                    "rank":                       int(m.group(2)),
                    "mileage_bid":                int(m.group(3)),
                    "is_major_incl_quota":        m.group(4),
                    "num_courses_applied":        int(m.group(5)),
                    "graduation_applicant":       m.group(6),
                    "first_time_enrollment":      m.group(7),
                    "total_credit_ratio":         float(m.group(8)),
                    "prev_semester_credit_ratio": float(m.group(9)),
                    "grade_year":                 int(m.group(10)),
                    "enrolled":                   m.group(11),
                })

    return summary, rows


# ── Main loop ──────────────────────────────────────────────────────────────────

async def scrape_all(courses: dict, years: list, semesters: list,
                     headless: bool = True, delay: float = 3.0):

    OUT_DIR.mkdir(exist_ok=True)

    all_rows      = []
    all_summaries = []
    failed        = []
    consecutive_errors = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        page    = await (await browser.new_context()).new_page()

        total = len(courses) * len(years) * len(semesters)
        done  = 0

        for code, name in courses.items():
            base, section, lab = parse_course_code(code)

            for year in years:
                for semester in semesters:
                    done += 1
                    tag   = f"{code}_{year}_{semester}학기"
                    fname = safe_filename(tag, name)
                    fpath = OUT_DIR / fname
                    label = f"[{done}/{total}] {code} ({year}-{semester}학기)"

                    # Skip if this combination already saved
                    if fpath.exists():
                        print(f"  {label}... skipped (already saved)")
                        try:
                            existing = pd.read_csv(fpath, encoding="utf-8-sig")
                            all_rows.extend(existing.to_dict("records"))
                        except Exception:
                            pass
                        continue

                    print(f"  Fetching {label}...", end=" ", flush=True)

                    try:
                        summary, rows = await fetch_one(
                            page, base, section, lab, year, semester
                        )
                        consecutive_errors = 0

                    except Exception as e:
                        consecutive_errors += 1
                        print(f"ERROR: {e}")
                        failed.append((code, year, semester, str(e)))
                        if consecutive_errors >= 3:
                            print(f"\n  ⚠ {consecutive_errors} consecutive errors "
                                  f"— pausing 60s for server to recover...")
                            await asyncio.sleep(60)
                            consecutive_errors = 0
                        continue

                    if not rows:
                        print("no data")
                        await asyncio.sleep(delay)
                        continue

                    print(f"{len(rows)} student records")

                    # Tag every row with metadata
                    for r in rows:
                        r["course_code"] = code
                        r["course_name"] = name
                        r["year"]        = year
                        r["semester"]    = semester

                    # Save individual CSV immediately
                    pd.DataFrame(rows).to_csv(fpath, index=False, encoding="utf-8-sig")
                    all_rows.extend(rows)

                    # Save summary
                    if summary:
                        summary.update({
                            "course_code": code,
                            "course_name": name,
                            "year":        year,
                            "semester":    semester,
                        })
                        all_summaries.append(summary)

                    # Write merged files after every course (safe progress)
                    _save_merged(all_rows, all_summaries)

                    await asyncio.sleep(delay)

        await browser.close()

    # Final clean save
    _save_merged(all_rows, all_summaries, final=True)

    if failed:
        print(f"\n⚠  Failed {len(failed)} requests:")
        for code, year, sem, err in failed:
            print(f"   {code} {year}-{sem}학기: {err}")

    return all_rows, all_summaries


def _save_merged(all_rows: list, all_summaries: list, final: bool = False):
    """Write merged CSVs with cleaned numeric types."""
    if all_rows:
        df = pd.DataFrame(all_rows)
        for col in ["rank", "no", "mileage_bid", "num_courses_applied", "grade_year"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["total_credit_ratio", "prev_semester_credit_ratio"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
        if final:
            print(f"\n✅ {MERGED_CSV}  →  {len(df)} student rows")

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
        if final:
            print(f"✅ {SUMMARY_CSV}  →  {len(all_summaries)} course-semester records")


# ── Merge-only mode ────────────────────────────────────────────────────────────

def merge_only():
    """Re-merge all CSVs in mileage_csvs/ without scraping."""
    files = sorted(OUT_DIR.glob("*.csv"))
    if not files:
        print(f"No CSVs found in {OUT_DIR}/")
        return

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
        except Exception as e:
            print(f"  Skip {f.name}: {e}")

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
        print(f"✅ Merged {len(files)} files → {MERGED_CSV}  ({len(merged)} rows)")
    else:
        print("No valid CSVs to merge.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Scrape Yonsei mileage history")
    p.add_argument("--years",      nargs="+", type=int, default=[2026],
                   help="Academic years  (default: 2026)")
    p.add_argument("--semesters",  nargs="+", type=int, default=[1],
                   help="Semesters 1/2  (default: 1)")
    p.add_argument("--visible",    action="store_true",
                   help="Show browser window")
    p.add_argument("--delay",      type=float, default=3.0,
                   help="Seconds between requests (default: 3.0)")
    p.add_argument("--merge-only", action="store_true",
                   help="Only merge existing CSVs, skip scraping")
    p.add_argument("--json",       default=JSON_PATH,
                   help=f"Path to course JSON (default: {JSON_PATH})")
    args = p.parse_args()

    if args.merge_only:
        merge_only()
        return

    courses = load_courses(args.json)
    print(f"Loaded {len(courses)} courses from {args.json}")
    print(f"Years: {args.years}  |  Semesters: {args.semesters}  |  Delay: {args.delay}s")
    print(f"Total requests to make: {len(courses) * len(args.years) * len(args.semesters)}")
    print(f"Output dir: {OUT_DIR}/\n")

    asyncio.run(scrape_all(
        courses=courses,
        years=args.years,
        semesters=args.semesters,
        headless=not args.visible,
        delay=args.delay,
    ))


if __name__ == "__main__":
    main()