"""
scrape_mileage.py
-----------------
Scrapes mileage history back to 2022-1 using course code mapping
for pre-2025-1 semesters.

KEY BEHAVIORS:
1. For 2025-1 onwards: scrapes using new CAS/MAT codes directly
2. For pre-2025-1: scrapes all old codes from mapping for that course
3. Cache: if multiple new codes map to same old code+semester, fetch once copy rest
4. Professor embedded in every student row
5. Scrapes back to 2022-1 by default

PROCESSING LOGIC (in process_mileage_data.py):
- Groups by base_course_code + professor (ignores section number)
- Tier 1: same new_code + same professor  → weight 5.0
- Tier 2: same base_code + diff professor → weight 2.0
- Professor filter keeps relevant history, discards mismatches

USAGE:
  python3 scrape_mileage.py                           # 2022-2026, semesters 1+2
  python3 scrape_mileage.py --years 2022 2023 2024    # specific years
  python3 scrape_mileage.py --visible                  # watch browser
  python3 scrape_mileage.py --merge-only               # re-merge existing CSVs
"""

from __future__ import annotations

import asyncio
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

BASE_URL      = "https://yonsei-mileage-helper.vercel.app"
JSON_PATH     = "segmented_cs_courses.json"
MAPPING_PATH  = "course_code_mapping.json"
OUT_DIR       = Path("mileage_csvs")
MERGED_CSV    = "mileage_history_all.csv"
SUMMARY_CSV   = "course_summary_all.csv"

# Semesters using new CAS/MAT codes
NEW_CODE_FROM_YEAR     = 2025
NEW_CODE_FROM_SEMESTER = 1


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_course_code(code: str) -> tuple:
    """'CAS3116-01' → ('CAS3116', '01', '00')"""
    parts   = code.strip().split("-")
    base    = parts[0]
    section = parts[1].zfill(2) if len(parts) > 1 else "01"
    lab     = parts[2].zfill(2) if len(parts) > 2 else "00"
    return base, section, lab


def get_base_code(code: str) -> str:
    """'CAS3116-01' → 'CAS3116'"""
    return code.strip().split("-")[0]


def safe_filename(tag: str, name: str) -> str:
    name_clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()[:40]
    return f"{tag}_{name_clean}.csv"


def load_courses(json_path: str) -> dict:
    """Returns {course_code: course_name}"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    courses = {}
    for year_data in data.values():
        for cat_data in year_data.values():
            for code, course in cat_data.items():
                if code not in courses:
                    courses[code] = course["metadata"].get("name", code)
    return courses


def load_mapping(mapping_path: str) -> dict:
    """Returns {new_code: [old_code1, old_code2, ...]}"""
    if not Path(mapping_path).exists():
        print(f"⚠  No mapping file at {mapping_path} — pre-2025 scraping limited")
        return {}
    with open(mapping_path, encoding="utf-8") as f:
        return json.load(f)


def is_pre_new_code(year: int, semester: int) -> bool:
    """True if this semester uses old course codes (pre-2025-1)"""
    if year < NEW_CODE_FROM_YEAR:
        return True
    if year == NEW_CODE_FROM_YEAR and semester < NEW_CODE_FROM_SEMESTER:
        return True
    return False


def normalize_professor(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace(" ", "")


# ── Core fetch ─────────────────────────────────────────────────────────────────

async def fetch_one(page, base_code: str, section: str, lab: str,
                    year: int, semester: int) -> tuple:
    """
    Fetches one course-semester from the mileage helper.
    Returns (summary_dict, [student_row_dicts])
    Returns (None, []) if no data found.
    """
    # Load page with retry
    for attempt in range(3):
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(800)
            break
        except Exception as e:
            if attempt == 2:
                raise
            wait = 15 * (attempt + 1)
            print(f"\n    [load error, waiting {wait}s… retry {attempt+2}/3]",
                  end=" ", flush=True)
            await asyncio.sleep(wait)
            try:
                await page.goto("about:blank", timeout=5000)
                await asyncio.sleep(2)
            except Exception:
                pass

    # Set year
    current = int(await page.locator("input[name='year']").input_value())
    while current < year:
        await page.locator("button.btn.join-item").filter(has_text="»").click()
        await page.wait_for_timeout(350)
        current = int(await page.locator("input[name='year']").input_value())
    while current > year:
        await page.locator("button.btn.join-item").filter(has_text="«").click()
        await page.wait_for_timeout(350)
        current = int(await page.locator("input[name='year']").input_value())

    # Set semester
    label = "1학기" if semester == 1 else "2학기"
    await page.locator(f"input[type='radio'][aria-label='{label}']").click()
    await page.wait_for_timeout(250)

    # Fill course code fields
    for name_attr, value in [
        ("courseCode", base_code),
        ("section",    section),
        ("lab",        lab),
    ]:
        loc = page.locator(f"input[name='{name_attr}']")
        await loc.click(click_count=3)
        await loc.fill(value)
        await page.wait_for_timeout(100)

    v0 = await page.locator("input[name='courseCode']").input_value()
    v1 = await page.locator("input[name='section']").input_value()
    v2 = await page.locator("input[name='lab']").input_value()
    print(f"[{v0}|{v1}|{v2}]", end=" ", flush=True)

    await page.locator("button[type='submit']").click()

    # Wait for result
    try:
        await page.wait_for_selector("text=정원", timeout=12000)
    except PWTimeout:
        body = await page.inner_text("body")
        if "정원" not in body and "지원자" not in body:
            return None, []
    await page.wait_for_timeout(600)

    # Extract summary
    body_text = await page.inner_text("body")
    lines     = [l.strip() for l in body_text.split("\n") if l.strip()]

    label_map = {
        "학정번호":      "course_code_portal",
        "과목명":        "course_name_portal",
        "교수명":        "professor",
        "강의시간":      "lecture_time",
        "강의실":        "classroom",
        "정원":          "capacity",
        "전공자정원":    "major_quota",
        "지원자":        "total_applicants",
        "평균 마일리지": "avg_mileage",
    }
    numeric = {"capacity", "major_quota", "total_applicants", "avg_mileage"}

    summary = {}
    for i, line in enumerate(lines):
        if line in label_map and i + 1 < len(lines):
            key = label_map[line]
            val = lines[i + 1]
            if key in numeric:
                clean = re.sub(r"\([^)]*\)", "", val).strip()
                try:
                    summary[key] = float(clean)
                except ValueError:
                    summary[key] = clean
            else:
                summary[key] = val

    # Extract student rows
    rows   = []
    tables = await page.query_selector_all("table")
    target = None
    for t in tables:
        txt = await t.inner_text()
        if "순위" in txt and "마일리지" in txt:
            target = t
            break

    if target:
        headers_raw = [
            (await th.inner_text()).replace("\n", " ").strip()
            for th in await target.query_selector_all("th")
        ]
        col_map = {
            "No": "no", "순위": "rank", "마일리지": "mileage_bid",
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
        headers = [col_map.get(h, h.lower().replace(" ", "_"))
                   for h in headers_raw]

        for tr in await target.query_selector_all("tr"):
            cells = [(await td.inner_text()).strip()
                     for td in await tr.query_selector_all("td")]
            if not cells:
                continue
            if headers and len(headers) == len(cells):
                row = dict(zip(headers, cells))
            else:
                fallback = ["no", "rank", "mileage_bid", "is_major_incl_quota",
                            "num_courses_applied", "graduation_applicant",
                            "first_time_enrollment", "total_credit_ratio",
                            "prev_semester_credit_ratio", "grade_year",
                            "enrolled", "note"]
                row = {fallback[i] if i < len(fallback) else f"col_{i}": v
                       for i, v in enumerate(cells)}
            rows.append(row)
    else:
        pat = re.compile(
            r"^(\d+)\s+(\d+)\s+(\d+)\s+([YN]\([YN]\))\s+(\d+)"
            r"\s+([YN])\s+([YN])\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([YN])$"
        )
        for line in lines:
            m = pat.match(line)
            if m:
                rows.append({
                    "no": int(m.group(1)), "rank": int(m.group(2)),
                    "mileage_bid": int(m.group(3)),
                    "is_major_incl_quota": m.group(4),
                    "num_courses_applied": int(m.group(5)),
                    "graduation_applicant": m.group(6),
                    "first_time_enrollment": m.group(7),
                    "total_credit_ratio": float(m.group(8)),
                    "prev_semester_credit_ratio": float(m.group(9)),
                    "grade_year": int(m.group(10)),
                    "enrolled": m.group(11),
                })

    return summary, rows


def tag_rows(rows: list, summary: dict, new_code: str,
             old_code: str, name: str, year: int, semester: int) -> list:
    """
    Tag every student row with metadata.
    Professor is embedded here so it's never lost during merging.
    new_code = the 2026-1 CAS code (always saved under this)
    old_code = the actual code that was scraped (CCO/CSI etc for pre-2025)
    """
    professor    = summary.get("professor",        "") if summary else ""
    lecture_time = summary.get("lecture_time",     "") if summary else ""
    capacity     = summary.get("capacity",         "") if summary else ""
    major_quota  = summary.get("major_quota",      "") if summary else ""
    avg_mileage  = summary.get("avg_mileage",      "") if summary else ""
    total_apps   = summary.get("total_applicants", "") if summary else ""

    tagged = []
    for r in rows:
        r = r.copy()
        r["course_code"]      = new_code        # always the new CAS code
        r["scraped_code"]     = old_code        # what was actually queried
        r["course_name"]      = name
        r["year"]             = year
        r["semester"]         = semester
        r["professor"]        = professor       # embedded at row level
        r["professor_norm"]   = normalize_professor(professor)
        r["lecture_time"]     = lecture_time
        r["capacity"]         = capacity
        r["major_quota"]      = major_quota
        r["avg_mileage"]      = avg_mileage
        r["total_applicants"] = total_apps
        tagged.append(r)
    return tagged


# ── Main scrape loop ───────────────────────────────────────────────────────────

async def scrape_all(courses: dict, years: list, semesters: list,
                     mapping: dict, headless: bool = True,
                     delay: float = 3.0):

    OUT_DIR.mkdir(exist_ok=True)

    all_rows           = []
    all_summaries      = []
    failed             = []
    consecutive_errors = 0

    # Cache: {(scraped_code, year, semester): (summary, rows)}
    # Avoids re-fetching same old code for multiple new codes
    fetch_cache: dict = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        page    = await (await browser.new_context()).new_page()

        total = len(courses) * len(years) * len(semesters)
        done  = 0

        for new_code, name in courses.items():
            for year in years:
                for semester in semesters:
                    done += 1
                    label = f"[{done}/{total}] {new_code} ({year}-{semester}학기)"

                    # Which codes to scrape for this semester?
                    if is_pre_new_code(year, semester):
                        # Pre-2025-1: use mapped old codes
                        old_codes = mapping.get(new_code, [])
                        if not old_codes:
                            # No mapping — try new code anyway in case it existed
                            codes_to_fetch = [new_code]
                        else:
                            codes_to_fetch = old_codes
                    else:
                        # 2025-1 onwards: use new code directly
                        codes_to_fetch = [new_code]

                    for fetch_code in codes_to_fetch:
                        # Filename always uses new_code so data is grouped correctly
                        tag   = f"{new_code}_{fetch_code}_{year}_{semester}학기"
                        fname = safe_filename(tag, name)
                        fpath = OUT_DIR / fname

                        # Skip if already saved
                        if fpath.exists():
                            print(f"  {label} [{fetch_code}]... skipped")
                            try:
                                ex = pd.read_csv(fpath, encoding="utf-8-sig")
                                all_rows.extend(ex.to_dict("records"))
                            except Exception:
                                pass
                            continue

                        cache_key = (fetch_code, year, semester)

                        # Check cache — already fetched this exact code+semester
                        if cache_key in fetch_cache:
                            cached_summary, cached_rows = fetch_cache[cache_key]
                            if not cached_rows:
                                print(f"  {label} [{fetch_code}]... "
                                      f"cached (no data)")
                                continue

                            # Re-tag with this new_code and save
                            tagged = tag_rows(
                                [r.copy() for r in cached_rows],
                                cached_summary, new_code, fetch_code,
                                name, year, semester
                            )
                            pd.DataFrame(tagged).to_csv(
                                fpath, index=False, encoding="utf-8-sig"
                            )
                            all_rows.extend(tagged)
                            print(f"  {label} [{fetch_code}]... "
                                  f"copied ({len(tagged)} rows from cache)")

                            if cached_summary:
                                s = cached_summary.copy()
                                s.update({
                                    "course_code":  new_code,
                                    "scraped_code": fetch_code,
                                    "course_name":  name,
                                    "year":         year,
                                    "semester":     semester,
                                })
                                all_summaries.append(s)
                            continue

                        # Fresh fetch
                        print(f"  Fetching {label} [{fetch_code}]...",
                              end=" ", flush=True)

                        base, section, lab = parse_course_code(fetch_code)

                        try:
                            summary, rows = await fetch_one(
                                page, base, section, lab, year, semester
                            )
                            consecutive_errors = 0
                        except Exception as e:
                            consecutive_errors += 1
                            print(f"ERROR: {e}")
                            failed.append(
                                (new_code, fetch_code, year, semester, str(e))
                            )
                            fetch_cache[cache_key] = (None, [])
                            if consecutive_errors >= 3:
                                print(f"\n  ⚠ {consecutive_errors} errors "
                                      f"— pausing 60s...")
                                await asyncio.sleep(60)
                                consecutive_errors = 0
                            continue

                        # Store in cache
                        fetch_cache[cache_key] = (summary, rows or [])

                        if not rows:
                            print("no data")
                            await asyncio.sleep(delay)
                            continue

                        print(f"{len(rows)} student records  "
                              f"(prof: {summary.get('professor','')})")

                        tagged = tag_rows(
                            rows, summary, new_code, fetch_code,
                            name, year, semester
                        )

                        pd.DataFrame(tagged).to_csv(
                            fpath, index=False, encoding="utf-8-sig"
                        )
                        all_rows.extend(tagged)

                        if summary:
                            s = summary.copy()
                            s.update({
                                "course_code":  new_code,
                                "scraped_code": fetch_code,
                                "course_name":  name,
                                "year":         year,
                                "semester":     semester,
                            })
                            all_summaries.append(s)

                        _save_merged(all_rows, all_summaries)
                        await asyncio.sleep(delay)

        await browser.close()

    _save_merged(all_rows, all_summaries, final=True)

    if failed:
        print(f"\n⚠  Failed {len(failed)} requests:")
        for nc, oc, y, s, err in failed:
            print(f"   {nc} via {oc} {y}-{s}: {err}")

    return all_rows, all_summaries


def _save_merged(all_rows: list, all_summaries: list, final: bool = False):
    if all_rows:
        df = pd.DataFrame(all_rows)
        for col in ["rank", "no", "mileage_bid",
                    "num_courses_applied", "grade_year"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["total_credit_ratio", "prev_semester_credit_ratio"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
        if final:
            print(f"\n✅ {MERGED_CSV}  →  {len(df)} rows")
            if "professor" in df.columns:
                has = df["professor"].notna() & (df["professor"] != "")
                print(f"   Professor coverage: {has.sum()}/{len(df)} rows")
                print("   Per year/semester:")
                cov = df[has].groupby(
                    ["year", "semester"]
                )["professor"].count().reset_index()
                print(cov.to_string(index=False))

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(
            SUMMARY_CSV, index=False, encoding="utf-8-sig"
        )
        if final:
            print(f"✅ {SUMMARY_CSV}  →  {len(all_summaries)} records")


def merge_only():
    files = sorted(OUT_DIR.glob("*.csv"))
    if not files:
        print(f"No CSVs in {OUT_DIR}/")
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
        print(f"✅ Merged {len(files)} files → {MERGED_CSV} ({len(merged)} rows)")
        if "professor" in merged.columns:
            has = merged["professor"].notna() & (merged["professor"] != "")
            print(f"   Professor coverage: {has.sum()}/{len(merged)}")
    else:
        print("No valid CSVs to merge.")


def main():
    p = argparse.ArgumentParser(
        description="Scrape Yonsei mileage history with course code mapping"
    )
    p.add_argument("--years",      nargs="+", type=int,
                   default=list(range(2022, 2027)),
                   help="Years to scrape (default: 2022-2026)")
    p.add_argument("--semesters",  nargs="+", type=int, default=[1, 2],
                   help="Semesters (default: 1 2)")
    p.add_argument("--visible",    action="store_true")
    p.add_argument("--delay",      type=float, default=3.0)
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--json",       default=JSON_PATH)
    p.add_argument("--mapping",    default=MAPPING_PATH)
    args = p.parse_args()

    if args.merge_only:
        merge_only()
        return

    courses = load_courses(args.json)
    mapping = load_mapping(args.mapping)

    mapped   = sum(1 for v in mapping.values() if v)
    unmapped = sum(1 for v in mapping.values() if not v)
    print(f"Loaded {len(courses)} courses")
    print(f"  {mapped} courses with old code mappings")
    print(f"  {unmapped} courses with no old codes (will use new code only)")
    print(f"Years: {args.years}  |  Semesters: {args.semesters}")
    print(f"Pre-2025-1 semesters will use mapped old codes\n")

    asyncio.run(scrape_all(
        courses=courses,
        years=args.years,
        semesters=args.semesters,
        mapping=mapping,
        headless=not args.visible,
        delay=args.delay,
    ))


if __name__ == "__main__":
    main()