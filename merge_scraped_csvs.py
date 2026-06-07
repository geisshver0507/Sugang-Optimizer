"""merge_scraped_csvs.py — merge per-semester CSVs into mileage_history_all.csv"""
import sys, argparse, pandas as pd
from pathlib import Path
OUTPUT_FILE = "mileage_history_all.csv"
def merge(folder="."):
    fp = Path(folder).expanduser().resolve()
    files = [f for f in sorted(fp.glob("*.csv")) if f.name != OUTPUT_FILE]
    if not files: print("No CSVs in " + str(fp)); sys.exit(1)
    print("Found " + str(len(files)) + " files")
    frames, failed = [], []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if not df.empty: frames.append(df)
        except Exception as e: failed.append(str(f.name) + ": " + str(e))
    if not frames: print("Nothing readable"); sys.exit(1)
    m = pd.concat(frames, ignore_index=True)
    m["rank"] = pd.to_numeric(m["rank"], errors="coerce")
    m["mileage_bid"] = pd.to_numeric(m["mileage_bid"], errors="coerce")
    m["grade_year"] = pd.to_numeric(m["grade_year"], errors="coerce")
    dc = [c for c in ["course_code","scraped_code","year","semester","rank","mileage_bid","grade_year","enrolled"] if c in m.columns]
    b = len(m); m = m.drop_duplicates(subset=dc); a = len(m)
    out = fp / OUTPUT_FILE; m.to_csv(out, index=False, encoding="utf-8-sig")
    print("Merged -> " + str(out) + " | " + str(a) + " rows (dropped " + str(b-a) + " dupes)")
    for (yr,sem),cnt in m.groupby(["year","semester"]).size().sort_index().items():
        print("  " + str(yr) + "-" + str(sem) + ": " + str(cnt))
    if failed:
        for f in failed: print("FAIL: " + f)
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default=".")
    merge(folder=p.parse_args().folder)
