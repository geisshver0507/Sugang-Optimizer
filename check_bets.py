"""
check_bets.py — Check bets against 2026-1 results. Concrete YES/NO.

Input: betting.txt
  Line 1: year of study
  Line 2+: course_code,mileage_bet

Usage: python3 check_bets.py [filename]
"""

import sys
import pandas as pd
from pathlib import Path

RAW_CSV = "database/mileage_history_all.csv"


def check(betting_file="betting.txt"):
    if not Path(RAW_CSV).exists():
        print("ERROR: " + RAW_CSV + " not found"); sys.exit(1)
    if not Path(betting_file).exists():
        print("ERROR: " + betting_file + " not found"); sys.exit(1)

    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=["mileage_bid"])
    df["mileage_bid"] = pd.to_numeric(df["mileage_bid"], errors="coerce")
    df["grade_year"] = pd.to_numeric(df["grade_year"], errors="coerce")
    df["enrolled_bool"] = (df["enrolled"].astype(str).str.upper() == "Y").astype(int)
    d26 = df[(df["year"] == 2026) & (df["semester"] == 1)]

    with open(betting_file, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    year = int(lines[0])
    bets = []
    for line in lines[1:]:
        parts = line.split(",")
        bets.append((parts[0].strip(), int(parts[1].strip())))

    total_pts = sum(b for _, b in bets)

    print("")
    print("Year: " + str(year) + "  |  Courses: " + str(len(bets)) + "  |  Total bet: " + str(total_pts) + "/72")
    print("")
    print("{:<16s} {:>5s}  {:>6s}  {}".format("Course", "Bid", "Result", "Reason"))
    print("-" * 70)

    wins = 0
    for code, bid in bets:
        sub = d26[d26["course_code"] == code]
        if len(sub) == 0:
            print("{:<16s} {:>5d}  {:>6s}  {}".format(code, bid, "??", "No 2026-1 data"))
            continue

        capacity = int(sub["capacity"].iloc[0])

        # Use year-specific data if enough exists, else all years
        yr_sub = sub[sub["grade_year"] == year]
        pool = yr_sub if len(yr_sub) >= 5 else sub
        pool_label = "yr" + str(year) if len(yr_sub) >= 5 else "all"

        # Find the max rejected bid — the highest bid that did NOT get in
        rejected = pool[pool["enrolled_bool"] == 0]
        accepted = pool[pool["enrolled_bool"] == 1]

        if len(rejected) == 0:
            # Everyone got in — you'd get in too
            result = "YES"
            reason = "Everyone got in, no rejections"
        else:
            max_rejected_bid = int(rejected["mileage_bid"].max())
            min_accepted_bid = int(accepted["mileage_bid"].min()) if len(accepted) > 0 else 0

            if bid > max_rejected_bid:
                # You bid higher than the highest person who got rejected
                result = "YES"
                reason = "Bid " + str(bid) + " > highest rejected bid " + str(max_rejected_bid) + " (" + pool_label + ")"
            elif bid < min_accepted_bid:
                # You bid lower than the lowest person who got in
                result = "NO"
                reason = "Bid " + str(bid) + " < lowest accepted bid " + str(min_accepted_bid) + " (" + pool_label + ")"
            else:
                # Tiebreak zone — check what actually happened at your bid
                at_bid = pool[pool["mileage_bid"] == bid]
                if len(at_bid) > 0:
                    if at_bid["enrolled_bool"].mean() >= 0.5:
                        result = "YES"
                        reason = str(at_bid["enrolled_bool"].sum()) + "/" + str(len(at_bid)) + " at bid=" + str(bid) + " got in (" + pool_label + ")"
                    else:
                        result = "NO"
                        reason = str(at_bid["enrolled_bool"].sum()) + "/" + str(len(at_bid)) + " at bid=" + str(bid) + " got in (" + pool_label + ")"
                else:
                    # Nobody bet this exact amount — check nearest below
                    lower = pool[pool["mileage_bid"] < bid].sort_values("mileage_bid", ascending=False)
                    higher = pool[pool["mileage_bid"] > bid].sort_values("mileage_bid", ascending=True)

                    if len(lower) > 0:
                        nearest_below = int(lower["mileage_bid"].iloc[0])
                        below_rate = lower[lower["mileage_bid"] == nearest_below]["enrolled_bool"].mean()
                        if below_rate >= 0.5:
                            result = "YES"
                            reason = "Nobody bet " + str(bid) + ", but bid=" + str(nearest_below) + " got in (" + pool_label + ")"
                        else:
                            # Check above too
                            if len(higher) > 0:
                                nearest_above = int(higher["mileage_bid"].iloc[0])
                                above_rate = higher[higher["mileage_bid"] == nearest_above]["enrolled_bool"].mean()
                                if above_rate < 0.5:
                                    result = "NO"
                                    reason = "Even bid=" + str(nearest_above) + " mostly rejected (" + pool_label + ")"
                                else:
                                    result = "NO"
                                    reason = "bid=" + str(nearest_below) + " rejected, need >=" + str(nearest_above) + " (" + pool_label + ")"
                            else:
                                result = "NO"
                                reason = "bid=" + str(nearest_below) + " rejected (" + pool_label + ")"
                    elif len(higher) > 0:
                        nearest_above = int(higher["mileage_bid"].iloc[0])
                        above_rate = higher[higher["mileage_bid"] == nearest_above]["enrolled_bool"].mean()
                        if above_rate >= 0.5:
                            result = "NO"
                            reason = "Need at least " + str(nearest_above) + " (" + pool_label + ")"
                        else:
                            result = "NO"
                            reason = "Even higher bids rejected (" + pool_label + ")"
                    else:
                        result = "NO"
                        reason = "No comparable data"

        tag = " ✅" if result == "YES" else " ❌"
        if result == "YES":
            wins += 1
        print("{:<16s} {:>5d}  {:>4s}{}  {}".format(code, bid, result, tag, reason))

    print("-" * 70)
    print("Result: " + str(wins) + "/" + str(len(bets)) + " courses won")
    if total_pts > 72:
        print("WARNING: Total " + str(total_pts) + " exceeds 72pt limit!")
    elif total_pts < 72:
        print("Unused: " + str(72 - total_pts) + " points wasted")
    print("")


if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else "betting.txt"
    check(f)