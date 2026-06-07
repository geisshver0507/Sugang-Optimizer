"""
survival_model.py
=================
The core intelligence of the mileage optimizer.

THE KEY IDEA
------------
The raw scraped data is PER-APPLICANT: for every student who bid on a course
we know their bid, their year, and whether they got a seat. This lets us build
a *survival curve* for each course:

        P(get a seat | you bid X points, you are year Y)

instead of collapsing everything into one "winning threshold" number (which
the old pipeline did, and which threw away ~95% of the signal).

From the survival curve we can answer the question students actually have:

    "What is the CHEAPEST bid that still gets me in with high confidence?"

This is what makes the system smarter than a student bidding naively. A course
can be important (rank 1) yet winnable for 2-3 points because the class is large
relative to demand — the curve reveals that, and we free those points up for
genuinely contested courses.

DATA SOURCES (priority order, professor-aware)
----------------------------------------------
  1. Same course code + same 2026 professor   (best — exact match)
  2. Same course code, any professor, recent   (good — course-level demand)
  3. Same course code, older years (pre-2025)  (ok — mapped via course_code)
  4. No data at all                            (ETA/capacity demand proxy)

Old codes (CSI*, CCO*, AAI*) are already unified into the new CAS code in the
`course_code` column, so pre-2025 history is automatically included.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = ROOT / "database"
RAW_CSV = str(DATABASE_DIR / "mileage_history_all.csv")
JSON_PATH = str(DATABASE_DIR / "segmented_cs_courses.json")

# How recent semesters are weighted when blending across years.
# Backtesting showed MASSIVE year-over-year drift (e.g. OS went 36pt→2pt,
# CAS1102 went 36pt→1pt between 2022 and 2026), so old data is steeply
# discounted — the most recent identical offering dominates.
RECENCY_WEIGHTS = {2026: 1.0, 2025: 0.7, 2024: 0.35, 2023: 0.18, 2022: 0.12}

MAX_BID = 36


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _norm_prof(name) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace(" ", "")


def load_applicant_data(raw_csv: str = RAW_CSV) -> pd.DataFrame:
    """Load the per-applicant rows, cleaned."""
    df = pd.read_csv(raw_csv, encoding="utf-8-sig")
    df = df.dropna(subset=["mileage_bid"]).copy()
    df["mileage_bid"]   = pd.to_numeric(df["mileage_bid"], errors="coerce")
    df["grade_year"]    = pd.to_numeric(df["grade_year"], errors="coerce")
    df["enrolled_bool"] = (df["enrolled"].astype(str).str.upper() == "Y").astype(int)
    df["professor_norm"] = df["professor"].apply(_norm_prof)
    df = df.dropna(subset=["mileage_bid"])
    return df


def load_course_meta(json_path: str = JSON_PATH,
                     raw_csv: str = RAW_CSV) -> dict[str, dict]:
    """
    Course-level fallback signals (capacity + actual competition demand).

    eta_added = total_applicants (지원자) from the most recent scraped semester.
    This is the number of students who actually BID mileage for the course,
    NOT the ETA add count from the JSON — ETA is inflated by students who
    add courses after the second-round first-come-first-served registration
    (they use ETA as a timetable, not as a competition signal).

    capacity is read from the scraped data first, falling back to the JSON.
    """
    # ── Step 1: read capacity from JSON as baseline ───────────────────────
    out = {}
    try:
        with open(json_path) as f:
            data = json.load(f)
        for _, yr in data.items():
            for _, cat in yr.items():
                for code, course in cat.items():
                    if code not in out:
                        m = course.get("metadata", {})
                        out[code] = {
                            "eta_added": 0.0,   # will be overwritten below
                            "capacity":  float(m.get("capacity", 0) or 0),
                        }
    except Exception:
        pass

    # ── Step 2: overwrite eta_added with real 지원자 count ────────────────
    # Use the most recent semester's total_applicants per course — this is
    # the actual number of students who competed in the mileage bidding round.
    try:
        df = pd.read_csv(raw_csv, encoding="utf-8-sig")
        # Most recent semester per course
        df_sorted = df.sort_values(["year", "semester"], ascending=False)
        for code, grp in df_sorted.groupby("course_code"):
            # total_applicants and capacity are repeated on every row for a
            # course-semester; take the first non-null value
            app = grp["total_applicants"].dropna()
            cap = grp["capacity"].dropna()

            if code not in out:
                out[code] = {"eta_added": 0.0, "capacity": 0.0}

            if len(app) > 0:
                out[code]["eta_added"] = float(app.iloc[0])
            if len(cap) > 0 and out[code]["capacity"] == 0.0:
                out[code]["capacity"] = float(cap.iloc[0])
    except Exception:
        pass

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Survival curve
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurvivalCurve:
    """P(enroll | bid) for one course, built from weighted applicant rows."""
    code: str
    professor: str
    bids: np.ndarray              # sorted unique bid levels
    p_enroll: np.ndarray          # P(enroll | bid >= this level)
    n_effective: float            # effective sample size (sum of weights)
    source: str                   # how it was built
    data_years: list = field(default_factory=list)
    cutoff_floor_bid: int = 0
    cutoff_floor_reason: str = ""
    exact_bids: np.ndarray = field(default_factory=lambda: np.array([]))
    exact_p_enroll: np.ndarray = field(default_factory=lambda: np.array([]))
    exact_n: np.ndarray = field(default_factory=lambda: np.array([]))

    def survival_prob_at(self, bid: int) -> float:
        """Smoothed P(enroll | historical bid >= bid)."""
        if len(self.bids) == 0:
            return 0.5
        idx = np.searchsorted(self.bids, bid, side="right") - 1
        idx = max(0, min(idx, len(self.p_enroll) - 1))
        return float(self.p_enroll[idx])

    def local_prob_at(self, bid: int, window: int = 1) -> tuple[float | None, float]:
        """
        Local same/near-bid enrollment rate.

        This is less smooth than the survival curve, but more honest for output
        wording: a student bidding 18pt should not inherit all the success of
        students who bid 30-36pt.
        """
        if len(self.exact_bids) == 0:
            return None, 0.0
        mask = (self.exact_bids >= bid - window) & (self.exact_bids <= bid + window)
        if not np.any(mask):
            return None, 0.0
        n = float(self.exact_n[mask].sum())
        if n <= 0:
            return None, 0.0
        p = float(np.average(self.exact_p_enroll[mask], weights=self.exact_n[mask]))
        return p, n

    def prob_at(self, bid: int) -> float:
        """
        Calibrated P(enroll) if you bid `bid`.

        The old display probability used only P(enroll | bid >= X), which is
        often too optimistic near cutoffs. We blend that smooth curve with
        local exact/near-bid evidence when the data is strong enough.
        """
        smooth_p = self.survival_prob_at(bid)
        local_p, local_n = self.local_prob_at(bid, window=1)
        if local_p is None:
            return smooth_p

        if local_n >= 8:
            local_weight = 0.70
        elif local_n >= 5:
            local_weight = 0.55
        elif local_n >= 3:
            local_weight = 0.35
        else:
            local_weight = 0.0

        calibrated = local_weight * local_p + (1.0 - local_weight) * smooth_p
        if self.cutoff_floor_bid and bid < self.cutoff_floor_bid:
            calibrated = min(calibrated, 0.84)
        return float(np.clip(calibrated, 0.0, 1.0))

    def min_bid_for(self, target: float) -> int:
        """
        Cheapest bid achieving P(enroll) >= target.

        If the target is UNREACHABLE (the curve plateaus below it — e.g. a
        course where year-quota caps a 3rd-year's odds at 80% no matter how
        high they bid), return the 'knee': the cheapest bid that reaches within
        2% of the curve's own maximum. Bidding past the knee wastes points.
        """
        if len(self.bids) == 0:
            return max(5, self.cutoff_floor_bid)
        for b, p in zip(self.bids, self.p_enroll):
            if p >= target:
                return max(int(b), self.cutoff_floor_bid)
        # Target unreachable → find the knee (cheapest near-max bid)
        p_max = float(self.p_enroll.max())
        for b, p in zip(self.bids, self.p_enroll):
            if p >= p_max - 0.02:
                return max(int(b), self.cutoff_floor_bid)
        return max(int(self.bids[-1]), self.cutoff_floor_bid)

    def reachable_max(self) -> float:
        """Highest win probability this course allows (curve plateau)."""
        return float(self.p_enroll.max()) if len(self.p_enroll) else 0.5


def _weighted_survival(sub: pd.DataFrame, weight_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the survival curve P(enroll | bid >= b) using sample weights.

    Key design: we compute a WEIGHTED enrollment rate at each bid level.
    The weight_col already encodes recency * year_personalisation, so a
    2022 applicant with weight 0.12 contributes 0.12 to both numerator
    and denominator — meaning 144 old rows at weight 0.12 contribute the
    same total mass as 17 recent rows at weight 1.0, which is the right
    behaviour (recent semester dominates, old semester informs).

    Monotonic non-decreasing enforced with running max — higher bid should
    never *lower* the estimated win probability.
    Minimum 3 effective weight units required at a bid level to include it.
    """
    bids = np.sort(sub["mileage_bid"].unique())
    levels, probs = [], []
    for b in bids:
        s = sub[sub["mileage_bid"] >= b]
        w = s[weight_col].values
        e = s["enrolled_bool"].values
        total_w = w.sum()
        if total_w < 3.0:          # need meaningful weight to trust the estimate
            continue
        p = float(np.dot(e, w) / total_w)
        levels.append(b)
        probs.append(p)
    if not levels:
        return np.array([]), np.array([])
    levels = np.array(levels)
    probs  = np.array(probs)
    # Enforce monotonic non-decreasing
    probs = np.maximum.accumulate(probs)
    return levels, probs


def _weighted_exact_rates(sub: pd.DataFrame, weight_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted P(enroll | exact bid=b) for local calibration."""
    levels, probs, counts = [], [], []
    for bid, grp in sub.groupby("mileage_bid"):
        w = grp[weight_col].values
        e = grp["enrolled_bool"].values
        total_w = float(w.sum())
        if total_w <= 0:
            continue
        levels.append(float(bid))
        probs.append(float(np.dot(e, w) / total_w))
        counts.append(total_w)
    if not levels:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(levels)
    return np.array(levels)[order], np.array(probs)[order], np.array(counts)[order]


def _recent_year_cutoff_floor(
    rows: pd.DataFrame,
    student_year: int | None,
    curve_max: float,
) -> tuple[int, str]:
    """
    Conservative exact-bid guardrail.

    The survival curve estimates P(enroll | historical bid >= b). That is smooth
    and monotonic, but it can make an exact 18pt recommendation look safe because
    many 36pt bidders succeeded. If same-year, same-student-year applicants lost
    at or above a bid, do not recommend below the next point as "safe".
    """
    if student_year is None or len(rows) == 0:
        return 0, ""

    latest_year = rows["year"].max()
    latest_rows = rows[rows["year"] == latest_year]
    cap = latest_rows["capacity"].dropna()
    app = latest_rows["total_applicants"].dropna()
    demand_ratio = 0.0
    if len(cap) > 0 and len(app) > 0 and float(cap.iloc[0]) > 0:
        demand_ratio = float(app.iloc[0]) / float(cap.iloc[0])

    # Do not turn quota/tie-break noise in easy courses into scary cutoffs,
    # and do not force a high cutoff when the curve itself says the target is
    # unreachable no matter how much the student bids.
    if demand_ratio < 1.25 or curve_max < 0.90:
        return 0, ""

    recent = rows[
        (rows["year"] == latest_year)
        & (rows["grade_year"] == student_year)
        & rows["mileage_bid"].notna()
    ].copy()
    if len(recent) < 8:
        return 0, ""

    lost = recent[recent["enrolled_bool"] == 0]
    if lost.empty:
        return 0, ""

    max_lost = int(lost["mileage_bid"].max())
    floor = min(MAX_BID, max_lost + 1)
    reason = (
        f"Recent year-{student_year} history includes non-enrolled applicants at "
        f"{max_lost}pt, so bids below ~{floor}pt are not treated as safe."
    )
    return floor, reason


def build_curve(
    df: pd.DataFrame,
    code: str,
    professor_2026: str,
    student_year: int | None = None,
    meta: dict | None = None,
) -> SurvivalCurve:
    """
    Build the best available survival curve for a course, professor-aware.
    When student_year is given AND there's enough year-specific data, we build
    a year-filtered curve — this captures dynamics like "year-4 students have
    worse odds for RL because 95/142 applicants are year-4."
    """
    prof = _norm_prof(professor_2026)
    course_rows = df[df["course_code"] == code].copy()
    course_rows["recency_w"] = course_rows["year"].map(RECENCY_WEIGHTS).fillna(0.1)

    # ── Tier 1: same course + same professor ──────────────────────────────
    same_prof = course_rows[course_rows["professor_norm"] == prof]
    pool = same_prof if len(same_prof) >= 8 else course_rows

    if len(pool) < 8:
        # No usable history → demand proxy fallback
        meta = meta or {}
        info = meta.get(code, {})
        eta = info.get("eta_added", 0)
        cap = info.get("capacity", 0)
        if cap <= 0 and len(course_rows) > 0:
            cap = float(course_rows["capacity"].dropna().iloc[0]) if course_rows["capacity"].notna().any() else 0
        demand = (eta / cap) if cap > 0 else 1.2
        return _synthetic_curve(code, professor_2026, demand)

    source = "same_professor" if len(same_prof) >= 8 else "same_course_any_prof"

    # ── Year-specific curve when enough data exists ───────────────────────
    # If we have >=15 rows for this student's year, build a year-filtered
    # curve. This properly captures year-quota effects (e.g. year-4 students
    # competing against each other for limited seats → lower odds).
    #
    # IMPORTANT: when the most recent semester alone has enough year-specific
    # data (>=15), use ONLY that semester. With multi-year gaps (e.g. 2022→2026),
    # competition levels change so drastically that blending is actively harmful.
    if student_year is not None:
        year_rows = pool[pool["grade_year"] == student_year]
        if len(year_rows) >= 15:
            # Try most-recent-semester-only first
            most_recent_yr = year_rows["year"].max()
            recent_yr_rows = year_rows[year_rows["year"] == most_recent_yr]
            if len(recent_yr_rows) >= 15:
                yr = recent_yr_rows.copy()
                yr["w"] = 1.0  # single semester, no recency blend needed
            else:
                yr = year_rows.copy()
                yr["w"] = yr["recency_w"]
            bids, probs = _weighted_survival(yr, "w")
            if len(bids) > 0:
                exact_bids, exact_probs, exact_counts = _weighted_exact_rates(yr, "w")
                cutoff_floor, cutoff_reason = _recent_year_cutoff_floor(
                    pool, student_year, float(probs.max())
                )
                return SurvivalCurve(
                    code=code, professor=professor_2026, bids=bids,
                    p_enroll=probs, n_effective=float(yr["w"].sum()),
                    source=source,
                    data_years=sorted(yr["year"].unique().tolist()),
                    cutoff_floor_bid=cutoff_floor,
                    cutoff_floor_reason=cutoff_reason,
                    exact_bids=exact_bids,
                    exact_p_enroll=exact_probs,
                    exact_n=exact_counts,
                )

    # ── Fallback: all years blended with recency weighting ────────────────
    p = pool.copy()
    if len(same_prof) >= 8:
        # Downweight other-professor rows
        p["w"] = p["recency_w"] * np.where(p["professor_norm"] == prof, 1.0, 0.6)
    else:
        p["w"] = p["recency_w"]
    bids, probs = _weighted_survival(p, "w")
    if len(bids) > 0:
        n_profs = p["professor_norm"].nunique()
        exact_bids, exact_probs, exact_counts = _weighted_exact_rates(p, "w")
        cutoff_floor, cutoff_reason = _recent_year_cutoff_floor(
            pool, student_year, float(probs.max())
        )
        return SurvivalCurve(
            code=code, professor=professor_2026, bids=bids, p_enroll=probs,
            n_effective=float(p["w"].sum()),
            source=source if n_profs <= 1 else "same_course_any_prof",
            data_years=sorted(p["year"].unique().tolist()),
            cutoff_floor_bid=cutoff_floor,
            cutoff_floor_reason=cutoff_reason,
            exact_bids=exact_bids,
            exact_p_enroll=exact_probs,
            exact_n=exact_counts,
        )

    # ── Last resort: demand proxy ─────────────────────────────────────────
    meta = meta or {}
    info = meta.get(code, {})
    eta = info.get("eta_added", 0)
    cap = info.get("capacity", 0)
    demand = (eta / cap) if cap > 0 else 1.2
    return _synthetic_curve(code, professor_2026, demand)


def _synthetic_curve(code: str, professor: str, demand_ratio: float) -> SurvivalCurve:
    """
    When there is no applicant history, approximate a curve from how
    oversubscribed the course is (ETA adds / capacity).
    """
    # midpoint bid where P crosses 0.9, calibrated from observed courses
    if demand_ratio <= 1.05:
        knee = 2
    elif demand_ratio <= 1.20:
        knee = 6
    elif demand_ratio <= 1.40:
        knee = 13
    elif demand_ratio <= 1.60:
        knee = 20
    else:
        knee = 28
    bids = np.arange(1, 37)
    # logistic-ish rise to ~0.95 around the knee
    p = 0.55 + 0.40 / (1 + np.exp(-(bids - knee) / 3.0))
    p = np.clip(p, 0.3, 0.97)
    p = np.maximum.accumulate(p)
    return SurvivalCurve(
        code=code, professor=professor, bids=bids, p_enroll=p,
        n_effective=0.0, source="demand_proxy", data_years=[],
        exact_bids=bids, exact_p_enroll=p, exact_n=np.ones_like(bids),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Honest confidence
# ─────────────────────────────────────────────────────────────────────────────

def confidence_label(curve: SurvivalCurve) -> tuple[str, float]:
    """
    Confidence reflects how much real data backs the curve, NOT how sure we are
    the student gets in. (Those are different things — getting-in probability is
    the survival curve value; confidence is about data quality.)
    """
    if curve.source == "same_professor" and curve.n_effective >= 30:
        return "High", 85.0
    if curve.source == "same_professor":
        return "High", 72.0
    if curve.source in ("same_course", "same_course_any_prof"):
        if curve.n_effective >= 40:
            return "Medium", 60.0
        return "Medium", 48.0
    return "Low", 30.0




