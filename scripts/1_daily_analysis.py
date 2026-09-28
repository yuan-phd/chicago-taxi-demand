"""
Daily demand analysis for NNS (area 8).
Examines day-of-week patterns, specific date impacts, trend structure,
and autocorrelation to inform model design.

Usage: python scripts/1_daily_analysis.py
Input:  data/derived/area_day.csv
Output: outputs/daily_analysis.txt (printed to stdout, redirect if needed)
"""

import csv
from datetime import date, timedelta
from collections import defaultdict

# =============================================================
# LOAD DATA
# =============================================================

def load_nns_daily():
    rows = []
    with open("data/derived/area_day.csv") as f:
        for r in csv.DictReader(f):
            if int(r["pickup_community_area"]) == 8:
                d = date.fromisoformat(r["date"])
                rows.append({
                    "date": d,
                    "year": d.year,
                    "month": d.month,
                    "iso_year": int(r["iso_year"]),
                    "iso_week": int(r["iso_week"]),
                    "weekday": d.weekday(),  # 0=Mon, 6=Sun
                    "weekday_name": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d.weekday()],
                    "trips": int(r["trip_count"]),
                })
    rows.sort(key=lambda x: x["date"])
    return rows

DAY_NAMES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

daily = load_nns_daily()

# Filter out partial days (< 100 trips — likely reporting artifacts)
daily = [r for r in daily if r["trips"] >= 100]

print(f"NNS daily data: {len(daily)} days")
print(f"Date range: {daily[0]['date']} to {daily[-1]['date']}")

# =============================================================
# STEP 1A: DAY-OF-WEEK PATTERN
# =============================================================

print(f"\n{'='*70}")
print("STEP 1A: DAY-OF-WEEK PATTERN")
print(f"{'='*70}")

dow_trips = defaultdict(list)
for r in daily:
    dow_trips[r["weekday"]].append(r["trips"])

overall_mean = sum(r["trips"] for r in daily) / len(daily)

print(f"\n{'Day':<6} {'Count':>6} {'Mean':>8} {'Median':>8} {'StdDev':>8} {'vs Avg':>8} {'CV':>6}")
print("-" * 58)
for dow in range(7):
    vals = dow_trips[dow]
    n = len(vals)
    mean = sum(vals) / n
    sorted_vals = sorted(vals)
    median = sorted_vals[n // 2]
    std = (sum((v - mean)**2 for v in vals) / n) ** 0.5
    cv = std / mean * 100
    vs_avg = (mean - overall_mean) / overall_mean * 100
    print(f"{DAY_NAMES[dow]:<6} {n:>6} {mean:>8,.0f} {median:>8,.0f} {std:>8,.0f} {vs_avg:>+7.1f}% {cv:>5.1f}%")

print(f"\nOverall daily mean: {overall_mean:,.0f}")

# Weekday vs weekend
weekday_vals = [r["trips"] for r in daily if r["weekday"] < 5]
weekend_vals = [r["trips"] for r in daily if r["weekday"] >= 5]
wd_mean = sum(weekday_vals) / len(weekday_vals)
we_mean = sum(weekend_vals) / len(weekend_vals)
print(f"Weekday mean: {wd_mean:,.0f} ({len(weekday_vals)} days)")
print(f"Weekend mean: {we_mean:,.0f} ({len(weekend_vals)} days)")
print(f"Weekend/Weekday ratio: {we_mean/wd_mean:.3f}")

# =============================================================
# STEP 1B: TREND BY DAY-OF-WEEK
# =============================================================

print(f"\n{'='*70}")
print("STEP 1B: TREND BY DAY-OF-WEEK (is growth uniform?)")
print(f"{'='*70}")

# Compare same-day-of-week means: 2024 H1 vs 2025 H1 vs 2026 H1
periods = [
    ("2024 H1", lambda r: r["year"]==2024 and r["month"]<=6),
    ("2025 H1", lambda r: r["year"]==2025 and r["month"]<=6),
    ("2026 H1", lambda r: r["year"]==2026 and r["month"]<=6),
]

print(f"\n{'Day':<6}", end="")
for label, _ in periods:
    print(f" {label:>10}", end="")
print(f" {'24→25':>8} {'25→26':>8}")
print("-" * 65)

for dow in range(7):
    print(f"{DAY_NAMES[dow]:<6}", end="")
    means = []
    for label, filt in periods:
        vals = [r["trips"] for r in daily if filt(r) and r["weekday"]==dow]
        m = sum(vals)/len(vals) if vals else 0
        means.append(m)
        print(f" {m:>10,.0f}", end="")
    g1 = (means[1]-means[0])/means[0]*100 if means[0] else 0
    g2 = (means[2]-means[1])/means[1]*100 if means[1] else 0
    print(f" {g1:>+7.1f}% {g2:>+7.1f}%")

# Total weekday vs weekend growth
for label_type, dow_range in [("Weekday", range(5)), ("Weekend", range(5,7))]:
    means = []
    for _, filt in periods:
        vals = [r["trips"] for r in daily if filt(r) and r["weekday"] in dow_range]
        means.append(sum(vals)/len(vals) if vals else 0)
    g1 = (means[1]-means[0])/means[0]*100 if means[0] else 0
    g2 = (means[2]-means[1])/means[1]*100 if means[1] else 0
    print(f"{label_type:<6} {means[0]:>10,.0f} {means[1]:>10,.0f} {means[2]:>10,.0f} {g1:>+7.1f}% {g2:>+7.1f}%")

# =============================================================
# STEP 1C: AUTOCORRELATION (lag 1-14)
# =============================================================

print(f"\n{'='*70}")
print("STEP 1C: AUTOCORRELATION STRUCTURE")
print(f"{'='*70}")

trips_series = [r["trips"] for r in daily]
n_s = len(trips_series)
mean_s = sum(trips_series) / n_s
var_s = sum((x - mean_s)**2 for x in trips_series) / n_s

print(f"\n{'Lag':>5} {'ACF':>8} {'Interpretation'}")
print("-" * 45)
for lag in [1, 2, 3, 6, 7, 8, 13, 14]:
    ac = sum((trips_series[i]-mean_s)*(trips_series[i+lag]-mean_s)
             for i in range(n_s-lag)) / ((n_s-lag)*var_s)
    sig_threshold = 1.96 / (n_s ** 0.5)
    sig = "***" if abs(ac) > 3*sig_threshold else ("**" if abs(ac) > 2*sig_threshold else ("*" if abs(ac) > sig_threshold else ""))
    interp = ""
    if lag == 1: interp = "yesterday"
    elif lag == 7: interp = "same day last week"
    elif lag == 14: interp = "same day 2 weeks ago"
    print(f"{lag:>5} {ac:>+7.3f} {sig:<4} {interp}")

print(f"\n(significance: * p<0.05, ** p<0.01, *** p<0.001)")

# =============================================================
# STEP 2A: SPECIFIC DATE IMPACTS — HOLIDAYS
# =============================================================

print(f"\n{'='*70}")
print("STEP 2A: SPECIFIC DATE IMPACTS")
print(f"{'='*70}")

# Build lookup
date_lookup = {r["date"]: r["trips"] for r in daily}

def same_dow_neighbors(d, weeks_back=1, weeks_forward=1):
    """Get same-day-of-week values from surrounding weeks."""
    vals = []
    for w in range(-weeks_back, weeks_forward+1):
        if w == 0: continue
        nd = d + timedelta(weeks=w)
        if nd in date_lookup:
            vals.append(date_lookup[nd])
    return vals

def analyze_date(d, label):
    """Analyze a specific date vs same-day-of-week neighbors."""
    trips = date_lookup.get(d)
    if trips is None:
        return None
    neighbors = same_dow_neighbors(d, 2, 2)
    if not neighbors:
        return None
    avg_n = sum(neighbors) / len(neighbors)
    impact = (trips - avg_n) / avg_n * 100
    return {
        "date": d,
        "label": label,
        "day": DAY_NAMES[d.weekday()],
        "trips": trips,
        "neighbor_avg": avg_n,
        "impact_pct": impact,
        "n_neighbors": len(neighbors),
    }

# Key dates to test
test_dates = [
    # Rosh Hashanah 2024
    (date(2024, 10, 2), "Rosh Hashanah 2024 Day 1"),
    (date(2024, 10, 3), "Rosh Hashanah 2024 Day 2"),
    (date(2024, 10, 4), "Rosh Hashanah 2024 Day 3"),
    # Yom Kippur 2024
    (date(2024, 10, 11), "Yom Kippur 2024 Day 1"),
    (date(2024, 10, 12), "Yom Kippur 2024 Day 2"),
    # Rosh Hashanah 2025
    (date(2025, 9, 22), "Rosh Hashanah 2025 Day 1"),
    (date(2025, 9, 23), "Rosh Hashanah 2025 Day 2"),
    (date(2025, 9, 24), "Rosh Hashanah 2025 Day 3"),
    # Yom Kippur 2025
    (date(2025, 10, 1), "Yom Kippur 2025 Day 1"),
    (date(2025, 10, 2), "Yom Kippur 2025 Day 2"),
    # Thanksgiving
    (date(2024, 11, 28), "Thanksgiving 2024"),
    (date(2024, 11, 29), "Black Friday 2024"),
    (date(2025, 11, 27), "Thanksgiving 2025"),
    (date(2025, 11, 28), "Black Friday 2025"),
    # Christmas
    (date(2024, 12, 25), "Christmas 2024"),
    (date(2025, 12, 25), "Christmas 2025"),
    # New Year
    (date(2024, 1, 1), "New Year's Day 2024"),
    (date(2025, 1, 1), "New Year's Day 2025"),
    (date(2026, 1, 1), "New Year's Day 2026"),
    # July 4th
    (date(2024, 7, 4), "July 4th 2024"),
    (date(2025, 7, 4), "July 4th 2025"),
    # Labor Day
    (date(2024, 9, 2), "Labor Day 2024"),
    (date(2025, 9, 1), "Labor Day 2025"),
    # Halloween
    (date(2024, 10, 31), "Halloween 2024"),
    (date(2025, 10, 31), "Halloween 2025"),
    # W44 2024 investigation (Oct 28 - Nov 3)
    (date(2024, 10, 28), "W44 2024 Mon"),
    (date(2024, 10, 29), "W44 2024 Tue"),
    (date(2024, 10, 30), "W44 2024 Wed"),
    (date(2024, 10, 31), "W44 2024 Thu (Halloween)"),
    (date(2024, 11, 1), "W44 2024 Fri"),
    (date(2024, 11, 2), "W44 2024 Sat"),
    (date(2024, 11, 3), "W44 2024 Sun"),
]

print(f"\n{'Date':<12} {'Day':<5} {'Label':<28} {'Trips':>7} {'Nbr Avg':>8} {'Impact':>8}")
print("-" * 75)

current_group = ""
for d, label in test_dates:
    # Group separator
    group = label.split(" 202")[0] if "W44" not in label else "W44 2024"
    if group != current_group:
        if current_group:
            print()
        current_group = group
    
    result = analyze_date(d, label)
    if result:
        print(f"{str(d):<12} {result['day']:<5} {label:<28} "
              f"{result['trips']:>7,} {result['neighbor_avg']:>8,.0f} "
              f"{result['impact_pct']:>+7.1f}%")

# =============================================================
# STEP 2B: TOP ANOMALIES — LARGEST POSITIVE AND NEGATIVE DEVIATIONS
# =============================================================

print(f"\n{'='*70}")
print("STEP 2B: TOP 20 ANOMALOUS DAYS (vs same-day-of-week neighbors)")
print(f"{'='*70}")

anomalies = []
for r in daily:
    neighbors = same_dow_neighbors(r["date"], 2, 2)
    if len(neighbors) >= 2:
        avg_n = sum(neighbors) / len(neighbors)
        impact = (r["trips"] - avg_n) / avg_n * 100
        anomalies.append({
            "date": r["date"],
            "day": r["weekday_name"],
            "trips": r["trips"],
            "neighbor_avg": avg_n,
            "impact_pct": impact,
        })

# Sort by absolute impact
anomalies.sort(key=lambda x: abs(x["impact_pct"]), reverse=True)

print(f"\n{'Rank':<6} {'Date':<12} {'Day':<5} {'Trips':>7} {'Nbr Avg':>8} {'Impact':>8}")
print("-" * 52)
for i, a in enumerate(anomalies[:20]):
    print(f"{i+1:<6} {str(a['date']):<12} {a['day']:<5} "
          f"{a['trips']:>7,} {a['neighbor_avg']:>8,.0f} {a['impact_pct']:>+7.1f}%")

# =============================================================
# STEP 1D: MONTH-OF-YEAR PATTERN
# =============================================================

print(f"\n{'='*70}")
print("STEP 1D: MONTHLY PATTERN")
print(f"{'='*70}")

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
print(f"\n{'Month':<6} {'Mean':>8} {'vs Avg':>8}")
print("-" * 25)
for m in range(1, 13):
    vals = [r["trips"] for r in daily if r["month"] == m]
    if vals:
        mean = sum(vals) / len(vals)
        vs = (mean - overall_mean) / overall_mean * 100
        print(f"{month_names[m-1]:<6} {mean:>8,.0f} {vs:>+7.1f}%")

# =============================================================
# STEP 1E: MONTH BOUNDARY CHECK (payday effect)
# =============================================================

print(f"\n{'='*70}")
print("STEP 1E: MONTH BOUNDARY CHECK (day-of-month effect)")
print(f"{'='*70}")

dom_trips = defaultdict(list)
for r in daily:
    dom_trips[r["date"].day].append(r["trips"])

print(f"\n{'DoM':>4} {'Mean':>8} {'vs Avg':>8} {'N':>5}")
print("-" * 30)
for dom in range(1, 32):
    vals = dom_trips.get(dom, [])
    if vals:
        mean = sum(vals) / len(vals)
        vs = (mean - overall_mean) / overall_mean * 100
        flag = " ◄" if abs(vs) > 10 else ""
        print(f"{dom:>4} {mean:>8,.0f} {vs:>+7.1f}% {len(vals):>5}{flag}")

print(f"\nNote: day-of-month effect is confounded with day-of-week.")
print(f"Large deviations on specific days may reflect which weekdays they fell on.")
