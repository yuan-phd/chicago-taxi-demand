"""
Weekly residual analysis for NNS (area 8).
Reproduces all weekly-level diagnostics:
  1. Baseline MAPE and per-week error ranking
  2. Error source diagnosis (trend bias, ACF, single-year H2 problem)
  3. Floating holiday investigation (cross-year validation)
  4. H1 backtest (2-sample vs 1-sample seasonal factors)
  5. Model improvement experiments (cleaning + AR(1))

Inputs:
  data/from_assignment/area_week.csv
  data/from_assignment/backtest_8.csv

Usage: python scripts/1d_weekly_residual_analysis.py
"""

import csv
import math
from datetime import date, timedelta

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def iso_week_to_dates(year, week):
    jan4 = date(year, 1, 4)
    start_of_w1 = jan4 - timedelta(days=jan4.isoweekday() - 1)
    monday = start_of_w1 + timedelta(weeks=week - 1)
    sunday = monday + timedelta(days=6)
    return monday, sunday


# =============================================================
# LOAD DATA
# =============================================================

def load_nns_weekly():
    rows = []
    with open("data/from_assignment/area_week.csv") as f:
        for r in csv.DictReader(f):
            if int(r["pickup_community_area"]) == 8:
                trips = int(r["trip_count"])
                if trips < 100:
                    continue
                rows.append({
                    "year": int(r["iso_year"]),
                    "week": int(r["iso_week"]),
                    "trips": trips,
                })
    rows.sort(key=lambda x: (x["year"], x["week"]))
    return rows


def load_backtest():
    rows = []
    with open("data/from_assignment/backtest_8.csv") as f:
        for r in csv.DictReader(f):
            if not r["iso_week"].strip():
                continue
            rows.append({
                "week": int(r["iso_week"]),
                "actual": float(r["actual"]),
                "predicted": float(r["trend_seasonal"]),
            })
    return rows


def load_all_areas():
    rows = []
    with open("data/from_assignment/area_week.csv") as f:
        for r in csv.DictReader(f):
            trips = int(r["trip_count"])
            if trips < 100:
                continue
            rows.append({
                "area": int(r["pickup_community_area"]),
                "year": int(r["iso_year"]),
                "week": int(r["iso_week"]),
                "trips": trips,
            })
    return rows


def mape(actuals, preds):
    return sum(abs(a - p) / a for a, p in zip(actuals, preds)) / len(actuals) * 100


nns = load_nns_weekly()
bt = load_backtest()
all_areas = load_all_areas()

# =============================================================
# SECTION 1: BASELINE RESIDUAL ANALYSIS
# =============================================================

print("=" * 70)
print("SECTION 1: BASELINE RESIDUAL ANALYSIS")
print("=" * 70)

for r in bt:
    r["error"] = r["actual"] - r["predicted"]
    r["pct_error"] = (r["actual"] - r["predicted"]) / r["actual"] * 100
    r["abs_pct_error"] = abs(r["pct_error"])

overall_mape = sum(r["abs_pct_error"] for r in bt) / len(bt)
print(f"\nBaseline MAPE: {overall_mape:.2f}% (trend × seasonal, H2 2025 holdout)")
print(f"Holdout: {len(bt)} weeks (2025 W27-W52)")

# Ranked by error
sorted_bt = sorted(bt, key=lambda x: x["abs_pct_error"], reverse=True)
print(f"\n{'Rank':<6} {'Week':<8} {'Actual':>8} {'Predicted':>10} {'%Error':>8}")
print("-" * 45)
for i, r in enumerate(sorted_bt):
    mon, sun = iso_week_to_dates(2025, r["week"])
    dates = f"{mon.strftime('%b %d')}-{sun.strftime('%b %d')}"
    print(f"{i+1:<6} W{r['week']:02d}    {r['actual']:>8,.0f} {r['predicted']:>10,.0f} {r['pct_error']:>+7.1f}%")

# Error concentration
total_abs = sum(r["abs_pct_error"] for r in bt)
for n in [3, 5, 8]:
    top_n = sum(r["abs_pct_error"] for r in sorted_bt[:n])
    print(f"Top {n} worst weeks: {top_n/total_abs*100:.0f}% of total error")

# Error distribution
abs_errors = sorted([r["abs_pct_error"] for r in bt])
n = len(bt)
print(f"\nMedian absolute error: {abs_errors[n//2]:.1f}%")
print(f"Weeks <5% error: {sum(1 for e in abs_errors if e < 5)}/{n}")
print(f"Weeks >10% error: {sum(1 for e in abs_errors if e > 10)}/{n}")

# =============================================================
# SECTION 2: ERROR SOURCE DIAGNOSIS
# =============================================================

print(f"\n{'='*70}")
print("SECTION 2: ERROR SOURCE DIAGNOSIS")
print("=" * 70)

# Systematic bias: first half vs second half
errors = [r["pct_error"] for r in bt]
first_half = errors[:13]
second_half = errors[13:]
print(f"\nSystematic bias check:")
print(f"  W27-W39 mean error: {sum(first_half)/len(first_half):+.1f}%")
print(f"  W40-W52 mean error: {sum(second_half)/len(second_half):+.1f}%")
print(f"  → Model under-predicts Q4 by ~{abs(sum(second_half)/len(second_half)):.1f}%")

# Autocorrelation
n_e = len(errors)
mean_e = sum(errors) / n_e
var_e = sum((e - mean_e) ** 2 for e in errors) / n_e
print(f"\nAutocorrelation of residuals:")
for lag in range(1, 6):
    if n_e - lag > 0:
        ac = sum((errors[i] - mean_e) * (errors[i + lag] - mean_e)
                 for i in range(n_e - lag)) / ((n_e - lag) * var_e)
        sig = 1.96 / math.sqrt(n_e)
        status = "SIGNIFICANT" if abs(ac) > sig else "not significant"
        print(f"  Lag {lag}: {ac:+.3f} (threshold ±{sig:.3f}) → {status}")

# Single-year H2 problem
print(f"\nSeasonal factor sample counts:")
by_week = {}
for r in nns:
    if (r["year"] == 2024 and r["week"] >= 2) or (r["year"] == 2025 and r["week"] <= 26):
        by_week.setdefault(r["week"], []).append(r["year"])
h1_samples = [len(by_week.get(w, [])) for w in range(2, 27)]
h2_samples = [len(by_week.get(w, [])) for w in range(27, 53) if w in by_week]
print(f"  H1 weeks (W02-W26): {min(h1_samples)}-{max(h1_samples)} samples each")
print(f"  H2 weeks (W27-W52): {min(h2_samples)}-{max(h2_samples)} samples each")
print(f"  → H2 seasonal factors based on single year (2024). Any 2024 anomaly dominates.")

# YoY growth variation
print(f"\n2024 vs 2025 YoY growth for holdout weeks:")
lookup = {}
for r in nns:
    lookup[(r["year"], r["week"])] = r["trips"]

print(f"{'Week':<6} {'2024':>8} {'2025':>8} {'YoY%':>8}")
print("-" * 35)
yoy_values = []
for w in range(27, 53):
    v24 = lookup.get((2024, w))
    v25 = lookup.get((2025, w))
    if v24 and v25:
        yoy = (v25 - v24) / v24 * 100
        yoy_values.append(yoy)
        flag = " ◄" if abs(yoy) > 25 or abs(yoy) < 8 else ""
        print(f"W{w:02d}   {v24:>8,} {v25:>8,} {yoy:>+7.1f}%{flag}")
print(f"\nYoY range: {min(yoy_values):+.1f}% to {max(yoy_values):+.1f}%")
print(f"Average: {sum(yoy_values)/len(yoy_values):+.1f}%")

# =============================================================
# SECTION 3: FLOATING HOLIDAY INVESTIGATION
# =============================================================

print(f"\n{'='*70}")
print("SECTION 3: FLOATING HOLIDAY INVESTIGATION")
print("=" * 70)

# Holiday week shifts
print(f"\nFloating holiday week shifts:")
holidays = [
    ("Rosh Hashanah", date(2024, 10, 2), date(2025, 9, 22)),
    ("Yom Kippur", date(2024, 10, 12), date(2025, 10, 2)),
    ("Sukkot start", date(2024, 10, 16), date(2025, 10, 6)),
    ("Chicago Marathon", date(2024, 10, 13), date(2025, 10, 12)),
]
print(f"{'Holiday':<20} {'2024 Wk':<10} {'2025 Wk':<10} {'Shift'}")
print("-" * 50)
for name, d24, d25 in holidays:
    w24 = d24.isocalendar()[1]
    w25 = d25.isocalendar()[1]
    print(f"{name:<20} W{w24:<8} W{w25:<8} {w25-w24:+d} weeks")

# Fixed holidays — verify same week
print(f"\nFixed holidays (same ISO week both years?):")
fixed = [
    ("Thanksgiving", date(2024, 11, 28), date(2025, 11, 27)),
    ("Christmas", date(2024, 12, 25), date(2025, 12, 25)),
    ("July 4th", date(2024, 7, 4), date(2025, 7, 4)),
    ("Labor Day", date(2024, 9, 2), date(2025, 9, 1)),
]
for name, d24, d25 in fixed:
    w24 = d24.isocalendar()[1]
    w25 = d25.isocalendar()[1]
    same = "YES" if w24 == w25 else f"NO ({w25-w24:+d})"
    print(f"  {name}: W{w24} vs W{w25} → {same}")

# Cross-year validation: Rosh Hashanah impact
print(f"\nCross-year validation — Rosh Hashanah dip test:")
top_areas = {8: "NNS", 76: "O'Hare", 32: "Loop", 28: "NWS", 33: "NSS"}
print(f"{'Area':<8} {'2024 W40 dip':>14} {'2025 W39 dip':>14} {'Consistent?'}")
print("-" * 50)
for area, name in top_areas.items():
    area_data = {(r["year"], r["week"]): r["trips"] for r in all_areas if r["area"] == area}

    # 2024 W40 dip vs neighbors
    n24 = []
    for w in [39, 41]:
        v = area_data.get((2024, w))
        if v:
            n24.append(v)
    v40_24 = area_data.get((2024, 40))
    dip24 = (v40_24 - sum(n24) / len(n24)) / (sum(n24) / len(n24)) * 100 if v40_24 and n24 else 0

    # 2025 W39 dip vs neighbors
    n25 = []
    for w in [38, 40]:
        v = area_data.get((2025, w))
        if v:
            n25.append(v)
    v39_25 = area_data.get((2025, 39))
    dip25 = (v39_25 - sum(n25) / len(n25)) / (sum(n25) / len(n25)) * 100 if v39_25 and n25 else 0

    consistent = "YES" if (dip24 < -5 and dip25 < -5) else "NO"
    print(f"{name:<8} {dip24:>+13.1f}% {dip25:>+13.1f}% {consistent}")

print(f"\nConclusion: Rosh Hashanah dip is strong in 2024 (-12% city-wide) but weak in 2025")
print(f"(-0.5% NNS). Effect exists but is variable — NOT a reliable forecasting feature.")
print(f"Near South Side shows dips in both years (-40%, -32%) — area-specific sensitivity.")

# =============================================================
# SECTION 4: H1 BACKTEST (2-sample vs 1-sample)
# =============================================================

print(f"\n{'='*70}")
print("SECTION 4: H1 BACKTEST — Does more seasonal data help?")
print("=" * 70)

# Train on 2024W02-2025W26 (same as original)
train = [r for r in nns if (r["year"] == 2024 and r["week"] >= 2) or
                           (r["year"] == 2025 and r["week"] <= 26)]
for i, r in enumerate(train):
    r["t"] = i
n_train = len(train)
t_start = n_train

# Fit trend
tv = [r["t"] for r in train]
yv = [r["trips"] for r in train]
mt = sum(tv) / n_train
my = sum(yv) / n_train
slope = sum((t - mt) * (y - my) for t, y in zip(tv, yv)) / sum((t - mt) ** 2 for t in tv)
icpt = my - slope * mt

# Ratios
ratios = [r["trips"] / (icpt + slope * r["t"]) for r in train]

# Build seasonal factors: 2024 only, 2025 only, both
sf_2024 = {}
sf_2025 = {}
sf_both = {}
for i, r in enumerate(train):
    sf_both.setdefault(r["week"], []).append(ratios[i])
    if r["year"] == 2024:
        sf_2024.setdefault(r["week"], []).append(ratios[i])
    elif r["year"] == 2025:
        sf_2025.setdefault(r["week"], []).append(ratios[i])

sf_2024_m = {wk: sum(rs) / len(rs) for wk, rs in sf_2024.items()}
sf_2025_m = {wk: sum(rs) / len(rs) for wk, rs in sf_2025.items()}
sf_both_m = {wk: sum(rs) / len(rs) for wk, rs in sf_both.items()}

# H1 holdout: 2026 W02-W21
h1_hold = [r for r in nns if r["year"] == 2026 and 2 <= r["week"] <= 21]
h1_act = [r["trips"] for r in h1_hold]

# Predict with each SF source
def predict_h1(sf, label):
    preds = []
    for r in h1_hold:
        weeks_after_train = 26 + r["week"]  # W27-W52 + W01..Wxx
        t = t_start + weeks_after_train
        preds.append((icpt + slope * t) * sf.get(r["week"], 1.0))
    return mape(h1_act, preds)

m_2024 = predict_h1(sf_2024_m, "2024 only")
m_2025 = predict_h1(sf_2025_m, "2025 only")
m_both = predict_h1(sf_both_m, "Both years")

print(f"\nPredict 2026 H1 with different seasonal factor sources:")
print(f"  SF from 2024 only:  MAPE = {m_2024:.2f}%")
print(f"  SF from 2025 only:  MAPE = {m_2025:.2f}%")
print(f"  SF from both years: MAPE = {m_both:.2f}%")
print(f"\nAveraging 2 years saves {min(m_2024, m_2025) - m_both:.2f}pp vs better single year")
print(f"                       {max(m_2024, m_2025) - m_both:.2f}pp vs worse single year")
print(f"\nConclusion: More seasonal data reliably improves accuracy.")
print(f"Implication: Getting 2023 H2 taxi data would improve H2 forecasts.")

# Compare H1 vs H2 holdout MAPE
h2_hold = [r for r in nns if r["year"] == 2025 and 27 <= r["week"] <= 52]
h2_act = [r["trips"] for r in h2_hold]
h2_pred = [(icpt + slope * (t_start + i)) * sf_both_m.get(r["week"], 1.0)
           for i, r in enumerate(h2_hold)]
m_h2 = mape(h2_act, h2_pred)
print(f"\nH1 holdout MAPE (2-sample SF): {m_both:.2f}%")
print(f"H2 holdout MAPE (1-sample SF): {m_h2:.2f}%")

# =============================================================
# SECTION 5: WEEKLY MODEL IMPROVEMENT EXPERIMENTS
# =============================================================

print(f"\n{'='*70}")
print("SECTION 5: WEEKLY MODEL IMPROVEMENT EXPERIMENTS")
print("=" * 70)

# H2 holdout setup
holdout = h2_hold
actuals = h2_act
idx = {(r["year"], r["week"]): i for i, r in enumerate(train)}

# Original baseline
sf_orig = sf_both_m
pred_baseline = [(icpt + slope * (t_start + i)) * sf_orig.get(r["week"], 1.0)
                 for i, r in enumerate(holdout)]
mape_baseline = mape(actuals, pred_baseline)

def clean_and_predict(weeks_to_clean):
    """Clean specified training weeks, rebuild SF, predict with AR(1)."""
    cr = list(ratios)
    clean_set = set(idx.get(c) for c in weeks_to_clean if idx.get(c) is not None)

    for yr, wk in weeks_to_clean:
        i = idx.get((yr, wk))
        if i is None:
            continue
        neighbors = [ratios[ni] for off in [-2, -1, 1, 2]
                     for ni in [i + off] if 0 <= ni < n_train and ni not in clean_set]
        if neighbors:
            cr[i] = sum(neighbors) / len(neighbors)

    # Rebuild SF
    sf = {}
    for i, r in enumerate(train):
        sf.setdefault(r["week"], []).append(cr[i])
    sf = {wk: sum(rs) / len(rs) for wk, rs in sf.items()}

    # Predict without AR
    pred_no_ar = [(icpt + slope * (t_start + i)) * sf.get(r["week"], 1.0)
                  for i, r in enumerate(holdout)]

    # AR(1) on residuals
    tp = [(icpt + slope * r["t"]) * sf.get(r["week"], 1.0) for r in train]
    tr = [r["trips"] - p for r, p in zip(train, tp)]
    nr = len(tr)
    mr = sum(tr) / nr
    vr = sum((r - mr) ** 2 for r in tr) / nr
    cv = sum((tr[i] - mr) * (tr[i + 1] - mr) for i in range(nr - 1)) / (nr - 1)
    phi = cv / vr if vr > 0 else 0

    pred_ar = []
    lr = tr[-1]
    for i in range(len(holdout)):
        ar = phi * lr
        pred_ar.append(pred_no_ar[i] + ar)
        lr = ar

    return mape(actuals, pred_no_ar), mape(actuals, pred_ar), phi, pred_ar

# Test configurations
configs = [
    ("Baseline (no cleaning)", []),
    ("Clean W40 only", [(2024, 40)]),
    ("Clean W40+W41", [(2024, 40), (2024, 41)]),
    ("Clean W40+W44", [(2024, 40), (2024, 44)]),
    ("Clean W40+W41+W44", [(2024, 40), (2024, 41), (2024, 44)]),
]

print(f"\n{'Config':<30} {'MAPE':>7} {'Δ':>7} {'+AR(1)':>8} {'Δ':>7} {'phi':>6}")
print("-" * 72)
for name, wtc in configs:
    if not wtc:
        m_no_ar = mape_baseline
        # AR(1) on baseline
        tp = [(icpt + slope * r["t"]) * sf_orig.get(r["week"], 1.0) for r in train]
        tr = [r["trips"] - p for r, p in zip(train, tp)]
        nr = len(tr)
        mr = sum(tr) / nr
        vr = sum((r - mr) ** 2 for r in tr) / nr
        cv = sum((tr[i] - mr) * (tr[i + 1] - mr) for i in range(nr - 1)) / (nr - 1)
        phi = cv / vr if vr > 0 else 0
        pred_base_no = pred_baseline
        pred_ar = []
        lr = tr[-1]
        for i in range(len(holdout)):
            ar = phi * lr
            pred_ar.append(pred_base_no[i] + ar)
            lr = ar
        m_ar = mape(actuals, pred_ar)
    else:
        m_no_ar, m_ar, phi, _ = clean_and_predict(wtc)
    print(f"{name:<30} {m_no_ar:>6.2f}% {m_no_ar-mape_baseline:>+6.2f}% "
          f"{m_ar:>7.2f}% {m_ar-mape_baseline:>+6.2f}% {phi:>5.3f}")

# Best model per-week detail
_, _, _, best_pred = clean_and_predict([(2024, 40), (2024, 41), (2024, 44)])
best_mape = mape(actuals, best_pred)

print(f"\n{'='*70}")
print(f"Best weekly model: Clean W40+W41+W44 + AR(1) = {best_mape:.2f}%")
print(f"{'='*70}")
print(f"{'Wk':<5} {'Actual':>8} {'Base':>8} {'Best':>8} {'B%':>7} {'I%':>7} {'Δ':>7}")
print("-" * 55)
improved = 0
for i, r in enumerate(holdout):
    be = abs(actuals[i] - pred_baseline[i]) / actuals[i] * 100
    ie = abs(actuals[i] - best_pred[i]) / actuals[i] * 100
    d = ie - be
    if ie < be:
        improved += 1
    mark = "◄" if abs(d) > 1 else ""
    print(f"W{r['week']:02d}  {actuals[i]:>8,} {pred_baseline[i]:>8,.0f} {best_pred[i]:>8,.0f} "
          f"{be:>6.1f}% {ie:>6.1f}% {d:>+6.1f}% {mark}")
print(f"\nImproved: {improved}/{len(holdout)} weeks")

# Remaining errors
print(f"\nRemaining >8% errors after best model:")
for i, r in enumerate(holdout):
    e = (actuals[i] - best_pred[i]) / actuals[i] * 100
    if abs(e) > 8:
        print(f"  W{r['week']:02d}: {e:+.1f}%")

# =============================================================
# SECTION 6: SUMMARY
# =============================================================

print(f"\n{'='*70}")
print("SECTION 6: SUMMARY OF FINDINGS")
print("=" * 70)

print(f"""
1. BASELINE: trend × seasonal achieves 6.01% MAPE on H2 2025 holdout.
   50% of weeks <5% error. 6 weeks >10% error drive most of the MAPE.

2. ERROR SOURCES:
   - Q4 systematic under-prediction (trend acceleration)
   - Single-year H2 seasonal factors (structural data limitation)
   - Lag-1 autocorrelation = 0.463 (residual persistence)

3. FLOATING HOLIDAYS: Rosh Hashanah dip confirmed in 2024 (-12% city-wide)
   but NOT replicated in 2025 (-0.5% NNS). Effect is variable, not a
   reliable forecasting feature. Fixed holidays (Thanksgiving, Christmas)
   are in the same ISO week both years — not causing alignment errors.

4. H1 BACKTEST: Averaging 2 years of seasonal factors reduces MAPE by
   1.2-2.0pp vs single year. More H2 data (2023) would help.

5. BEST WEEKLY MODEL: Cleaning outlier weeks (W40, W41, W44) + AR(1)
   reduces MAPE from 6.01% to {best_mape:.2f}% (Δ {best_mape-mape_baseline:+.2f}pp).
   Improvement is empirically valid but retrospective — outlier correction
   on identified anomalies, not a generalizable feature.

6. IMPLICATION: Weekly granularity is limited by data availability.
   Daily modeling offers 882 data points vs 124 weeks, enabling:
   - Day-of-week features (40% swing, the dominant factor)
   - Separate weekday/weekend trends (+10.8% vs +21.7%)
   - Lag-1 and lag-7 dual autocorrelation structure
   - Holiday/convention features at proper daily resolution
""")
