"""
Quick test: v3 features + v1 approach.
XGBoost predicts RAW demand (not residuals), using:
  - Continuous holiday/convention factors (from v3)
  - Month seasonality as feature (from v3)
  - Honest lag_7 only (next-week horizon)

Usage: python scripts/2b_v3v1_test.py
Requires: xgboost, numpy
"""

import csv
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import xgboost as xgb

# Major holidays only
MAJOR_HOLIDAYS = {
    "Thanksgiving", "Christmas", "Christmas Eve", "Day After Christmas",
    "New Year's Day", "Day After NY", "New Year's Eve",
    "July 4th", "Memorial Day", "Labor Day", "Black Friday",
    "Thanksgiving Weekend", "Pre-Thanksgiving Wed", "Pre-Thanksgiving Tue",
    "St Patrick's Parade Chicago", "St Patrick's Day", "St Patrick's Sunday",
}

# Load data
daily = []
with open("data/derived/area_day.csv") as f:
    for r in csv.DictReader(f):
        if int(r["pickup_community_area"]) == 8:
            d = date.fromisoformat(r["date"])
            t = int(r["trip_count"])
            if t >= 100:
                daily.append({"date": d, "trips": t})
daily.sort(key=lambda x: x["date"])
for i, r in enumerate(daily):
    r["t"] = i
date_trips = {r["date"]: r["trips"] for r in daily}

holidays = {}
with open("data/external/holidays_chicago.csv") as f:
    for r in csv.DictReader(f):
        if r["event_name"] in MAJOR_HOLIDAYS:
            d = date.fromisoformat(r["date"])
            holidays.setdefault(d, []).append(r["event_name"])

conventions = []
with open("data/external/conventions_chicago.csv") as f:
    for r in csv.DictReader(f):
        conventions.append({
            "start": date.fromisoformat(r["date_start"]),
            "end": date.fromisoformat(r["date_end"]),
            "name": r["event_name"],
            "attendees": int(r["est_attendees"]),
        })

# Split
HOLDOUT_START = date(2025, 6, 30)
HOLDOUT_END = date(2025, 12, 28)
train = [r for r in daily if r["date"] < HOLDOUT_START]
holdout = [r for r in daily if HOLDOUT_START <= r["date"] <= HOLDOUT_END]

# --- Fit M0 (DOW trend + month) to compute holiday/convention factors ---

dow_data = defaultdict(lambda: {"t": [], "y": []})
for r in train:
    dow_data[r["date"].weekday()]["t"].append(r["t"])
    dow_data[r["date"].weekday()]["y"].append(r["trips"])
dow_models = {}
for dow in range(7):
    t = np.array(dow_data[dow]["t"]); y = np.array(dow_data[dow]["y"])
    mt, my = t.mean(), y.mean()
    s = np.sum((t-mt)*(y-my)) / np.sum((t-mt)**2)
    dow_models[dow] = (s, my - s*mt)

month_ratios = defaultdict(list)
for r in train:
    s, ic = dow_models[r["date"].weekday()]
    pred = s * r["t"] + ic
    if pred > 0:
        month_ratios[r["date"].month].append(r["trips"] / pred)
month_factors = {m: np.mean(rs) for m, rs in month_ratios.items()}

def m0_pred(r):
    s, ic = dow_models[r["date"].weekday()]
    return (s * r["t"] + ic) * month_factors.get(r["date"].month, 1.0)

# Holiday factors
hol_factors = {}
hol_samples = defaultdict(list)
for r in train:
    h = holidays.get(r["date"], [])
    if h:
        expected = m0_pred(r)
        if expected > 0:
            hol_samples[h[0]].append(r["trips"] / expected)
hol_factors = {n: np.mean(rs) for n, rs in hol_samples.items()}

# Convention factors
conv_samples = defaultdict(list)
for r in train:
    active = [c for c in conventions if c["start"] <= r["date"] <= c["end"]]
    if active and not holidays.get(r["date"]):
        expected = m0_pred(r)
        if expected > 0:
            for c in active:
                conv_samples[c["name"]].append(r["trips"] / expected)
conv_factors = {n: np.mean(rs) for n, rs in conv_samples.items()}

# --- Build features for XGBoost on RAW demand ---

def get_hol_factor(d):
    h = holidays.get(d, [])
    if h:
        return hol_factors.get(h[0], 1.0)
    return 1.0

def get_conv_factor(d):
    active = [c for c in conventions if c["start"] <= d <= c["end"]]
    if active and not holidays.get(d):
        biggest = max(active, key=lambda c: c["attendees"])
        return conv_factors.get(biggest["name"], 1.0)
    return 1.0

def near_holiday(d):
    for off in [-3, -2, -1, 1, 2, 3]:
        if get_hol_factor(d + timedelta(days=off)) < 0.8:
            return 1
    return 0

FEATURES = [
    "t", "day_of_week", "is_weekend", "month", "week_of_year", "day_of_year",
    "holiday_factor", "has_holiday", "convention_factor", "has_convention",
    "near_holiday", "lag_7", "lag_14",
]

def build_row(r):
    d = r["date"]
    iso = d.isocalendar()
    hf = get_hol_factor(d)
    cf = get_conv_factor(d)
    l7 = date_trips.get(d - timedelta(days=7), np.nan)
    l14 = date_trips.get(d - timedelta(days=14), np.nan)
    return {
        "date": d, "trips": r["trips"], "t": r["t"],
        "day_of_week": d.weekday(), "is_weekend": 1 if d.weekday() >= 5 else 0,
        "month": d.month, "week_of_year": iso[1],
        "day_of_year": d.timetuple().tm_yday,
        "holiday_factor": hf, "has_holiday": 1 if hf != 1.0 else 0,
        "convention_factor": cf, "has_convention": 1 if cf != 1.0 else 0,
        "near_holiday": near_holiday(d),
        "lag_7": l7, "lag_14": l14,
    }

train_rows = [build_row(r) for r in train]
hold_rows = [build_row(r) for r in holdout]

# Filter valid lag rows
train_valid = [r for r in train_rows if not np.isnan(r["lag_7"])]
hold_valid = [r for r in hold_rows if not np.isnan(r["lag_7"])]

X_train = np.array([[r[f] for f in FEATURES] for r in train_valid])
y_train = np.array([r["trips"] for r in train_valid])
X_hold = np.array([[r[f] for f in FEATURES] for r in hold_valid])
y_hold = np.array([r["trips"] for r in hold_valid])

# --- Train and predict ---

model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
)
model.fit(X_train, y_train)
pred = model.predict(X_hold)

# --- Metrics ---

def daily_mape(a, p):
    return np.mean(np.abs(a - p) / a) * 100

def weekly_mape(rows, preds):
    wa = defaultdict(int); wp = defaultdict(float)
    for r, p in zip(rows, preds):
        k = (r["date"].isocalendar()[0], r["date"].isocalendar()[1])
        wa[k] += r["trips"]; wp[k] += p
    return np.mean([abs(wa[k]-wp[k])/wa[k] for k in wa]) * 100

dm = daily_mape(y_hold, pred)
wm = weekly_mape(hold_valid, pred)

print("=" * 70)
print("v3+v1 TEST: XGBoost on RAW demand")
print("Features: continuous holiday/convention factors + month + lag_7 + lag_14")
print("Horizon: next-week (honest — no lag_1)")
print("=" * 70)
print(f"\nDaily MAPE:  {dm:.2f}%")
print(f"Weekly MAPE: {wm:.2f}%")
print(f"vs Weekly baseline: {wm - 6.01:+.2f}pp")

# Feature importance
imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
print(f"\nTop features:")
for name, score in imp:
    print(f"  {name:<25} {score:.3f}")

# Per-week
print(f"\n{'Week':<9} {'Actual':>8} {'Pred':>8} {'Err%':>7}")
print("-" * 35)
wa = defaultdict(int); wp = defaultdict(float)
for r, p in zip(hold_valid, pred):
    k = (r["date"].isocalendar()[0], r["date"].isocalendar()[1])
    wa[k] += r["trips"]; wp[k] += p
for k in sorted(wa.keys()):
    a = wa[k]; p = wp[k]
    e = abs(a-p)/a*100
    print(f"{k[0]}W{k[1]:02d}   {a:>8,} {p:>8,.0f} {e:>6.1f}%")
