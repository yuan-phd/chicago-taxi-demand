"""
Phase 3 v3: Daily NNS demand forecast model.
Clean rewrite with:
  1. M0 = per-DOW trend × month seasonality (complete baseline)
  2. Only major proven holidays (~10)
  3. Convention factors computed on clean M0 (no month contamination)
  4. Residual modeling + honest horizon benchmarking

Inputs:
  data/derived/area_day.csv
  data/external/holidays_chicago.csv
  data/external/conventions_chicago.csv

Usage: python scripts/2_daily_forecast.py
Requires: xgboost, numpy
"""

import csv
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import xgboost as xgb

# =============================================================
# MAJOR HOLIDAYS ONLY (proven >15% impact, consistent across years)
# =============================================================

MAJOR_HOLIDAYS = {
    "Thanksgiving", "Christmas", "Christmas Eve", "Day After Christmas",
    "New Year's Day", "Day After NY", "New Year's Eve",
    "July 4th", "Memorial Day", "Labor Day", "Black Friday",
    "Thanksgiving Weekend", "Pre-Thanksgiving Wed", "Pre-Thanksgiving Tue",
    "St Patrick's Parade Chicago", "St Patrick's Day", "St Patrick's Sunday",
}

# =============================================================
# LOAD DATA
# =============================================================

def load_taxi_daily():
    rows = []
    with open("data/derived/area_day.csv") as f:
        for r in csv.DictReader(f):
            if int(r["pickup_community_area"]) == 8:
                d = date.fromisoformat(r["date"])
                trips = int(r["trip_count"])
                if trips < 100:
                    continue
                rows.append({"date": d, "trips": trips})
    rows.sort(key=lambda x: x["date"])
    return rows


def load_holidays():
    """Load only major holidays."""
    events = {}
    with open("data/external/holidays_chicago.csv") as f:
        for r in csv.DictReader(f):
            if r["event_name"] not in MAJOR_HOLIDAYS:
                continue
            d = date.fromisoformat(r["date"])
            events.setdefault(d, []).append({
                "name": r["event_name"],
                "impact": r["expected_impact"],
            })
    return events


def load_conventions():
    events = []
    with open("data/external/conventions_chicago.csv") as f:
        for r in csv.DictReader(f):
            events.append({
                "start": date.fromisoformat(r["date_start"]),
                "end": date.fromisoformat(r["date_end"]),
                "name": r["event_name"],
                "attendees": int(r["est_attendees"]),
            })
    return events


daily = load_taxi_daily()
holidays = load_holidays()
conventions = load_conventions()
date_trips = {r["date"]: r["trips"] for r in daily}

# =============================================================
# M0: Per-DOW linear trend × month seasonality
# =============================================================

def fit_m0(train_data):
    """
    Fit M0 baseline: per-DOW linear trend × month factor.
    Step 1: fit 7 separate linear trends (one per DOW)
    Step 2: compute month adjustment factors from residuals
    """
    # Step 1: per-DOW trends
    dow_data = defaultdict(lambda: {"t": [], "y": []})
    for r in train_data:
        dow_data[r["date"].weekday()]["t"].append(r["t"])
        dow_data[r["date"].weekday()]["y"].append(r["trips"])

    dow_models = {}
    for dow in range(7):
        t = np.array(dow_data[dow]["t"])
        y = np.array(dow_data[dow]["y"])
        mt, my = t.mean(), y.mean()
        slope = np.sum((t - mt) * (y - my)) / np.sum((t - mt) ** 2)
        intercept = my - slope * mt
        dow_models[dow] = (slope, intercept)

    # Step 2: month factors from DOW-trend residuals
    month_ratios = defaultdict(list)
    for r in train_data:
        slope, intercept = dow_models[r["date"].weekday()]
        trend_pred = slope * r["t"] + intercept
        if trend_pred > 0:
            month_ratios[r["date"].month].append(r["trips"] / trend_pred)

    month_factors = {m: np.mean(rs) for m, rs in month_ratios.items()}

    return dow_models, month_factors


def predict_m0(dow_models, month_factors, feat_list):
    """Predict M0: DOW trend × month factor."""
    preds = []
    for f in feat_list:
        slope, intercept = dow_models[f["date"].weekday()]
        trend = slope * f["t"] + intercept
        mf = month_factors.get(f["date"].month, 1.0)
        preds.append(trend * mf)
    return np.array(preds)

# =============================================================
# HOLIDAY & CONVENTION FACTORS
# =============================================================

def compute_holiday_factors(train_data, dow_models, month_factors):
    """
    Compute per-holiday impact factor = avg(actual / M0_prediction).
    M0 already includes month seasonality, so factors are clean.
    """
    factors = defaultdict(list)
    for r in train_data:
        h_list = holidays.get(r["date"], [])
        if not h_list:
            continue
        slope, intercept = dow_models[r["date"].weekday()]
        trend = slope * r["t"] + intercept
        mf = month_factors.get(r["date"].month, 1.0)
        expected = trend * mf
        if expected <= 0:
            continue
        ratio = r["trips"] / expected
        # If multiple holidays on same date, assign to the first (most impactful)
        # Since we only have major holidays, this rarely happens
        factors[h_list[0]["name"]].append(ratio)

    return {name: {"factor": np.mean(rs), "n": len(rs), "std": np.std(rs) if len(rs) > 1 else 0}
            for name, rs in factors.items()}


def compute_convention_factors(train_data, dow_models, month_factors):
    """Compute per-convention impact factor on clean M0 baseline."""
    factors = defaultdict(list)
    for r in train_data:
        active = [c for c in conventions if c["start"] <= r["date"] <= c["end"]]
        if not active:
            continue
        # Skip days that also have a major holiday (holiday dominates)
        if holidays.get(r["date"]):
            continue
        slope, intercept = dow_models[r["date"].weekday()]
        trend = slope * r["t"] + intercept
        mf = month_factors.get(r["date"].month, 1.0)
        expected = trend * mf
        if expected <= 0:
            continue
        ratio = r["trips"] / expected
        for c in active:
            factors[c["name"]].append(ratio)

    return {name: {"factor": np.mean(rs), "n": len(rs)}
            for name, rs in factors.items()}


def get_holiday_factor(d, hol_factors):
    """Get holiday factor for a date. Returns (factor, name)."""
    h_list = holidays.get(d, [])
    if not h_list:
        return 1.0, None
    name = h_list[0]["name"]
    hf = hol_factors.get(name)
    return (hf["factor"], name) if hf else (1.0, None)


def get_convention_factor(d, conv_factors):
    """Get convention factor for a date (largest convention if multiple)."""
    active = [c for c in conventions if c["start"] <= d <= c["end"]]
    if not active:
        return 1.0, None
    # Skip if major holiday on same date
    if holidays.get(d):
        return 1.0, None
    biggest = max(active, key=lambda c: c["attendees"])
    cf = conv_factors.get(biggest["name"])
    return (cf["factor"], biggest["name"]) if cf else (1.0, None)

# =============================================================
# FEATURE ENGINEERING
# =============================================================

def build_features(data, dow_models, month_factors, hol_factors, conv_factors,
                   wd_offset=0, we_offset=0):
    rows = []
    wd_count = wd_offset
    we_count = we_offset

    for r in data:
        d = r["date"]
        t = r["t"]
        iso = d.isocalendar()

        if d.weekday() < 5:
            wd_count += 1
        else:
            we_count += 1

        # M0 prediction
        slope, intercept = dow_models[d.weekday()]
        trend = slope * t + intercept
        mf = month_factors.get(d.month, 1.0)
        m0_pred = trend * mf

        # Factors
        hol_factor, hol_name = get_holiday_factor(d, hol_factors)
        conv_factor, conv_name = get_convention_factor(d, conv_factors)
        m0_adjusted = m0_pred * hol_factor * conv_factor

        f = {
            "date": d,
            "trips": r["trips"],
            "t": t,
            "m0_pred": m0_pred,
            "m0_adjusted": m0_adjusted,
            "residual": r["trips"] - m0_adjusted,
            # Temporal
            "day_of_week": d.weekday(),
            "is_weekend": 1 if d.weekday() >= 5 else 0,
            "month": d.month,
            "week_of_year": iso[1],
            "day_of_year": d.timetuple().tm_yday,
            "weekday_t": wd_count if d.weekday() < 5 else 0,
            "weekend_t": we_count if d.weekday() >= 5 else 0,
            # Holiday (continuous factor + binary flag)
            "holiday_factor": hol_factor,
            "has_holiday": 1 if hol_factor != 1.0 else 0,
            # Convention (continuous factor + binary flag)
            "convention_factor": conv_factor,
            "has_convention": 1 if conv_factor != 1.0 else 0,
            # Near-holiday proximity
            "near_holiday": 0,
            # Labels for reporting
            "_hol_name": hol_name,
            "_conv_name": conv_name,
        }

        # Near-holiday (major negative holiday within ±3 days)
        for offset in [-3, -2, -1, 1, 2, 3]:
            nd = d + timedelta(days=offset)
            nf, _ = get_holiday_factor(nd, hol_factors)
            if nf < 0.8:
                f["near_holiday"] = 1
                break

        # Lags
        for lag, name in [(1, "lag_1"), (7, "lag_7"), (14, "lag_14")]:
            f[name] = date_trips.get(d - timedelta(days=lag), np.nan)

        # Rolling 7-day
        vals = [date_trips[d - timedelta(days=k)]
                for k in range(1, 8) if (d - timedelta(days=k)) in date_trips]
        f["rolling_mean_7"] = np.mean(vals) if len(vals) >= 5 else np.nan
        f["rolling_std_7"] = np.std(vals) if len(vals) >= 5 else np.nan

        rows.append(f)

    return rows

# =============================================================
# METRICS
# =============================================================

def daily_mape(actual, pred):
    return np.mean(np.abs(actual - pred) / actual) * 100


def weekly_mape(feat_list, preds):
    wa = defaultdict(int)
    wp = defaultdict(float)
    for f, p in zip(feat_list, preds):
        key = (f["date"].isocalendar()[0], f["date"].isocalendar()[1])
        wa[key] += f["trips"]
        wp[key] += p
    return np.mean([abs(wa[k] - wp[k]) / wa[k] for k in wa]) * 100

# =============================================================
# MAIN
# =============================================================

print("=" * 70)
print("PHASE 3: DAILY FORECAST MODEL (v3 — clean)")
print("=" * 70)

# Global t index
for i, r in enumerate(daily):
    r["t"] = i

# Split
HOLDOUT_START = date(2025, 6, 30)
HOLDOUT_END = date(2025, 12, 28)

train_raw = [r for r in daily if r["date"] < HOLDOUT_START]
holdout_raw = [r for r in daily if HOLDOUT_START <= r["date"] <= HOLDOUT_END]

print(f"\nTrain: {len(train_raw)} days ({train_raw[0]['date']} to {train_raw[-1]['date']})")
print(f"Holdout: {len(holdout_raw)} days ({holdout_raw[0]['date']} to {holdout_raw[-1]['date']})")
print(f"Major holidays tracked: {len(MAJOR_HOLIDAYS)}")

# Fit M0
dow_models, month_factors = fit_m0(train_raw)

print(f"\nMonth seasonality factors (from M0):")
for m in range(1, 13):
    mf = month_factors.get(m, 1.0)
    name = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1]
    print(f"  {name}: {mf:.3f} ({(mf-1)*100:+.1f}%)")

# Compute factors
hol_factors = compute_holiday_factors(train_raw, dow_models, month_factors)
conv_factors = compute_convention_factors(train_raw, dow_models, month_factors)

print(f"\n{'='*70}")
print("HOLIDAY IMPACT FACTORS (on M0 with month seasonality)")
print("=" * 70)
print(f"{'Holiday':<30} {'Factor':>8} {'Impact':>8} {'N':>4}")
print("-" * 55)
for name, info in sorted(hol_factors.items(), key=lambda x: x[1]["factor"]):
    print(f"{name:<30} {info['factor']:>7.3f} {(info['factor']-1)*100:>+7.1f}% {info['n']:>4}")

print(f"\n{'Convention':<45} {'Factor':>8} {'Impact':>8} {'N':>4}")
print("-" * 65)
for name, info in sorted(conv_factors.items(), key=lambda x: -x[1]["factor"]):
    print(f"{name:<45} {info['factor']:>7.3f} {(info['factor']-1)*100:>+7.1f}% {info['n']:>4}")

# Build features
train_feat = build_features(train_raw, dow_models, month_factors, hol_factors, conv_factors)
wd_off = sum(1 for f in train_feat if f["day_of_week"] < 5)
we_off = sum(1 for f in train_feat if f["day_of_week"] >= 5)
holdout_feat = build_features(holdout_raw, dow_models, month_factors, hol_factors, conv_factors,
                              wd_offset=wd_off, we_offset=we_off)

holdout_y = np.array([f["trips"] for f in holdout_feat])

# =============================================================
# MODEL 0: Per-DOW trend × month (baseline)
# =============================================================

print(f"\n{'='*70}")
print("MODEL 0: Per-DOW trend × month seasonality")
print("=" * 70)

pred_m0 = np.array([f["m0_pred"] for f in holdout_feat])
print(f"Daily MAPE: {daily_mape(holdout_y, pred_m0):.2f}%")
print(f"Weekly MAPE: {weekly_mape(holdout_feat, pred_m0):.2f}%")

# =============================================================
# MODEL 1: M0 × holiday × convention factors
# =============================================================

print(f"\n{'='*70}")
print("MODEL 1: M0 × holiday factors × convention factors")
print("=" * 70)

pred_m1 = np.array([f["m0_adjusted"] for f in holdout_feat])
dm1 = daily_mape(holdout_y, pred_m1)
wm1 = weekly_mape(holdout_feat, pred_m1)
print(f"Daily MAPE: {dm1:.2f}%")
print(f"Weekly MAPE: {wm1:.2f}%")
print(f"Holiday/convention improvement: {daily_mape(holdout_y, pred_m0) - dm1:.2f}pp daily")

# =============================================================
# MODEL 2: M1 + XGBoost residual (no lags)
# =============================================================

print(f"\n{'='*70}")
print("MODEL 2: M1 + XGBoost residual (NO lags)")
print("=" * 70)

FEAT_NO_LAG = [
    "day_of_week", "is_weekend", "month", "week_of_year", "day_of_year",
    "weekday_t", "weekend_t",
    "has_holiday", "holiday_factor", "has_convention", "convention_factor",
    "near_holiday",
]

X_tr = np.array([[f[c] for c in FEAT_NO_LAG] for f in train_feat])
y_tr = np.array([f["residual"] for f in train_feat])
X_ho = np.array([[f[c] for c in FEAT_NO_LAG] for f in holdout_feat])

m2 = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
m2.fit(X_tr, y_tr)
pred_m2 = pred_m1 + m2.predict(X_ho)
print(f"Daily MAPE: {daily_mape(holdout_y, pred_m2):.2f}%")
print(f"Weekly MAPE: {weekly_mape(holdout_feat, pred_m2):.2f}%")

imp = sorted(zip(FEAT_NO_LAG, m2.feature_importances_), key=lambda x: -x[1])
print(f"\nResidual model features:")
for name, score in imp[:8]:
    print(f"  {name:<25} {score:.3f}")

# =============================================================
# MODEL 3a: + lag_7 only (next-week horizon)
# =============================================================

print(f"\n{'='*70}")
print("MODEL 3a: + lag_7/lag_14 (NEXT-WEEK forecast horizon)")
print("=" * 70)

FEAT_WEEK = FEAT_NO_LAG + ["lag_7", "lag_14"]
tr_w = [f for f in train_feat if not np.isnan(f["lag_7"])]
ho_w = [f for f in holdout_feat if not np.isnan(f["lag_7"])]

X_tr_w = np.array([[f[c] for c in FEAT_WEEK] for f in tr_w])
y_tr_w = np.array([f["residual"] for f in tr_w])
X_ho_w = np.array([[f[c] for c in FEAT_WEEK] for f in ho_w])

m3a = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
m3a.fit(X_tr_w, y_tr_w)
pred_m3a = np.array([f["m0_adjusted"] for f in ho_w]) + m3a.predict(X_ho_w)
y_ho_w = np.array([f["trips"] for f in ho_w])
print(f"Daily MAPE: {daily_mape(y_ho_w, pred_m3a):.2f}%")
print(f"Weekly MAPE: {weekly_mape(ho_w, pred_m3a):.2f}%")

# =============================================================
# MODEL 3b: + all lags (next-day horizon)
# =============================================================

print(f"\n{'='*70}")
print("MODEL 3b: + all lags (NEXT-DAY forecast horizon)")
print("=" * 70)

FEAT_DAY = FEAT_NO_LAG + ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7"]
tr_d = [f for f in train_feat if not np.isnan(f["lag_1"]) and not np.isnan(f["rolling_mean_7"])]
ho_d = [f for f in holdout_feat if not np.isnan(f["lag_1"]) and not np.isnan(f["rolling_mean_7"])]

X_tr_d = np.array([[f[c] for c in FEAT_DAY] for f in tr_d])
y_tr_d = np.array([f["residual"] for f in tr_d])
X_ho_d = np.array([[f[c] for c in FEAT_DAY] for f in ho_d])

m3b = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
m3b.fit(X_tr_d, y_tr_d)
pred_m3b = np.array([f["m0_adjusted"] for f in ho_d]) + m3b.predict(X_ho_d)
y_ho_d = np.array([f["trips"] for f in ho_d])
print(f"Daily MAPE: {daily_mape(y_ho_d, pred_m3b):.2f}%")
print(f"Weekly MAPE: {weekly_mape(ho_d, pred_m3b):.2f}%")

imp = sorted(zip(FEAT_DAY, m3b.feature_importances_), key=lambda x: -x[1])
print(f"\nTop features:")
for name, score in imp[:10]:
    print(f"  {name:<25} {score:.3f}")

# =============================================================
# BENCHMARK TABLE
# =============================================================

print(f"\n{'='*70}")
print("BENCHMARK TABLE")
print("=" * 70)

dm0 = daily_mape(holdout_y, pred_m0)
wm0 = weekly_mape(holdout_feat, pred_m0)
dm2 = daily_mape(holdout_y, pred_m2)
wm2 = weekly_mape(holdout_feat, pred_m2)
dm3a = daily_mape(y_ho_w, pred_m3a)
wm3a = weekly_mape(ho_w, pred_m3a)
dm3b = daily_mape(y_ho_d, pred_m3b)
wm3b = weekly_mape(ho_d, pred_m3b)

print(f"\n{'Model':<58} {'Daily':>7} {'Weekly':>7}")
print("-" * 73)
rows = [
    ("REFERENCE: Weekly baseline (trend × seasonal)", "—", "6.01%"),
    ("", "", ""),
    (f"M0: Per-DOW trend × month seasonality", f"{dm0:.2f}%", f"{wm0:.2f}%"),
    (f"M1: M0 × holiday/convention factors", f"{dm1:.2f}%", f"{wm1:.2f}%"),
    (f"M2: M1 + XGBoost residual (no lags)", f"{dm2:.2f}%", f"{wm2:.2f}%"),
    (f"M3a: M2 + lag_7 (next-week horizon)", f"{dm3a:.2f}%", f"{wm3a:.2f}%"),
    (f"M3b: M2 + all lags (next-day horizon)", f"{dm3b:.2f}%", f"{wm3b:.2f}%"),
]
for name, d, w in rows:
    print(f"{name:<58} {d:>7} {w:>7}") if name else print()

# =============================================================
# PER-WEEK DETAIL (M3a vs baseline)
# =============================================================

print(f"\n{'='*70}")
print("PER-WEEK: M3a (next-week) vs M0 (baseline)")
print("=" * 70)

wa = defaultdict(int)
wp3a = defaultdict(float)
wp0 = defaultdict(float)
for f, p in zip(ho_w, pred_m3a):
    k = (f["date"].isocalendar()[0], f["date"].isocalendar()[1])
    wa[k] += f["trips"]
    wp3a[k] += p
for f in holdout_feat:
    k = (f["date"].isocalendar()[0], f["date"].isocalendar()[1])
    wp0[k] += f["m0_pred"]

print(f"{'Week':<9} {'Actual':>8} {'M0':>8} {'M3a':>8} {'M0%':>7} {'M3a%':>7}")
print("-" * 52)
for k in sorted(wa.keys()):
    a = wa[k]; p0 = wp0.get(k, 0); p3 = wp3a[k]
    e0 = abs(a - p0) / a * 100; e3 = abs(a - p3) / a * 100
    b = "◄" if e3 < e0 - 1 else ""
    print(f"{k[0]}W{k[1]:02d}   {a:>8,} {p0:>8,.0f} {p3:>8,.0f} {e0:>6.1f}% {e3:>6.1f}% {b}")

# =============================================================
# HOLIDAY ACCURACY
# =============================================================

print(f"\n{'='*70}")
print("HOLIDAY PREDICTION: M1 (factor-based) vs M0 (no factors)")
print("=" * 70)
print(f"{'Date':<12} {'Event':<25} {'Actual':>7} {'M0':>7} {'M1':>7} {'M0%':>7} {'M1%':>7}")
print("-" * 80)
for f in holdout_feat:
    name = f["_hol_name"] or f["_conv_name"]
    if name:
        a = f["trips"]; p0 = f["m0_pred"]; p1 = f["m0_adjusted"]
        e0 = (a - p0) / a * 100; e1 = (a - p1) / a * 100
        b = "◄" if abs(e1) < abs(e0) else ""
        print(f"{f['date']} {name:<25} {a:>7,} {p0:>7,.0f} {p1:>7,.0f} {e0:>+6.1f}% {e1:>+6.1f}% {b}")
