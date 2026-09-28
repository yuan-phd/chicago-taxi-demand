"""
Phase 6 — Total Market View (Taxi + Ride-hail)
================================================
Core question: Is NNS +30% taxi growth real demand growth
or mode-share shift from ride-hail?

Analysis:
1. NNS taxi vs ride-hail trend comparison
2. Mode-share (taxi / total) over time
3. YoY growth: taxi vs ride-hail
4. Top 5 areas comparison
5. Forecasting test: is total demand more predictable?

Run from project root:
    python scripts/4_total_market.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
Path("outputs").mkdir(exist_ok=True)

# ── 1. Load and merge ───────────────────────────────────────────────
print("Loading data...")
taxi = pd.read_csv("data/from_assignment/area_week.csv")
tnp = pd.read_csv("data/external/tnp_area_week.csv")

# Standardize columns
taxi = taxi.rename(columns={"trip_count": "taxi_trips"})
tnp = tnp.rename(columns={"trip_count": "tnp_trips"})

# Merge on overlapping weeks
merged = taxi[["pickup_community_area", "iso_year", "iso_week", "taxi_trips"]].merge(
    tnp[["pickup_community_area", "iso_year", "iso_week", "tnp_trips"]],
    on=["pickup_community_area", "iso_year", "iso_week"],
    how="inner",
)
merged["total_trips"] = merged["taxi_trips"] + merged["tnp_trips"]
merged["taxi_share"] = merged["taxi_trips"] / merged["total_trips"]

# Week ordering
merged["week_start"] = pd.to_datetime(
    merged["iso_year"].astype(str)
    + merged["iso_week"].astype(str).str.zfill(2) + "1",
    format="%G%V%u",
)

print(f"  Merged rows: {len(merged):,}")
print(f"  Areas: {merged['pickup_community_area'].nunique()}")
print(f"  Weeks: {merged['week_start'].min().date()} to {merged['week_start'].max().date()}")

# Identify top 5 areas by taxi volume
top5 = (
    merged.groupby("pickup_community_area")["taxi_trips"]
    .sum().nlargest(5).index.tolist()
)
area_names = {8: "Near North Side", 32: "Loop", 76: "O'Hare",
              28: "Near West Side", 33: "Near South Side",
              6: "Lake View", 24: "West Town", 7: "Lincoln Park"}
print(f"  Top 5 areas by taxi volume: {top5}")

# ══════════════════════════════════════════════════════════════════════
# 2. NNS (Area 8) Deep Dive
# ══════════════════════════════════════════════════════════════════════
nns = merged[merged["pickup_community_area"] == 8].sort_values("week_start").copy()

print("\n" + "=" * 60)
print("NNS (AREA 8): TAXI vs RIDE-HAIL")
print("=" * 60)

print(f"\n  {'Metric':<30s} {'Taxi':>12} {'Ride-hail':>12} {'Total':>12}")
print(f"  {'-' * 68}")
print(f"  {'Avg weekly trips':<30s} {nns['taxi_trips'].mean():>12,.0f} "
      f"{nns['tnp_trips'].mean():>12,.0f} {nns['total_trips'].mean():>12,.0f}")
print(f"  {'Ratio (ride-hail / taxi)':<30s} {'':>12} "
      f"{nns['tnp_trips'].mean() / nns['taxi_trips'].mean():>12.1f}x {'':>12}")
print(f"  {'Taxi mode share (avg)':<30s} {nns['taxi_share'].mean():>12.1%} {'':>12} {'':>12}")

# ── 2a. YoY growth comparison ───────────────────────────────────────
print("\n  YoY Growth (like-for-like weeks):")

for y1, y2 in [(2024, 2025), (2025, 2026)]:
    yr1 = nns[nns["iso_year"] == y1]
    yr2 = nns[nns["iso_year"] == y2]
    overlap_weeks = set(yr1["iso_week"]) & set(yr2["iso_week"])
    if not overlap_weeks:
        continue
    yr1_ol = yr1[yr1["iso_week"].isin(overlap_weeks)]
    yr2_ol = yr2[yr2["iso_week"].isin(overlap_weeks)]

    taxi_g = (yr2_ol["taxi_trips"].sum() / yr1_ol["taxi_trips"].sum() - 1) * 100
    tnp_g = (yr2_ol["tnp_trips"].sum() / yr1_ol["tnp_trips"].sum() - 1) * 100
    total_g = (yr2_ol["total_trips"].sum() / yr1_ol["total_trips"].sum() - 1) * 100

    # Mode share change
    share_y1 = yr1_ol["taxi_trips"].sum() / yr1_ol["total_trips"].sum()
    share_y2 = yr2_ol["taxi_trips"].sum() / yr2_ol["total_trips"].sum()

    print(f"\n  {y1}→{y2} ({len(overlap_weeks)} overlapping weeks):")
    print(f"    Taxi growth:      {taxi_g:+.1f}%")
    print(f"    Ride-hail growth: {tnp_g:+.1f}%")
    print(f"    Total growth:     {total_g:+.1f}%")
    print(f"    Taxi share:       {share_y1:.1%} → {share_y2:.1%} "
          f"({(share_y2 - share_y1) * 100:+.1f}pp)")

# ── 2b. Mode-share trend ────────────────────────────────────────────
print("\n  Mode-share trend (taxi / total) by half-year:")
for y in sorted(nns["iso_year"].unique()):
    for half, (wlo, whi) in [("H1", (1, 26)), ("H2", (27, 52))]:
        subset = nns[(nns["iso_year"] == y) & nns["iso_week"].between(wlo, whi)]
        if len(subset) == 0:
            continue
        share = subset["taxi_trips"].sum() / subset["total_trips"].sum()
        print(f"    {y} {half}: {share:.1%}  (taxi={subset['taxi_trips'].mean():,.0f}/wk, "
              f"tnp={subset['tnp_trips'].mean():,.0f}/wk)")

# ── 2c. Correlation ─────────────────────────────────────────────────
print(f"\n  Weekly correlation (taxi vs ride-hail): {nns['taxi_trips'].corr(nns['tnp_trips']):.3f}")
print(f"  Weekly correlation (taxi vs total):     {nns['taxi_trips'].corr(nns['total_trips']):.3f}")

# ══════════════════════════════════════════════════════════════════════
# 3. TOP 5 AREAS COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TOP 5 AREAS: GROWTH COMPARISON")
print("=" * 60)

print(f"\n  {'Area':<20s} {'Taxi 24→25':>12} {'TNP 24→25':>12} "
      f"{'Total 24→25':>12} {'Share Δ':>8}")
print(f"  {'-' * 66}")

for area in top5:
    area_data = merged[merged["pickup_community_area"] == area]
    yr1 = area_data[area_data["iso_year"] == 2024]
    yr2 = area_data[area_data["iso_year"] == 2025]
    ow = set(yr1["iso_week"]) & set(yr2["iso_week"])
    if not ow:
        continue
    yr1_o = yr1[yr1["iso_week"].isin(ow)]
    yr2_o = yr2[yr2["iso_week"].isin(ow)]

    tg = (yr2_o["taxi_trips"].sum() / yr1_o["taxi_trips"].sum() - 1) * 100
    ng = (yr2_o["tnp_trips"].sum() / yr1_o["tnp_trips"].sum() - 1) * 100
    totg = (yr2_o["total_trips"].sum() / yr1_o["total_trips"].sum() - 1) * 100
    s1 = yr1_o["taxi_trips"].sum() / yr1_o["total_trips"].sum()
    s2 = yr2_o["taxi_trips"].sum() / yr2_o["total_trips"].sum()
    sd = (s2 - s1) * 100

    name = area_names.get(area, f"Area {area}")
    print(f"  {name:<20s} {tg:>+11.1f}% {ng:>+11.1f}% "
          f"{totg:>+11.1f}% {sd:>+7.1f}pp")

# ══════════════════════════════════════════════════════════════════════
# 4. FORECASTING TEST: TAXI-ONLY vs TOTAL DEMAND
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FORECASTING TEST: TAXI vs TOTAL DEMAND (NNS)")
print("=" * 60)


def trend_seasonal_forecast(train_trips, train_weeks, test_weeks):
    """
    Multiplicative trend × seasonal (same method as assignment).
    train_trips: array of weekly trip counts
    train_weeks: array of iso_week (1-52) for training period
    test_weeks: array of iso_week (1-52) for test period
    Returns: array of predicted trip counts for test period
    """
    n = len(train_trips)
    x = np.arange(n)

    # Linear trend
    slope, intercept = np.polyfit(x, train_trips, 1)
    trend_train = slope * x + intercept

    # Detrend
    detrended = train_trips / trend_train

    # Seasonal factors: average detrended value per week-of-year
    seasonal = {}
    for woy, dt in zip(train_weeks, detrended):
        seasonal.setdefault(woy, []).append(dt)
    seasonal_factors = {w: np.mean(v) for w, v in seasonal.items()}

    # Forecast
    preds = []
    for i, woy in enumerate(test_weeks):
        t_idx = n + i
        trend_val = slope * t_idx + intercept
        sf = seasonal_factors.get(woy, 1.0)
        preds.append(trend_val * sf)

    return np.array(preds)


# Train/test split
train_nns = nns[(nns["iso_year"] < 2025) |
                ((nns["iso_year"] == 2025) & (nns["iso_week"] <= 26))].copy()
test_nns = nns[(nns["iso_year"] == 2025) &
               nns["iso_week"].between(27, 52)].copy()

if len(test_nns) > 0 and len(train_nns) > 0:
    # Taxi-only forecast
    taxi_pred = trend_seasonal_forecast(
        train_nns["taxi_trips"].values,
        train_nns["iso_week"].values,
        test_nns["iso_week"].values,
    )

    # Total demand forecast
    total_pred = trend_seasonal_forecast(
        train_nns["total_trips"].values,
        train_nns["iso_week"].values,
        test_nns["iso_week"].values,
    )

    # Derived taxi forecast: total_pred × train-period taxi share
    train_taxi_share = train_nns["taxi_trips"].sum() / train_nns["total_trips"].sum()
    taxi_from_total_pred = total_pred * train_taxi_share

    # Evaluate
    taxi_actual = test_nns["taxi_trips"].values
    total_actual = test_nns["total_trips"].values

    taxi_mape = np.mean(np.abs(taxi_actual - taxi_pred) / taxi_actual) * 100
    total_mape = np.mean(np.abs(total_actual - total_pred) / total_actual) * 100
    derived_mape = np.mean(np.abs(taxi_actual - taxi_from_total_pred) / taxi_actual) * 100

    taxi_mae = np.mean(np.abs(taxi_actual - taxi_pred))
    total_mae = np.mean(np.abs(total_actual - total_pred))

    print(f"\n  Train: {len(train_nns)} weeks | Test: {len(test_nns)} weeks")
    print(f"  Train taxi share: {train_taxi_share:.1%}")
    print()
    print(f"  {'Model':<40s} {'MAPE':>8} {'MAE':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'Taxi-only (trend×seasonal)':<40s} {taxi_mape:>7.1f}% {taxi_mae:>10,.0f}")
    print(f"  {'Total demand (trend×seasonal)':<40s} {total_mape:>7.1f}% {total_mae:>10,.0f}")
    print(f"  {'Derived taxi (total×share)':<40s} {derived_mape:>7.1f}% {'—':>10}")

    # Coefficient of variation comparison
    taxi_cv = np.std(taxi_actual) / np.mean(taxi_actual)
    total_cv = np.std(total_actual) / np.mean(total_actual)
    print(f"\n  Test period variability (CV):")
    print(f"    Taxi:  {taxi_cv:.3f}")
    print(f"    Total: {total_cv:.3f}")
    print(f"    Total is {'less' if total_cv < taxi_cv else 'more'} variable")

# ══════════════════════════════════════════════════════════════════════
# 5. VERDICT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

# Compute key numbers for verdict
yr1_nns = nns[nns["iso_year"] == 2024]
yr2_nns = nns[nns["iso_year"] == 2025]
ow = set(yr1_nns["iso_week"]) & set(yr2_nns["iso_week"])
yr1_o = yr1_nns[yr1_nns["iso_week"].isin(ow)]
yr2_o = yr2_nns[yr2_nns["iso_week"].isin(ow)]

taxi_yoy = (yr2_o["taxi_trips"].sum() / yr1_o["taxi_trips"].sum() - 1) * 100
tnp_yoy = (yr2_o["tnp_trips"].sum() / yr1_o["tnp_trips"].sum() - 1) * 100
total_yoy = (yr2_o["total_trips"].sum() / yr1_o["total_trips"].sum() - 1) * 100

share_24 = yr1_o["taxi_trips"].sum() / yr1_o["total_trips"].sum()
share_25 = yr2_o["taxi_trips"].sum() / yr2_o["total_trips"].sum()

print(f"""
  NNS 2024→2025:
    Taxi growth:      {taxi_yoy:+.1f}%
    Ride-hail growth: {tnp_yoy:+.1f}%
    Total growth:     {total_yoy:+.1f}%
    Taxi share:       {share_24:.1%} → {share_25:.1%}

  If taxi up AND ride-hail up → REAL DEMAND GROWTH
  If taxi up AND ride-hail down → MODE-SHARE SHIFT
  If taxi up AND ride-hail flat → PARTIAL MODE-SHARE SHIFT
""")

# ── Save key data for slides ────────────────────────────────────────
nns_export = nns[["iso_year", "iso_week", "week_start",
                   "taxi_trips", "tnp_trips", "total_trips", "taxi_share"]].copy()
nns_export.to_csv("outputs/phase6_nns_total_market.csv", index=False)
print("Saved: outputs/phase6_nns_total_market.csv")
print("\nDone.")
