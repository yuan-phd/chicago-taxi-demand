"""
Phase 5 — O-D Corridor EDA
===========================
Validates data before modeling:
1. Density tier distribution
2. Time coverage per corridor
3. Weekly aggregation + zero-trip weeks
4. Train/test split sizes by tier
5. Stationarity check (can trailing mean work?)

Inputs:
    data/derived/corridor_day.csv
    data/derived/corridor_characteristics.csv

Run from project root:
    python scripts/3_corridor_eda.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────
corr_day = pd.read_csv("data/derived/corridor_day.csv")
corr_char = pd.read_csv("data/derived/corridor_characteristics.csv")

corr_day["date"] = pd.to_datetime(corr_day["date"])
corr_day["corridor_id"] = (
    corr_day["pickup_community_area"].astype(str)
    + "→"
    + corr_day["dropoff_community_area"].astype(str)
)
corr_char["corridor_id"] = (
    corr_char["pickup_community_area"].astype(str)
    + "→"
    + corr_char["dropoff_community_area"].astype(str)
)

print(f"corridor_day: {len(corr_day):,} rows, {corr_day['corridor_id'].nunique()} corridors")
print(f"corridor_char: {len(corr_char)} corridors")
print(f"Date range: {corr_day['date'].min()} to {corr_day['date'].max()}")
print()

# ── 1. Density tiers ──────────────────────────────────────────────────
def assign_tier(avg_daily):
    if avg_daily >= 100:
        return "dense (≥100/day)"
    elif avg_daily >= 20:
        return "medium (20-99/day)"
    else:
        return "sparse (<20/day)"

corr_char["tier"] = corr_char["avg_daily_trips"].apply(assign_tier)

print("=" * 60)
print("1. DENSITY TIERS")
print("=" * 60)
tier_summary = (
    corr_char.groupby("tier")
    .agg(
        n_corridors=("corridor_id", "count"),
        total_trips=("total_trips", "sum"),
        avg_daily_min=("avg_daily_trips", "min"),
        avg_daily_median=("avg_daily_trips", "median"),
        avg_daily_max=("avg_daily_trips", "max"),
    )
    .sort_values("avg_daily_median", ascending=False)
)
tier_summary["trip_share"] = (
    tier_summary["total_trips"] / tier_summary["total_trips"].sum() * 100
).round(1)
print(tier_summary.to_string())
print()

# ── 2. Time coverage per corridor ─────────────────────────────────────
print("=" * 60)
print("2. TIME COVERAGE")
print("=" * 60)

coverage = (
    corr_day.groupby("corridor_id")
    .agg(
        n_days=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        total_trips=("trip_count", "sum"),
    )
)
coverage = coverage.merge(
    corr_char[["corridor_id", "tier"]], on="corridor_id", how="left"
)

# Total possible days
total_days = (corr_day["date"].max() - corr_day["date"].min()).days + 1
coverage["coverage_pct"] = (coverage["n_days"] / total_days * 100).round(1)

print(f"Total possible days: {total_days}")
print()
print("Coverage by tier:")
cov_by_tier = (
    coverage.groupby("tier")
    .agg(
        n_corridors=("corridor_id", "count"),
        days_min=("n_days", "min"),
        days_median=("n_days", "median"),
        days_max=("n_days", "max"),
        coverage_min=("coverage_pct", "min"),
        coverage_median=("coverage_pct", "median"),
    )
    .sort_values("days_median", ascending=False)
)
print(cov_by_tier.to_string())
print()

# How many corridors have gaps (not present every day)?
full_coverage = coverage[coverage["n_days"] == total_days]
print(f"Corridors with data every day: {len(full_coverage)} / {len(coverage)}")
print(f"Corridors missing >10% of days: {len(coverage[coverage['coverage_pct'] < 90])}")
print()

# ── 3. Weekly aggregation ─────────────────────────────────────────────
print("=" * 60)
print("3. WEEKLY AGGREGATION")
print("=" * 60)

# Aggregate to weekly
corr_week = (
    corr_day.groupby(["corridor_id", "pickup_community_area",
                       "dropoff_community_area", "iso_year", "iso_week"])
    .agg(weekly_trips=("trip_count", "sum"), n_days=("date", "count"))
    .reset_index()
)

# Create week_start date for ordering
corr_week["week_start"] = pd.to_datetime(
    corr_week["iso_year"].astype(str) + corr_week["iso_week"].astype(str) + "1",
    format="%G%V%u"
)

print(f"Weekly rows: {len(corr_week):,}")
print(f"Week range: {corr_week['week_start'].min().date()} to {corr_week['week_start'].max().date()}")

# Total possible weeks
all_weeks = corr_week[["iso_year", "iso_week", "week_start"]].drop_duplicates().sort_values("week_start")
n_total_weeks = len(all_weeks)
print(f"Total weeks: {n_total_weeks}")
print()

# Zero-trip weeks: corridors missing from certain weeks
weeks_per_corridor = corr_week.groupby("corridor_id").size()
print("Weeks per corridor:")
print(f"  Min: {weeks_per_corridor.min()}, Median: {weeks_per_corridor.median():.0f}, Max: {weeks_per_corridor.max()}")

corridors_full_weeks = (weeks_per_corridor == n_total_weeks).sum()
print(f"  Corridors with all {n_total_weeks} weeks: {corridors_full_weeks}")
print()

# ── 4. Train/test split ──────────────────────────────────────────────
print("=" * 60)
print("4. TRAIN/TEST SPLIT")
print("=" * 60)

# Train: up to 2025 W26, Test: 2025 W27 - 2025 W52
train_mask = (
    (corr_week["iso_year"] < 2025)
    | ((corr_week["iso_year"] == 2025) & (corr_week["iso_week"] <= 26))
)
test_mask = (corr_week["iso_year"] == 2025) & (corr_week["iso_week"].between(27, 52))

corr_week["split"] = "unused"
corr_week.loc[train_mask, "split"] = "train"
corr_week.loc[test_mask, "split"] = "test"

corr_week_with_tier = corr_week.merge(
    corr_char[["corridor_id", "tier"]], on="corridor_id", how="left"
)

split_by_tier = (
    corr_week_with_tier.groupby(["tier", "split"])
    .agg(
        n_rows=("weekly_trips", "count"),
        avg_weekly_trips=("weekly_trips", "mean"),
        median_weekly_trips=("weekly_trips", "median"),
    )
    .round(1)
)
print(split_by_tier.to_string())
print()

# Weeks per corridor in train vs test
train_weeks = corr_week_with_tier[corr_week_with_tier["split"] == "train"]
test_weeks = corr_week_with_tier[corr_week_with_tier["split"] == "test"]

train_wpc = train_weeks.groupby(["corridor_id", "tier"]).size().reset_index(name="n_train_weeks")
test_wpc = test_weeks.groupby(["corridor_id", "tier"]).size().reset_index(name="n_test_weeks")

split_summary = train_wpc.merge(test_wpc, on=["corridor_id", "tier"], how="outer").fillna(0)

print("Train/test weeks per corridor by tier:")
for tier in ["dense (≥100/day)", "medium (20-99/day)", "sparse (<20/day)"]:
    subset = split_summary[split_summary["tier"] == tier]
    if len(subset) == 0:
        continue
    print(f"\n  {tier} ({len(subset)} corridors):")
    print(f"    Train weeks: min={subset['n_train_weeks'].min():.0f}, "
          f"median={subset['n_train_weeks'].median():.0f}, "
          f"max={subset['n_train_weeks'].max():.0f}")
    print(f"    Test weeks:  min={subset['n_test_weeks'].min():.0f}, "
          f"median={subset['n_test_weeks'].median():.0f}, "
          f"max={subset['n_test_weeks'].max():.0f}")

print()

# ── 5. Trailing mean feasibility ──────────────────────────────────────
print("=" * 60)
print("5. TRAILING MEAN FEASIBILITY")
print("=" * 60)

# For the trailing 8-week mean baseline: at the start of test (W27 2025),
# how many corridors have at least 8 prior weeks of data?
train_weeks_count = train_wpc.set_index("corridor_id")["n_train_weeks"]
has_8w = (train_weeks_count >= 8).sum()
has_4w = (train_weeks_count >= 4).sum()
print(f"Corridors with ≥8 training weeks: {has_8w} / {len(train_weeks_count)}")
print(f"Corridors with ≥4 training weeks: {has_4w} / {len(train_weeks_count)}")
print()

# Check if sparse corridors have consistent enough data for trailing mean
print("Sparse corridor weekly trip stats (training period):")
sparse_train = train_weeks[train_weeks["tier"] == "sparse (<20/day)"]
sparse_cv = (
    sparse_train.groupby("corridor_id")
    .agg(
        mean_trips=("weekly_trips", "mean"),
        std_trips=("weekly_trips", "std"),
        min_trips=("weekly_trips", "min"),
        max_trips=("weekly_trips", "max"),
    )
)
sparse_cv["cv"] = sparse_cv["std_trips"] / sparse_cv["mean_trips"]
print(f"  Coefficient of variation: median={sparse_cv['cv'].median():.2f}, "
      f"max={sparse_cv['cv'].max():.2f}")
print(f"  Mean weekly trips: median={sparse_cv['mean_trips'].median():.1f}, "
      f"min={sparse_cv['mean_trips'].min():.1f}")
print()

# ── 6. Top corridors preview ─────────────────────────────────────────
print("=" * 60)
print("6. TOP 15 CORRIDORS BY DAILY TRIPS")
print("=" * 60)
top15 = corr_char.nlargest(15, "avg_daily_trips")[
    ["corridor_id", "avg_daily_trips", "avg_fare", "avg_miles", "is_airport", "tier"]
]
print(top15.to_string(index=False))
print()

# ── 7. Sparse corridor zero-week issue ────────────────────────────────
print("=" * 60)
print("7. ZERO-TRIP WEEKS IN HOLDOUT (SPARSE CORRIDORS)")
print("=" * 60)
# How many sparse corridors have zero-trip weeks in holdout?
sparse_test = test_weeks[test_weeks["tier"] == "sparse (<20/day)"]
sparse_test_stats = (
    sparse_test.groupby("corridor_id")
    .agg(
        n_weeks=("weekly_trips", "count"),
        zero_weeks=("weekly_trips", lambda x: (x == 0).sum()),
        mean_trips=("weekly_trips", "mean"),
    )
)
sparse_test_stats["zero_pct"] = (
    sparse_test_stats["zero_weeks"] / sparse_test_stats["n_weeks"] * 100
).round(1)

# But also: weeks where corridor is simply absent from the data
# (not a zero row, but no row at all)
n_test_weeks_expected = 26
sparse_missing = test_wpc[test_wpc["tier"] == "sparse (<20/day)"]
sparse_missing["missing_weeks"] = n_test_weeks_expected - sparse_missing["n_test_weeks"]

print(f"Sparse corridors in test: {len(sparse_test_stats)}")
print(f"  With zero-trip weeks (present but 0): median={sparse_test_stats['zero_weeks'].median():.0f}")
print(f"  With missing weeks (absent from data): median={sparse_missing['missing_weeks'].median():.0f}")
print(f"  Zero-trip week %: median={sparse_test_stats['zero_pct'].median():.1f}%")
print()

print("Done. Review output before building models.")
