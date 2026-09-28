"""
Phase 5d — Daily Corridor Forecasting
=======================================
Per-corridor trailing same-DOW mean vs Global XGBoost (daily).
Results aggregated to weekly for unified comparison with Phase 4 & 5.

Key additions over weekly model:
  - Day-of-week as cross-corridor feature (40% daily swing)
  - Holiday/convention factors from Phase 3 (validated daily impacts)
  - 110K training rows (8x more than weekly)
  - Direct comparison with TimesFM (8.20% weekly MAPE)

Run from project root:
    python scripts/3e_corridor_daily.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
Path("outputs").mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# PHASE 3 VALIDATED FACTORS (learned from NNS daily M0 baseline)
# ══════════════════════════════════════════════════════════════════════
# Holiday factors: multiplier on expected demand (1.0 = normal)
HOLIDAY_FACTORS = {
    "Christmas": 0.254,
    "Thanksgiving": 0.390,
    "Day After Christmas": 0.482,
    "July 4th": 0.540,
    "Christmas Eve": 0.628,
    "Memorial Day": 0.637,
    "Black Friday": 0.639,
    "Labor Day": 0.698,
    "St Patrick's Parade Chicago": 1.460,
    "St Patrick's Day": 1.294,
    "St Patrick's Sunday": 1.294,
}

# Convention factors: multiplier on expected demand
CONVENTION_FACTORS = {
    "National Restaurant Association Show": 1.236,
    "RSNA Annual Meeting": 1.179,
    "ProMat": 1.165,
    "PACK EXPO International": 1.123,
    "ASCO Annual Meeting": 1.121,
}

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE
# ══════════════════════════════════════════════════════════════════════
print("Loading data...")
corr_day = pd.read_csv("data/derived/corridor_day.csv")
corr_char = pd.read_csv("data/derived/corridor_characteristics.csv")
holidays = pd.read_csv("data/external/holidays_chicago.csv")
conventions = pd.read_csv("data/external/conventions_chicago.csv")

corr_day["date"] = pd.to_datetime(corr_day["date"])
for df in [corr_day, corr_char]:
    df["corridor_id"] = (
        df["pickup_community_area"].astype(str) + "→"
        + df["dropoff_community_area"].astype(str)
    )

# ── Holiday factor lookup (date → factor) ────────────────────────────
holidays["date"] = pd.to_datetime(holidays["date"])
holiday_date_factor = {}
for _, row in holidays.iterrows():
    name = row["event_name"]
    if name in HOLIDAY_FACTORS:
        holiday_date_factor[row["date"]] = HOLIDAY_FACTORS[name]

# ── Convention factor lookup (date → factor) ─────────────────────────
conv_date_factor = {}
for _, row in conventions.iterrows():
    name = row["event_name"]
    if name in CONVENTION_FACTORS:
        for d in pd.date_range(row["date_start"], row["date_end"]):
            conv_date_factor[d] = CONVENTION_FACTORS[name]

print(f"  Holiday dates with factors: {len(holiday_date_factor)}")
print(f"  Convention dates with factors: {len(conv_date_factor)}")

# ── Tiers & filtering (same as weekly) ───────────────────────────────
def assign_tier(v):
    return "dense" if v >= 100 else ("medium" if v >= 20 else "sparse")

corr_char["tier"] = corr_char["avg_daily_trips"].apply(assign_tier)
total_days = (corr_day["date"].max() - corr_day["date"].min()).days + 1
corr_char["coverage_pct"] = corr_char["n_days"] / total_days * 100
corr_char["avg_weekly_trips"] = corr_char["avg_daily_trips"] * 7

sparse = corr_char["tier"] == "sparse"
keep = ~sparse | (sparse & (corr_char["coverage_pct"] > 80) & (corr_char["avg_weekly_trips"] > 5))
filtered_ids = set(corr_char.loc[keep, "corridor_id"])
corr_day = corr_day[corr_day["corridor_id"].isin(filtered_ids)].copy()
corr_char_f = corr_char[corr_char["corridor_id"].isin(filtered_ids)].copy()
tier_map = corr_char_f.set_index("corridor_id")["tier"].to_dict()

print(f"  Corridors after filtering: {len(filtered_ids)}")

# ══════════════════════════════════════════════════════════════════════
# 2. FEATURES
# ══════════════════════════════════════════════════════════════════════
print("Computing features...")
corr_day = corr_day.sort_values(["corridor_id", "date"]).reset_index(drop=True)
corr_day["tier"] = corr_day["corridor_id"].map(tier_map)

# 2a. Day-of-week (0=Mon, 6=Sun)
corr_day["dow"] = corr_day["weekday"] - 1  # iso_weekday is 1=Mon

# 2b. Holiday & convention factors
corr_day["holiday_factor"] = corr_day["date"].map(holiday_date_factor).fillna(1.0)
corr_day["convention_factor"] = corr_day["date"].map(conv_date_factor).fillna(1.0)

# 2c. Corridor statics
corr_day["is_airport_origin"] = (corr_day["pickup_community_area"] == 76).astype(int)
corr_day["is_airport_dest"] = (corr_day["dropoff_community_area"] == 76).astype(int)
corr_day = corr_day.merge(
    corr_char_f[["corridor_id", "avg_fare", "avg_miles"]], on="corridor_id", how="left"
)
corr_day["tier_num"] = corr_day["tier"].map({"dense": 2, "medium": 1, "sparse": 0})

# 2d. Temporal
corr_day["week_of_year"] = corr_day["iso_week"]
corr_day["month"] = corr_day["date"].dt.month

# 2e. Lag features (per corridor)
corr_day["lag_7"] = corr_day.groupby("corridor_id")["trip_count"].shift(7)
corr_day["lag_14"] = corr_day.groupby("corridor_id")["trip_count"].shift(14)
corr_day["lag_21"] = corr_day.groupby("corridor_id")["trip_count"].shift(21)
corr_day["lag_28"] = corr_day.groupby("corridor_id")["trip_count"].shift(28)

# Same-DOW rolling mean: mean of past 4 same-weekday observations
corr_day["rolling_same_dow_4"] = corr_day[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(
    axis=1, skipna=True
)

# General rolling mean (past 7 days, shifted by 1)
corr_day["rolling_mean_7"] = (
    corr_day.groupby("corridor_id")["trip_count"]
    .transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
)

# 2f. Train/test split
train_mask = (corr_day["iso_year"] < 2025) | (
    (corr_day["iso_year"] == 2025) & (corr_day["iso_week"] <= 26)
)
test_mask = (corr_day["iso_year"] == 2025) & corr_day["iso_week"].between(27, 52)
corr_day["split"] = "unused"
corr_day.loc[train_mask, "split"] = "train"
corr_day.loc[test_mask, "split"] = "test"

# 2g. Scale feature (training-only)
train_mean = (
    corr_day.loc[corr_day["split"] == "train"]
    .groupby("corridor_id")["trip_count"].mean()
    .reset_index().rename(columns={"trip_count": "train_mean_daily"})
)
corr_day = corr_day.merge(train_mean, on="corridor_id", how="left")
corr_day["log_train_mean"] = np.log1p(corr_day["train_mean_daily"])

# ══════════════════════════════════════════════════════════════════════
# 3. BASELINE: Per-corridor trailing 4-week same-DOW mean
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BASELINE: Per-corridor trailing 4-week same-DOW mean")
print("=" * 60)

# For each day, baseline = mean of same weekday over past 4 weeks
# This equals rolling_same_dow_4 (already computed in features)
corr_day["baseline_pred"] = corr_day["rolling_same_dow_4"]

test_data = corr_day[corr_day["split"] == "test"].copy()
print(f"  Test rows: {len(test_data):,}")
print(f"  NaN baseline: {test_data['baseline_pred'].isna().sum()}")

# ══════════════════════════════════════════════════════════════════════
# 4. GLOBAL XGBoost (daily, cross-corridor)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GLOBAL MODEL: XGBoost (daily, cross-corridor)")
print("=" * 60)

FEATURES = [
    "dow", "week_of_year", "month",
    "holiday_factor", "convention_factor",
    "is_airport_origin", "is_airport_dest",
    "avg_fare", "avg_miles", "tier_num", "log_train_mean",
    "lag_7", "lag_14",
    "rolling_same_dow_4", "rolling_mean_7",
]

train_df = corr_day[corr_day["split"] == "train"].dropna(subset=FEATURES).copy()
test_df = test_data.dropna(subset=FEATURES).copy()

print(f"  Train: {len(train_df):,} rows, {train_df['corridor_id'].nunique()} corridors")
print(f"  Test:  {len(test_df):,} rows, {test_df['corridor_id'].nunique()} corridors")

model = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
)
model.fit(train_df[FEATURES], train_df["trip_count"])
test_df["xgb_pred"] = model.predict(test_df[FEATURES]).clip(min=0)

# Merge back
test_data = test_data.merge(
    test_df[["corridor_id", "date", "xgb_pred"]],
    on=["corridor_id", "date"], how="left",
)

# ══════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
print("\nFeature importance (gain):")
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, val in imp.items():
    print(f"  {feat:<22s} {val:.3f}  {'█' * int(val * 50)}")

# ══════════════════════════════════════════════════════════════════════
# 6. AGGREGATE TO WEEKLY + EVALUATE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EVALUATION: Daily predictions aggregated to weekly")
print("=" * 60)


def weekly_eval(df, pred_col):
    """Aggregate daily predictions to weekly, compute MAE and MAPE."""
    valid = df.dropna(subset=[pred_col]).copy()
    if len(valid) == 0:
        return dict(n_weeks=0, weekly_mae=np.nan, weekly_mape=np.nan, daily_mape=np.nan)

    # Daily MAPE
    nz = valid["trip_count"] > 0
    daily_mape = (
        np.abs(valid.loc[nz, "trip_count"] - valid.loc[nz, pred_col])
        / valid.loc[nz, "trip_count"]
    ).mean() * 100

    # Weekly aggregation
    weekly = valid.groupby(["corridor_id", "iso_year", "iso_week"]).agg(
        actual=("trip_count", "sum"),
        predicted=(pred_col, "sum"),
    ).reset_index()
    weekly["error"] = np.abs(weekly["actual"] - weekly["predicted"])
    nz_w = weekly["actual"] > 0
    weekly_mape = (weekly.loc[nz_w, "error"] / weekly.loc[nz_w, "actual"]).mean() * 100
    weekly_mae = weekly["error"].mean()

    return dict(
        n_weeks=len(weekly),
        weekly_mae=weekly_mae,
        weekly_mape=weekly_mape,
        daily_mape=daily_mape,
    )


# By tier
print(f"\n  {'Tier':<8} {'Corr':>5}  {'Baseline MAPE':>14} {'XGB MAPE':>10} {'Baseline MAE':>13} {'XGB MAE':>9}")
print(f"  {'-' * 64}")

for tier_name in ["dense", "medium", "sparse", "ALL"]:
    s = test_data if tier_name == "ALL" else test_data[test_data["tier"] == tier_name]
    bm = weekly_eval(s, "baseline_pred")
    xm = weekly_eval(s, "xgb_pred")
    n_c = s["corridor_id"].nunique()
    print(f"  {tier_name:<8} {n_c:>5}  {bm['weekly_mape']:>13.1f}% {xm['weekly_mape']:>9.1f}% "
          f"{bm['weekly_mae']:>13.1f} {xm['weekly_mae']:>9.1f}")

# ══════════════════════════════════════════════════════════════════════
# 7. NNS-ONLY COMPARISON (for Phase 4 unified table)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("NNS-ONLY: For unified benchmark with Phase 4")
print("=" * 60)

# NNS corridors: any corridor starting with "8→"
nns_corridors = [c for c in test_data["corridor_id"].unique() if c.startswith("8→")]
nns_test = test_data[test_data["corridor_id"].isin(nns_corridors)]

# Also: total NNS (all corridors with origin=8, summed)
nns_total = nns_test.groupby(["iso_year", "iso_week"]).agg(
    actual=("trip_count", "sum"),
).reset_index()

nns_xgb = nns_test.dropna(subset=["xgb_pred"]).groupby(["iso_year", "iso_week"]).agg(
    predicted=("xgb_pred", "sum"),
    actual=("trip_count", "sum"),
).reset_index()

nns_baseline = nns_test.dropna(subset=["baseline_pred"]).groupby(["iso_year", "iso_week"]).agg(
    predicted=("baseline_pred", "sum"),
    actual=("trip_count", "sum"),
).reset_index()

nz = nns_xgb["actual"] > 0
nns_xgb_mape = (np.abs(nns_xgb.loc[nz, "actual"] - nns_xgb.loc[nz, "predicted"]) / nns_xgb.loc[nz, "actual"]).mean() * 100

nz_b = nns_baseline["actual"] > 0
nns_bl_mape = (np.abs(nns_baseline.loc[nz_b, "actual"] - nns_baseline.loc[nz_b, "predicted"]) / nns_baseline.loc[nz_b, "actual"]).mean() * 100

print(f"\n  NNS (origin area 8) — {len(nns_corridors)} corridors summed to area level:")
print(f"  Baseline (trailing 4-wk same-DOW):  {nns_bl_mape:.2f}% weekly MAPE")
print(f"  Global XGBoost (daily, corridors):  {nns_xgb_mape:.2f}% weekly MAPE")
print(f"  TimesFM zero-shot (Phase 4):         8.20% weekly MAPE")
print(f"  Weekly trend × seasonal (baseline):  6.01% weekly MAPE")

# ══════════════════════════════════════════════════════════════════════
# 8. UNIFIED BENCHMARK TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("UNIFIED BENCHMARK TABLE (all phases)")
print("=" * 60)

print(f"""
  {'Model':<45s} {'Scope':<20s} {'Weekly MAPE':>12}
  {'-' * 80}
  {'Trend × seasonal':<45s} {'NNS weekly':<20s} {'6.01%':>12}
  {'M1 (DOW×month×holiday)':<45s} {'NNS daily→weekly':<20s} {'6.95%':>12}
  {'TimesFM zero-shot':<45s} {'NNS daily→weekly':<20s} {'8.20%':>12}
  {'N-BEATS trained':<45s} {'NNS daily→weekly':<20s} {'10.23%':>12}
  {'Global XGB (daily corridors, NNS sum)':<45s} {'Corridors→NNS':<20s} {f'{nns_xgb_mape:.2f}%':>12}
  {'Trailing 4-wk same-DOW':<45s} {'Corridors→NNS':<20s} {f'{nns_bl_mape:.2f}%':>12}
""")

# ══════════════════════════════════════════════════════════════════════
# 9. SAVE
# ══════════════════════════════════════════════════════════════════════
# Per-corridor results
crows = []
for cid, g in test_data.groupby("corridor_id"):
    bm = weekly_eval(g, "baseline_pred")
    xm = weekly_eval(g, "xgb_pred")
    crows.append(dict(
        corridor_id=cid, tier=g["tier"].iloc[0],
        baseline_weekly_mape=round(bm["weekly_mape"], 1) if bm["weekly_mape"] else np.nan,
        xgb_weekly_mape=round(xm["weekly_mape"], 1) if xm["weekly_mape"] else np.nan,
    ))
cdf = pd.DataFrame(crows)
cdf.to_csv("outputs/phase5_daily_corridor_detail.csv", index=False)
print("Saved: outputs/phase5_daily_corridor_detail.csv")
print("\nDone.")
