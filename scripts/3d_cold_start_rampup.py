"""
Phase 5c — Cold-Start Ramp-Up Experiment
==========================================
Hold out 20 corridors entirely from training. Train global model
on remaining 163 corridors. For each held-out corridor, simulate
a new route launch: no historical data, then observe 2, 4, 8 weeks
and measure how quickly forecast accuracy improves.

Deliverable: ramp-up curve showing "weeks since launch → MAE"
Directly actionable for route-level products: how long after a new route launch
does the forecast become reliable?

Run from project root:
    python scripts/3d_cold_start_rampup.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
Path("outputs").mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# DATA PREP (same as 3b)
# ══════════════════════════════════════════════════════════════════════
print("Loading and preparing data...")
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

# Events
holidays["date"] = pd.to_datetime(holidays["date"])
iso_h = holidays["date"].dt.isocalendar()
holidays["iso_year"] = iso_h["year"].astype(int)
holidays["iso_week"] = iso_h["week"].astype(int)
hol_weekly = (
    holidays.groupby(["iso_year", "iso_week"])
    .agg(
        holiday_neg_days=("expected_impact", lambda x: (x == "negative").sum()),
        holiday_pos_days=("expected_impact", lambda x: (x == "positive").sum()),
    )
    .reset_index()
)
conv_rows = []
for _, row in conventions.iterrows():
    for d in pd.date_range(row["date_start"], row["date_end"]):
        conv_rows.append({"date": d, "est_attendees": row["est_attendees"]})
conv_df = pd.DataFrame(conv_rows)
iso_c = conv_df["date"].dt.isocalendar()
conv_df["iso_year"] = iso_c["year"].astype(int)
conv_df["iso_week"] = iso_c["week"].astype(int)
conv_weekly = (
    conv_df.groupby(["iso_year", "iso_week"])
    .agg(convention_days=("date", "count"), max_attendees=("est_attendees", "max"))
    .reset_index()
)

# Tiers & filtering
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

# Weekly aggregation + grid
corr_week = (
    corr_day.groupby(["corridor_id", "pickup_community_area",
                       "dropoff_community_area", "iso_year", "iso_week"])
    .agg(weekly_trips=("trip_count", "sum"))
    .reset_index()
)
corr_week["week_start"] = pd.to_datetime(
    corr_week["iso_year"].astype(str)
    + corr_week["iso_week"].astype(str).str.zfill(2) + "1",
    format="%G%V%u",
)
all_weeks = corr_week[["iso_year", "iso_week", "week_start"]].drop_duplicates().sort_values("week_start")
grids = []
for cid, grp in corr_week.groupby("corridor_id"):
    cw = all_weeks[all_weeks["week_start"].between(grp["week_start"].min(), grp["week_start"].max())].copy()
    cw["corridor_id"] = cid
    cw["pickup_community_area"] = grp.iloc[0]["pickup_community_area"]
    cw["dropoff_community_area"] = grp.iloc[0]["dropoff_community_area"]
    grids.append(cw)
full = pd.concat(grids, ignore_index=True)
full = full.merge(
    corr_week[["corridor_id", "iso_year", "iso_week", "weekly_trips"]],
    on=["corridor_id", "iso_year", "iso_week"], how="left",
)
full["weekly_trips"] = full["weekly_trips"].fillna(0).astype(int)
full["tier"] = full["corridor_id"].map(tier_map)
full = full.sort_values(["corridor_id", "week_start"]).reset_index(drop=True)

# Split
train_mask = (full["iso_year"] < 2025) | ((full["iso_year"] == 2025) & (full["iso_week"] <= 26))
test_mask = (full["iso_year"] == 2025) & full["iso_week"].between(27, 52)
full["split"] = "unused"
full.loc[train_mask, "split"] = "train"
full.loc[test_mask, "split"] = "test"

# Features (same pipeline as 3b)
full = full.merge(hol_weekly, on=["iso_year", "iso_week"], how="left")
full = full.merge(conv_weekly, on=["iso_year", "iso_week"], how="left")
for c in ["holiday_neg_days", "holiday_pos_days", "convention_days", "max_attendees"]:
    full[c] = full[c].fillna(0).astype(int)
full["is_airport_origin"] = (full["pickup_community_area"] == 76).astype(int)
full["is_airport_dest"] = (full["dropoff_community_area"] == 76).astype(int)
full = full.merge(corr_char_f[["corridor_id", "avg_fare", "avg_miles"]], on="corridor_id", how="left")
full["tier_num"] = full["tier"].map({"dense": 2, "medium": 1, "sparse": 0})
full["week_of_year"] = full["iso_week"]

for lag in [1, 2, 4]:
    full[f"lag_{lag}"] = full.groupby("corridor_id")["weekly_trips"].shift(lag)
full["rolling_mean_4"] = full.groupby("corridor_id")["weekly_trips"].transform(
    lambda x: x.shift(1).rolling(4, min_periods=2).mean()
)
full["rolling_std_4"] = full.groupby("corridor_id")["weekly_trips"].transform(
    lambda x: x.shift(1).rolling(4, min_periods=2).std()
)
full["rolling_std_4"] = full["rolling_std_4"].fillna(0)

train_mean = (
    full.loc[full["split"] == "train"]
    .groupby("corridor_id")["weekly_trips"].mean()
    .reset_index().rename(columns={"weekly_trips": "train_mean_weekly"})
)
full = full.merge(train_mean, on="corridor_id", how="left")
full["log_train_mean"] = np.log1p(full["train_mean_weekly"])

FEATURES = [
    "avg_fare", "avg_miles", "is_airport_origin", "is_airport_dest",
    "tier_num", "log_train_mean", "week_of_year",
    "holiday_neg_days", "holiday_pos_days", "convention_days", "max_attendees",
    "lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4",
]
HISTORY_FEATURES = ["lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4"]

print("Data prep complete.")

# ══════════════════════════════════════════════════════════════════════
# SELECT HELD-OUT CORRIDORS
# ══════════════════════════════════════════════════════════════════════
np.random.seed(42)
all_corridors = corr_char_f["corridor_id"].values
dense_ids = corr_char_f[corr_char_f["tier"] == "dense"]["corridor_id"].values
medium_ids = corr_char_f[corr_char_f["tier"] == "medium"]["corridor_id"].values
sparse_ids = corr_char_f[corr_char_f["tier"] == "sparse"]["corridor_id"].values

holdout_ids = np.concatenate([
    np.random.choice(dense_ids, size=3, replace=False),
    np.random.choice(medium_ids, size=5, replace=False),
    np.random.choice(sparse_ids, size=12, replace=False),
])
train_ids = set(all_corridors) - set(holdout_ids)

print(f"\nHeld out: {len(holdout_ids)} corridors "
      f"(3 dense, 5 medium, 12 sparse)")
print(f"Training: {len(train_ids)} corridors")

# ══════════════════════════════════════════════════════════════════════
# TRAIN COLD-START MODEL (163 corridors)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TRAINING: Global XGBoost on 163 corridors")
print("=" * 60)

train_df = (
    full[(full["split"] == "train") & (full["corridor_id"].isin(train_ids))]
    .dropna(subset=FEATURES)
)
print(f"  Training rows: {len(train_df):,}")

cold_model = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
)
cold_model.fit(train_df[FEATURES], train_df["weekly_trips"])
print("  Model trained.")

# ══════════════════════════════════════════════════════════════════════
# REFERENCE: Held-out corridors with FULL history
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("REFERENCE: Held-out corridors with full history")
print("=" * 60)

test_holdout = (
    full[(full["split"] == "test") & (full["corridor_id"].isin(holdout_ids))]
    .dropna(subset=FEATURES).copy()
)
test_holdout["ref_pred"] = cold_model.predict(test_holdout[FEATURES]).clip(min=0)

ref_mae = np.abs(test_holdout["weekly_trips"] - test_holdout["ref_pred"]).mean()
print(f"  Full-history MAE: {ref_mae:.1f} (model never trained on these corridors)")

# ══════════════════════════════════════════════════════════════════════
# COLD-START RAMP-UP: rebuild features from test actuals only
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COLD-START RAMP-UP SIMULATION")
print("=" * 60)

ramp_results = []

for cid in holdout_ids:
    cid_test = (
        full[(full["split"] == "test") & (full["corridor_id"] == cid)]
        .sort_values("week_start").copy()
    )
    if len(cid_test) == 0:
        continue

    n_weeks = len(cid_test)
    actuals = cid_test["weekly_trips"].values
    tier = cid_test["tier"].iloc[0]

    # Build cold features: lag/rolling from test-period actuals only
    cold_lag_1 = np.full(n_weeks, np.nan)
    cold_lag_2 = np.full(n_weeks, np.nan)
    cold_lag_4 = np.full(n_weeks, np.nan)
    cold_rm4 = np.full(n_weeks, np.nan)
    cold_rs4 = np.full(n_weeks, np.nan)

    for t in range(n_weeks):
        if t >= 1:
            cold_lag_1[t] = actuals[t - 1]
        if t >= 2:
            cold_lag_2[t] = actuals[t - 2]
        if t >= 4:
            cold_lag_4[t] = actuals[t - 4]
        # rolling_mean_4: mean of previous 4 weeks (or as many as available, min 2)
        if t >= 2:
            start = max(0, t - 4)
            window = actuals[start:t]
            cold_rm4[t] = np.mean(window)
            cold_rs4[t] = np.std(window) if len(window) >= 2 else 0

    # Replace history features with cold versions
    cid_cold = cid_test.copy()
    cid_cold["lag_1"] = cold_lag_1
    cid_cold["lag_2"] = cold_lag_2
    cid_cold["lag_4"] = cold_lag_4
    cid_cold["rolling_mean_4"] = cold_rm4
    cid_cold["rolling_std_4"] = cold_rs4
    # log_train_mean: keep original (simulates market-sizing estimate)

    # Predict all weeks
    X_cold = cid_cold[FEATURES].copy()
    cid_cold["cold_pred"] = cold_model.predict(X_cold).clip(min=0)

    # Also compute trailing mean (cold version: cumulative mean of observed)
    cold_trailing = np.full(n_weeks, np.nan)
    for t in range(1, n_weeks):
        cold_trailing[t] = np.mean(actuals[:t])
    cid_cold["trailing_cold"] = cold_trailing

    # Full-history reference prediction
    cid_ref = test_holdout[test_holdout["corridor_id"] == cid]

    # Store per-week results
    for t in range(n_weeks):
        row = {
            "corridor_id": cid,
            "tier": tier,
            "week_index": t,  # 0 = first test week (just launched)
            "actual": actuals[t],
            "cold_pred": cid_cold["cold_pred"].iloc[t],
            "trailing_cold": cold_trailing[t],
            "avg_weekly": np.mean(actuals),
        }
        # Match reference prediction
        week_row = cid_cold.iloc[t]
        ref_match = cid_ref[
            (cid_ref["iso_year"] == week_row["iso_year"])
            & (cid_ref["iso_week"] == week_row["iso_week"])
        ]
        row["ref_pred"] = ref_match["ref_pred"].values[0] if len(ref_match) > 0 else np.nan
        ramp_results.append(row)

ramp_df = pd.DataFrame(ramp_results)
ramp_df["cold_error"] = np.abs(ramp_df["actual"] - ramp_df["cold_pred"])
ramp_df["trailing_error"] = np.abs(ramp_df["actual"] - ramp_df["trailing_cold"])
ramp_df["ref_error"] = np.abs(ramp_df["actual"] - ramp_df["ref_pred"])

# ══════════════════════════════════════════════════════════════════════
# RAMP-UP CURVE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RAMP-UP CURVE: MAE by weeks since route launch")
print("=" * 60)

# Bin by weeks of observation
bins = [
    ("Week 1-2 (no/minimal history)", 0, 2),
    ("Week 3-4 (2-4 wks observed)", 2, 4),
    ("Week 5-8 (4-8 wks observed)", 4, 8),
    ("Week 9-26 (8+ wks observed)", 8, 26),
]

print(f"\n{'Window':<35s} {'Cold XGB':>10} {'Trail Mean':>10} {'Full Hist':>10} {'n_preds':>8}")
print("-" * 78)

for label, lo, hi in bins:
    b = ramp_df[(ramp_df["week_index"] >= lo) & (ramp_df["week_index"] < hi)]
    cold_mae = b["cold_error"].mean()
    trail_mae = b["trailing_error"].mean()
    ref_mae_bin = b["ref_error"].mean()
    print(f"{label:<35s} {cold_mae:>10.1f} {trail_mae:>10.1f} {ref_mae_bin:>10.1f} {len(b):>8}")

# Overall
print("-" * 78)
print(f"{'ALL (26 weeks)':<35s} {ramp_df['cold_error'].mean():>10.1f} "
      f"{ramp_df['trailing_error'].dropna().mean():>10.1f} "
      f"{ramp_df['ref_error'].mean():>10.1f} {len(ramp_df):>8}")

# ══════════════════════════════════════════════════════════════════════
# RAMP-UP BY TIER
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RAMP-UP BY TIER")
print("=" * 60)

for tier_name in ["dense", "medium", "sparse"]:
    tier_data = ramp_df[ramp_df["tier"] == tier_name]
    n_corr = tier_data["corridor_id"].nunique()
    print(f"\n  {tier_name.upper()} ({n_corr} corridors):")
    print(f"  {'Window':<35s} {'Cold XGB':>10} {'Trail Mean':>10} {'Full Hist':>10}")
    print(f"  {'-' * 68}")
    for label, lo, hi in bins:
        b = tier_data[(tier_data["week_index"] >= lo) & (tier_data["week_index"] < hi)]
        if len(b) == 0:
            continue
        cold_mae = b["cold_error"].mean()
        trail_mae = b["trailing_error"].mean()
        ref_mae_bin = b["ref_error"].mean()
        print(f"  {label:<35s} {cold_mae:>10.1f} {trail_mae:>10.1f} {ref_mae_bin:>10.1f}")

# ══════════════════════════════════════════════════════════════════════
# PER-CORRIDOR SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PER-CORRIDOR: cold-start vs full-history")
print("=" * 60)

for cid in sorted(holdout_ids):
    c = ramp_df[ramp_df["corridor_id"] == cid]
    tier = c["tier"].iloc[0]
    avg_w = c["avg_weekly"].iloc[0]
    cold_overall = c["cold_error"].mean()
    ref_overall = c["ref_error"].mean()
    # MAE at weeks 9-26 (after ramp)
    late = c[c["week_index"] >= 8]
    cold_late = late["cold_error"].mean() if len(late) > 0 else np.nan
    ref_late = late["ref_error"].mean() if len(late) > 0 else np.nan
    print(f"  {cid:>10s} [{tier:>6s}]  avg={avg_w:>6.0f}/wk  "
          f"cold_all={cold_overall:>6.1f}  cold_9+={cold_late:>6.1f}  "
          f"ref={ref_overall:>6.1f}")

# ══════════════════════════════════════════════════════════════════════
# CONVERGENCE: at what week does cold-start match full-history?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CONVERGENCE: week-by-week cold vs full-history MAE")
print("=" * 60)
print(f"  {'Week':>4s}  {'Cold MAE':>10s}  {'Full MAE':>10s}  {'Gap':>8s}  {'Gap%':>6s}")
print(f"  {'-' * 44}")
for w in range(26):
    wk = ramp_df[ramp_df["week_index"] == w]
    cold_w = wk["cold_error"].mean()
    ref_w = wk["ref_error"].mean()
    gap = cold_w - ref_w
    gap_pct = gap / ref_w * 100 if ref_w > 0 else np.nan
    marker = " ←" if gap_pct is not None and not np.isnan(gap_pct) and abs(gap_pct) < 20 else ""
    print(f"  {w + 1:>4d}  {cold_w:>10.1f}  {ref_w:>10.1f}  {gap:>8.1f}  {gap_pct:>5.1f}%{marker}")

# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
ramp_df.to_csv("outputs/phase5_rampup_detail.csv", index=False)
print(f"\nSaved: outputs/phase5_rampup_detail.csv")
print("\nDone.")
