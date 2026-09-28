"""
Phase 5b — Ablation & Cold-Start Simulation
=============================================
Test 1: Per-corridor XGBoost (same features, trained per corridor)
    → Isolates whether global XGB's advantage is cross-corridor
      learning or better feature engineering.

Test 2: Simulated cold-start (lag/rolling/scale → NaN for 20 corridors)
    → Demonstrates global model's value for zero-history routes.

Run from project root:
    python scripts/3c_corridor_ablation.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
Path("outputs").mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# DATA PREP (same as 3b_corridor_forecast.py)
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
print(f"  {len(filtered_ids)} corridors after filtering")

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

# Features
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

# Baseline
full["baseline_pred"] = full.groupby("corridor_id")["weekly_trips"].transform(
    lambda x: x.shift(1).rolling(8, min_periods=1).mean()
)

FEATURES = [
    "avg_fare", "avg_miles", "is_airport_origin", "is_airport_dest",
    "tier_num", "log_train_mean", "week_of_year",
    "holiday_neg_days", "holiday_pos_days", "convention_days", "max_attendees",
    "lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4",
]

train_df = full[full["split"] == "train"].dropna(subset=FEATURES).copy()
test_all = full[full["split"] == "test"].copy()
test_df = test_all.dropna(subset=FEATURES).copy()

print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print("Data prep complete.\n")

# ══════════════════════════════════════════════════════════════════════
# MODEL 1: GLOBAL XGBoost (same as 3b)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODEL 1: GLOBAL XGBoost")
print("=" * 60)

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
)

global_model = XGBRegressor(**XGB_PARAMS)
global_model.fit(train_df[FEATURES], train_df["weekly_trips"])
test_df["global_xgb_pred"] = global_model.predict(test_df[FEATURES]).clip(min=0)
print("  Trained on all corridors combined.")

# ══════════════════════════════════════════════════════════════════════
# MODEL 2: PER-CORRIDOR XGBoost
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2: PER-CORRIDOR XGBoost")
print("=" * 60)

MIN_TRAIN_ROWS = 20  # skip corridors with too few rows
pc_preds = []
skipped = 0

corridors = test_df["corridor_id"].unique()
for cid in corridors:
    cid_train = train_df[train_df["corridor_id"] == cid]
    cid_test = test_df[test_df["corridor_id"] == cid].copy()

    if len(cid_train) < MIN_TRAIN_ROWS:
        skipped += 1
        cid_test["pc_xgb_pred"] = np.nan
        pc_preds.append(cid_test)
        continue

    pc_model = XGBRegressor(**XGB_PARAMS)
    pc_model.fit(cid_train[FEATURES], cid_train["weekly_trips"])
    cid_test["pc_xgb_pred"] = pc_model.predict(cid_test[FEATURES]).clip(min=0)
    pc_preds.append(cid_test)

pc_results = pd.concat(pc_preds, ignore_index=True)
print(f"  Trained {len(corridors) - skipped} individual models, skipped {skipped}")

# Merge per-corridor predictions into test_df
test_df = test_df.merge(
    pc_results[["corridor_id", "iso_year", "iso_week", "pc_xgb_pred"]],
    on=["corridor_id", "iso_year", "iso_week"], how="left",
)

# ══════════════════════════════════════════════════════════════════════
# TEST 3: COLD-START SIMULATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: COLD-START SIMULATION")
print("=" * 60)

# Select 20 medium/sparse corridors
med_sparse = test_df[test_df["tier"].isin(["medium", "sparse"])]["corridor_id"].unique()
np.random.seed(42)
cold_cids = np.random.choice(med_sparse, size=min(20, len(med_sparse)), replace=False)
print(f"  Selected {len(cold_cids)} corridors for cold-start test")

cold_test = test_df[test_df["corridor_id"].isin(cold_cids)].copy()

# Zero out all corridor-history features
HISTORY_FEATURES = ["lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4", "log_train_mean"]
cold_features = cold_test[FEATURES].copy()
cold_features[HISTORY_FEATURES] = np.nan

cold_test["cold_start_pred"] = global_model.predict(cold_features).clip(min=0)

# What features remain for cold-start?
remaining = [f for f in FEATURES if f not in HISTORY_FEATURES]
print(f"  History features zeroed: {HISTORY_FEATURES}")
print(f"  Remaining features: {remaining}")

# ══════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════

def eval_preds(df, pred_col):
    v = df.dropna(subset=[pred_col])
    if len(v) == 0:
        return dict(n=0, mae=np.nan, mape=np.nan, cov30=np.nan)
    a, p = v["weekly_trips"].values, v[pred_col].values
    err = np.abs(a - p)
    nz = a > 0
    mape = (err[nz] / a[nz]).mean() * 100 if nz.sum() else np.nan
    within = np.zeros(len(a), dtype=bool)
    within[nz] = err[nz] / a[nz] <= 0.30
    within[~nz] = p[~nz] == 0
    return dict(n=len(v), mae=err.mean(), mape=mape, cov30=within.mean() * 100)


# ── A. Three-model tier comparison ───────────────────────────────────
print("\n" + "=" * 60)
print("A. THREE-MODEL COMPARISON BY TIER")
print("=" * 60)
print(f"{'tier':<8} {'corr':>5} {'':>3} {'baseline':>10} {'pc_xgb':>10} {'global_xgb':>10}  "
      f"{'baseline':>10} {'pc_xgb':>10} {'global_xgb':>10}")
print(f"{'':8} {'':>5} {'':>3} {'--- MAE ---':>31}  {'--- COV30% ---':>31}")

for tier_name in ["dense", "medium", "sparse", "ALL"]:
    s = test_df if tier_name == "ALL" else test_df[test_df["tier"] == tier_name]
    bm = eval_preds(s, "baseline_pred")
    pm = eval_preds(s, "pc_xgb_pred")
    gm = eval_preds(s, "global_xgb_pred")
    print(f"{tier_name:<8} {s['corridor_id'].nunique():>5}   "
          f"{bm['mae']:>10.1f} {pm['mae']:>10.1f} {gm['mae']:>10.1f}  "
          f"{bm['cov30']:>10.1f} {pm['cov30']:>10.1f} {gm['cov30']:>10.1f}")

# ── B. Per-corridor: global vs per-corridor XGBoost ──────────────────
print("\n" + "=" * 60)
print("B. GLOBAL vs PER-CORRIDOR XGBoost (per corridor)")
print("=" * 60)

crows = []
for cid, g in test_df.groupby("corridor_id"):
    gm = eval_preds(g, "global_xgb_pred")
    pm = eval_preds(g, "pc_xgb_pred")
    crows.append(dict(
        corridor_id=cid, tier=g["tier"].iloc[0],
        avg_weekly=round(g["weekly_trips"].mean(), 1),
        global_mae=round(gm["mae"], 1),
        pc_mae=round(pm["mae"], 1) if not np.isnan(pm["mae"]) else np.nan,
        global_wins=1 if gm["mae"] < pm["mae"] else 0,
    ))
cdf = pd.DataFrame(crows)

for t in ["dense", "medium", "sparse"]:
    sub = cdf[cdf["tier"] == t].dropna(subset=["pc_mae"])
    gw = (sub["global_wins"] == 1).sum()
    diff = ((sub["pc_mae"] - sub["global_mae"]) / sub["pc_mae"] * 100)
    print(f"\n  {t.upper()} ({len(sub)} corridors):")
    print(f"    Global wins: {gw}/{len(sub)}")
    print(f"    Global advantage (MAE reduction vs per-corridor): "
          f"median {diff.median():.1f}%, mean {diff.mean():.1f}%")

# ── C. Cold-start results ────────────────────────────────────────────
print("\n" + "=" * 60)
print("C. COLD-START SIMULATION (20 corridors, no history features)")
print("=" * 60)

# Compare: full global model vs cold-start global model vs baseline
cold_merged = cold_test.copy()
# Get global and baseline preds for same corridors
cold_global = test_df[test_df["corridor_id"].isin(cold_cids)]

full_model = eval_preds(cold_global, "global_xgb_pred")
baseline = eval_preds(cold_global, "baseline_pred")
cold_start = eval_preds(cold_test, "cold_start_pred")

print(f"\n  {'Model':<30s} {'MAE':>8} {'MAPE':>8} {'COV30%':>8}")
print(f"  {'-'*56}")
print(f"  {'Trailing mean (has history)':<30s} {baseline['mae']:>8.1f} {baseline['mape']:>8.1f} {baseline['cov30']:>8.1f}")
print(f"  {'Global XGB (has history)':<30s} {full_model['mae']:>8.1f} {full_model['mape']:>8.1f} {full_model['cov30']:>8.1f}")
print(f"  {'Global XGB (COLD START)':<30s} {cold_start['mae']:>8.1f} {cold_start['mape']:>8.1f} {cold_start['cov30']:>8.1f}")
print(f"  {'Per-corridor (COLD START)':<30s} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

# Per-corridor cold-start detail
print(f"\n  Per-corridor cold-start detail:")
for cid in sorted(cold_cids):
    g = cold_test[cold_test["corridor_id"] == cid]
    actual_mean = g["weekly_trips"].mean()
    cs_mae = np.abs(g["weekly_trips"] - g["cold_start_pred"]).mean()
    cs_mape_vals = np.abs(g["weekly_trips"] - g["cold_start_pred"]) / g["weekly_trips"].replace(0, np.nan)
    cs_mape = cs_mape_vals.mean() * 100
    print(f"    {cid:>10s}  avg_weekly={actual_mean:>6.1f}  cold_MAE={cs_mae:>6.1f}  cold_MAPE={cs_mape:>5.1f}%")

# ── D. Summary finding ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Three-layer finding:
  1. At sufficient data density, per-corridor XGBoost matches
     global XGBoost — cross-corridor learning adds little value.
  2. Global XGBoost beats trailing mean through better feature
     engineering (rolling mean + scale), not cross-corridor learning.
  3. At cold-start (zero history), only the global model can
     produce predictions using route characteristics. Per-corridor
     models cannot predict at all.
""")

# ── Save ─────────────────────────────────────────────────────────────
cdf.to_csv("outputs/phase5_ablation_corridor_detail.csv", index=False)
print("Saved: outputs/phase5_ablation_corridor_detail.csv")
print("Done.")
