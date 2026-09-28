"""
Phase 5 — Weekly Corridor Forecasting
=======================================
Per-corridor trailing mean vs Global XGBoost (weekly granularity).

Includes: sparse filtering, holiday/convention features, split airport
flags, tier as feature, training-only scale (no leakage).

Run from project root:
    python scripts/3b_corridor_forecast.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
Path("outputs").mkdir(exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────
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

# ── 2. Event features → per-ISO-week ────────────────────────────────
print("Building event features...")

# Holidays
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

# Conventions — expand date ranges to individual dates
conv_rows = []
for _, row in conventions.iterrows():
    for d in pd.date_range(row["date_start"], row["date_end"]):
        conv_rows.append({
            "date": d,
            "est_attendees": row["est_attendees"],
        })
conv_df = pd.DataFrame(conv_rows)
iso_c = conv_df["date"].dt.isocalendar()
conv_df["iso_year"] = iso_c["year"].astype(int)
conv_df["iso_week"] = iso_c["week"].astype(int)

conv_weekly = (
    conv_df.groupby(["iso_year", "iso_week"])
    .agg(convention_days=("date", "count"), max_attendees=("est_attendees", "max"))
    .reset_index()
)
print(f"  Holiday weeks: {len(hol_weekly)}, Convention weeks: {len(conv_weekly)}")

# ── 3. Density tiers & corridor filtering ────────────────────────────
def assign_tier(avg_daily):
    if avg_daily >= 100:
        return "dense"
    elif avg_daily >= 20:
        return "medium"
    return "sparse"

corr_char["tier"] = corr_char["avg_daily_trips"].apply(assign_tier)
total_days = (corr_day["date"].max() - corr_day["date"].min()).days + 1
corr_char["coverage_pct"] = corr_char["n_days"] / total_days * 100
corr_char["avg_weekly_trips"] = corr_char["avg_daily_trips"] * 7

# Keep all dense/medium; sparse only if >80% coverage AND >5 avg weekly trips
sparse = corr_char["tier"] == "sparse"
sparse_pass = sparse & (corr_char["coverage_pct"] > 80) & (corr_char["avg_weekly_trips"] > 5)
keep = ~sparse | sparse_pass
filtered_ids = set(corr_char.loc[keep, "corridor_id"])

print(f"\nCorridor filtering:")
print(f"  Dense:   {(corr_char['tier'] == 'dense').sum()} (all kept)")
print(f"  Medium:  {(corr_char['tier'] == 'medium').sum()} (all kept)")
print(f"  Sparse:  {sparse_pass.sum()} kept / {sparse.sum()} total "
      f"({sparse.sum() - sparse_pass.sum()} dropped)")
print(f"  Final:   {len(filtered_ids)} corridors")

corr_day = corr_day[corr_day["corridor_id"].isin(filtered_ids)].copy()
corr_char_f = corr_char[corr_char["corridor_id"].isin(filtered_ids)].copy()
tier_map = corr_char_f.set_index("corridor_id")["tier"].to_dict()

# ── 4. Aggregate to weekly + complete grid ───────────────────────────
print("\nAggregating to weekly...")
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

# Build complete grid per corridor (first observed week → last observed week)
all_weeks = (
    corr_week[["iso_year", "iso_week", "week_start"]]
    .drop_duplicates().sort_values("week_start")
)

grids = []
for cid, grp in corr_week.groupby("corridor_id"):
    cid_weeks = all_weeks[
        all_weeks["week_start"].between(grp["week_start"].min(), grp["week_start"].max())
    ].copy()
    cid_weeks["corridor_id"] = cid
    cid_weeks["pickup_community_area"] = grp.iloc[0]["pickup_community_area"]
    cid_weeks["dropoff_community_area"] = grp.iloc[0]["dropoff_community_area"]
    grids.append(cid_weeks)

full = pd.concat(grids, ignore_index=True)
full = full.merge(
    corr_week[["corridor_id", "iso_year", "iso_week", "weekly_trips"]],
    on=["corridor_id", "iso_year", "iso_week"], how="left",
)
full["weekly_trips"] = full["weekly_trips"].fillna(0).astype(int)
full["tier"] = full["corridor_id"].map(tier_map)
full = full.sort_values(["corridor_id", "week_start"]).reset_index(drop=True)

print(f"  Grid rows: {len(full):,} "
      f"(filled {(full['weekly_trips'] == 0).sum()} zero-trip weeks)")

# ── 5. Train/test split ─────────────────────────────────────────────
train_mask = (full["iso_year"] < 2025) | (
    (full["iso_year"] == 2025) & (full["iso_week"] <= 26)
)
test_mask = (full["iso_year"] == 2025) & full["iso_week"].between(27, 52)
full["split"] = "unused"
full.loc[train_mask, "split"] = "train"
full.loc[test_mask, "split"] = "test"

# ── 6. Features ──────────────────────────────────────────────────────
print("Computing features...")

# 6a. Events
full = full.merge(hol_weekly, on=["iso_year", "iso_week"], how="left")
full = full.merge(conv_weekly, on=["iso_year", "iso_week"], how="left")
for c in ["holiday_neg_days", "holiday_pos_days", "convention_days", "max_attendees"]:
    full[c] = full[c].fillna(0).astype(int)

# 6b. Airport flags (area 76 = O'Hare)
full["is_airport_origin"] = (full["pickup_community_area"] == 76).astype(int)
full["is_airport_dest"] = (full["dropoff_community_area"] == 76).astype(int)

# 6c. Corridor statics
full = full.merge(
    corr_char_f[["corridor_id", "avg_fare", "avg_miles"]], on="corridor_id", how="left"
)

# 6d. Tier numeric
full["tier_num"] = full["tier"].map({"dense": 2, "medium": 1, "sparse": 0})

# 6e. Temporal
full["week_of_year"] = full["iso_week"]

# 6f. Lags & rolling (per corridor)
for lag in [1, 2, 4]:
    full[f"lag_{lag}"] = full.groupby("corridor_id")["weekly_trips"].shift(lag)

full["rolling_mean_4"] = (
    full.groupby("corridor_id")["weekly_trips"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
)
full["rolling_std_4"] = (
    full.groupby("corridor_id")["weekly_trips"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=2).std())
)
full["rolling_std_4"] = full["rolling_std_4"].fillna(0)

# 6g. Scale (training-period only → no leakage)
train_mean = (
    full.loc[full["split"] == "train"]
    .groupby("corridor_id")["weekly_trips"].mean()
    .reset_index().rename(columns={"weekly_trips": "train_mean_weekly"})
)
full = full.merge(train_mean, on="corridor_id", how="left")
full["log_train_mean"] = np.log1p(full["train_mean_weekly"])

# ── 7. Baseline: per-corridor trailing 8-week mean ──────────────────
print("\n" + "=" * 60)
print("BASELINE: Per-corridor trailing 8-week mean")
print("=" * 60)

full["baseline_pred"] = (
    full.groupby("corridor_id")["weekly_trips"]
    .transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
)

test = full[full["split"] == "test"].copy()
print(f"  Test rows: {len(test):,}  NaN baseline: {test['baseline_pred'].isna().sum()}")

# ── 8. Global XGBoost ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("GLOBAL MODEL: XGBoost (cross-corridor)")
print("=" * 60)

FEATURES = [
    "avg_fare", "avg_miles", "is_airport_origin", "is_airport_dest",
    "tier_num", "log_train_mean", "week_of_year",
    "holiday_neg_days", "holiday_pos_days",
    "convention_days", "max_attendees",
    "lag_1", "lag_2", "lag_4",
    "rolling_mean_4", "rolling_std_4",
]

train_df = full[full["split"] == "train"].dropna(subset=FEATURES)
test_df = test.dropna(subset=FEATURES).copy()

print(f"  Train: {len(train_df):,} rows, {train_df['corridor_id'].nunique()} corridors")
print(f"  Test:  {len(test_df):,} rows, {test_df['corridor_id'].nunique()} corridors")

model = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
)
model.fit(train_df[FEATURES], train_df["weekly_trips"])

test_df["xgb_pred"] = model.predict(test_df[FEATURES]).clip(min=0)

# Merge XGB predictions into test
test = test.merge(
    test_df[["corridor_id", "iso_year", "iso_week", "xgb_pred"]],
    on=["corridor_id", "iso_year", "iso_week"], how="left",
)

# ── 9. Feature importance ───────────────────────────────────────────
print("\nFeature importance (gain):")
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, val in imp.items():
    print(f"  {feat:<22s} {val:.3f}  {'█' * int(val * 50)}")

# ── 10. Evaluation ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION BY DENSITY TIER")
print("=" * 60)


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


rows = []
for tier_name in ["dense", "medium", "sparse", "ALL"]:
    s = test if tier_name == "ALL" else test[test["tier"] == tier_name]
    bm, xm = eval_preds(s, "baseline_pred"), eval_preds(s, "xgb_pred")
    rows.append({
        "tier": tier_name,
        "corridors": s["corridor_id"].nunique(),
        "predictions": bm["n"],
        "baseline_MAE": round(bm["mae"], 1),
        "xgb_MAE": round(xm["mae"], 1),
        "MAE_reduction%": round((bm["mae"] - xm["mae"]) / bm["mae"] * 100, 1) if bm["mae"] else np.nan,
        "baseline_MAPE": round(bm["mape"], 1),
        "xgb_MAPE": round(xm["mape"], 1),
        "baseline_cov30%": round(bm["cov30"], 1),
        "xgb_cov30%": round(xm["cov30"], 1),
    })

res = pd.DataFrame(rows)
print()
print(res.to_string(index=False))

# ── 11. Per-corridor breakdown ──────────────────────────────────────
print("\n" + "=" * 60)
print("PER-CORRIDOR: XGB vs BASELINE")
print("=" * 60)

crows = []
for cid, g in test.groupby("corridor_id"):
    bm, xm = eval_preds(g, "baseline_pred"), eval_preds(g, "xgb_pred")
    crows.append(dict(
        corridor_id=cid, tier=g["tier"].iloc[0],
        avg_weekly=round(g["weekly_trips"].mean(), 1),
        baseline_mae=round(bm["mae"], 1), xgb_mae=round(xm["mae"], 1),
        improvement=round(bm["mae"] - xm["mae"], 1),
    ))
cdf = pd.DataFrame(crows)
cdf["pct_improv"] = np.where(
    cdf["baseline_mae"] > 0,
    (cdf["improvement"] / cdf["baseline_mae"] * 100).round(1), 0
)

for t in ["dense", "medium", "sparse"]:
    sub = cdf[cdf["tier"] == t]
    wins = (sub["improvement"] > 0).sum()
    print(f"\n  {t.upper()} ({len(sub)} corridors):")
    print(f"    XGB wins: {wins}/{len(sub)}")
    print(f"    MAE reduction: median {sub['pct_improv'].median():.1f}%, "
          f"mean {sub['pct_improv'].mean():.1f}%")

# ── 12. Sparse sub-tier detail ──────────────────────────────────────
print("\n" + "=" * 60)
print("SPARSE SUB-TIER DETAIL")
print("=" * 60)
sp = cdf[cdf["tier"] == "sparse"].copy()
sp["sub_tier"] = pd.cut(
    sp["avg_weekly"], bins=[0, 10, 30, 70, np.inf],
    labels=["<10/wk", "10-30/wk", "30-70/wk", "70+/wk"],
)
sub_agg = (
    sp.groupby("sub_tier", observed=True)
    .agg(n=("corridor_id", "count"),
         baseline_mae=("baseline_mae", "mean"),
         xgb_mae=("xgb_mae", "mean"),
         pct_improv=("pct_improv", "mean"))
    .round(1)
)
print(sub_agg.to_string())

# ── 13. Save ─────────────────────────────────────────────────────────
res.to_csv("outputs/phase5_weekly_tier_comparison.csv", index=False)
cdf.to_csv("outputs/phase5_weekly_corridor_detail.csv", index=False)
print("\nSaved: outputs/phase5_weekly_tier_comparison.csv")
print("Saved: outputs/phase5_weekly_corridor_detail.csv")
print("\nDone.")
