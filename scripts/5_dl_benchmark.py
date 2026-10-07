"""
Phase 4 — DL Forecast Benchmark
=================================
Benchmarks DL methods against the 6.01% weekly MAPE baseline.
All models predict daily, aggregated to weekly for comparison.

Models:
  1. Naive lag-7 (same weekday last week) — daily reference
  2. TimesFM zero-shot (no training, rolling 7-day)
  3. N-BEATS trained on NNS (rolling 7-day via cross_validation)

Prerequisites:
    pip install timesfm neuralforecast

Run from project root:
    python scripts/5_dl_benchmark.py
"""

import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")
Path("outputs").mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE
# ══════════════════════════════════════════════════════════════════════
print("Loading NNS daily data...")
df = pd.read_csv("data/derived/area_day.csv")
nns = df[df["pickup_community_area"] == 8].sort_values("date").copy()
nns["date"] = pd.to_datetime(nns["date"])

# Keep only through end of test period (2025 W52 = 2025-12-28)
nns = nns[nns["date"] <= "2025-12-28"].reset_index(drop=True)

train_mask = (nns["iso_year"] < 2025) | (
    (nns["iso_year"] == 2025) & (nns["iso_week"] <= 26)
)
test_mask = (nns["iso_year"] == 2025) & nns["iso_week"].between(27, 52)
train = nns[train_mask].copy()
test = nns[test_mask].copy()

all_trips = nns["trip_count"].values.astype(float)
train_trips = train["trip_count"].values.astype(float)

print(f"  Train: {len(train)} days | Test: {len(test)} days")
print(f"  Train period: {train['date'].min().date()} to {train['date'].max().date()}")
print(f"  Test period:  {test['date'].min().date()} to {test['date'].max().date()}")

# Actual weekly totals (ground truth for MAPE)
actual_weekly = (
    test.groupby(["iso_year", "iso_week"])["trip_count"]
    .sum().reset_index()
    .rename(columns={"trip_count": "actual"})
    .sort_values(["iso_year", "iso_week"])
)

# Store all daily predictions here, keyed by model name
daily_preds = {}

# ══════════════════════════════════════════════════════════════════════
# 2. NAIVE LAG-7 BASELINE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 1: Naive lag-7 (same weekday last week)")
print("=" * 60)

# For each test day, predict = actual from 7 days ago
lag7_preds = []
for idx in test.index:
    lag_idx = idx - 7  # 7 rows back (data is sorted daily)
    if lag_idx >= 0:
        lag7_preds.append(nns.loc[lag_idx, "trip_count"])
    else:
        lag7_preds.append(np.nan)

daily_preds["naive_lag7"] = np.array(lag7_preds, dtype=float)
print("  Done.")

# ══════════════════════════════════════════════════════════════════════
# 3. TIMESFM ZERO-SHOT (rolling 7-day)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2: TimesFM zero-shot (rolling 7-day)")
print("=" * 60)

try:
    import timesfm

    t0 = time.time()
    print("  Loading TimesFM checkpoint...")

    # TimesFM 2.5 API
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False
    )
    tfm.compile(timesfm.ForecastConfig(max_horizon=128, max_context=512))

    print(f"  Checkpoint loaded in {time.time() - t0:.1f}s")
    print("  Running rolling 7-day predictions (26 weeks)...")

    tfm_preds = []
    n_train = len(train)

    for week_idx in range(26):
        # Context: all data up to current week
        ctx_end = n_train + week_idx * 7
        context = all_trips[:ctx_end]

        # Forecast next 7 days
        point_forecast, quantiles = tfm.forecast(horizon=7, inputs=[context])
        week_pred = point_forecast[0][:7]  # Take first 7 days
        tfm_preds.extend(week_pred.tolist())

        if (week_idx + 1) % 10 == 0:
            print(f"    Week {week_idx + 1}/26 done")

    daily_preds["timesfm"] = np.array(tfm_preds).clip(min=0)
    print(f"  TimesFM done in {time.time() - t0:.1f}s total")

except ImportError:
    print("  TimesFM not installed. pip install timesfm")
    print("  Skipping.")
except Exception as e:
    print(f"  TimesFM error: {e}")
    print("  Skipping.")

# ══════════════════════════════════════════════════════════════════════
# 4. N-BEATS (trained, rolling via cross_validation)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 3: N-BEATS (trained on NNS, rolling 7-day)")
print("=" * 60)

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS

    t0 = time.time()

    # NeuralForecast expects: unique_id, ds, y
    nf_df = pd.DataFrame({
        "unique_id": "NNS",
        "ds": nns["date"].values,
        "y": nns["trip_count"].values.astype(float),
    })

    model = NBEATS(
        h=7,
        input_size=28,  # 4 weeks lookback
        max_steps=500,
        scaler_type="standard",
        random_seed=42,
    )
    nf = NeuralForecast(models=[model], freq="D")

    print("  Running cross_validation (trains once, predicts 26 weeks)...")
    cv_result = nf.cross_validation(
        df=nf_df,
        step_size=7,
        n_windows=26,
    )

    # cv_result has columns: unique_id, ds, cutoff, y, NBEATS
    cv_result = cv_result.sort_values("ds").reset_index(drop=True)
    daily_preds["nbeats"] = cv_result["NBEATS"].values.clip(min=0)

    print(f"  N-BEATS done in {time.time() - t0:.1f}s")
    print(f"  Predictions: {len(cv_result)} days")

except ImportError:
    print("  NeuralForecast not installed. pip install neuralforecast")
    print("  Skipping.")
except Exception as e:
    print(f"  N-BEATS error: {e}")
    print("  Skipping.")

# ══════════════════════════════════════════════════════════════════════
# 5. AGGREGATE TO WEEKLY + MAPE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("AGGREGATE: Daily predictions → Weekly MAPE")
print("=" * 60)

test_weeks = test[["iso_year", "iso_week"]].values
results = []

# Weekly baseline (from the baseline analysis)
results.append({
    "model": "Trend × seasonal (weekly baseline)",
    "granularity": "Weekly",
    "weekly_mape": 6.01,
    "source": "Baseline",
})

results.append({
    "model": "M1 (DOW×month×holiday factors)",
    "granularity": "Daily→Weekly",
    "weekly_mape": 6.95,
    "source": "Phase 3",
})

for model_name, preds in daily_preds.items():
    if len(preds) != len(test):
        print(f"  WARNING: {model_name} has {len(preds)} predictions, expected {len(test)}")
        continue

    # Add predictions to test dataframe
    pred_df = test[["iso_year", "iso_week"]].copy()
    pred_df["pred"] = preds
    pred_df["actual"] = test["trip_count"].values

    # Aggregate to weekly
    weekly = pred_df.groupby(["iso_year", "iso_week"]).agg(
        pred_weekly=("pred", "sum"),
        actual_weekly=("actual", "sum"),
    ).reset_index()

    # MAPE
    mape = (
        np.abs(weekly["actual_weekly"] - weekly["pred_weekly"])
        / weekly["actual_weekly"]
    ).mean() * 100

    # MAE
    mae = np.abs(weekly["actual_weekly"] - weekly["pred_weekly"]).mean()

    results.append({
        "model": model_name,
        "granularity": "Daily→Weekly",
        "weekly_mape": round(mape, 2),
        "weekly_mae": round(mae, 0),
        "source": "This experiment",
    })

    # Also compute daily MAPE for reference
    valid = ~np.isnan(preds) & (test["trip_count"].values > 0)
    daily_mape = (
        np.abs(test["trip_count"].values[valid] - preds[valid])
        / test["trip_count"].values[valid]
    ).mean() * 100

    print(f"  {model_name}: weekly MAPE = {mape:.2f}%, daily MAPE = {daily_mape:.2f}%")

# ══════════════════════════════════════════════════════════════════════
# 6. COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BENCHMARK COMPARISON TABLE")
print("=" * 60)

res_df = pd.DataFrame(results)
print()
print(f"  {'Model':<45s} {'Granularity':<15s} {'Weekly MAPE':>12}")
print(f"  {'-' * 74}")
for _, row in res_df.iterrows():
    print(f"  {row['model']:<45s} {row['granularity']:<15s} {row['weekly_mape']:>11.2f}%")

print(f"""
Notes:
  - All models evaluated on same holdout: 2025 W27-W52 (26 weeks)
  - Daily predictions aggregated to weekly by summing within ISO week
  - TimesFM: zero-shot (no training on NNS data), rolling 7-day horizon
  - N-BEATS: trained on NNS daily data only, rolling 7-day via cross_validation
  - Naive lag-7: predict each day as same weekday last week
  - Weekly baseline: multiplicative trend × 52 seasonal factors (from the baseline analysis)
""")

# ── Save ─────────────────────────────────────────────────────────────
res_df.to_csv("outputs/phase4_dl_benchmark.csv", index=False)
print("Saved: outputs/phase4_dl_benchmark.csv")
print("\nDone.")
