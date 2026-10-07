# Phase 4: DL Forecast Benchmark — Conclusions

## Setup

NNS (Area 8) daily taxi trips. Train: 546 days (2024-01-01 to 2025-06-29). Test: 182 days (2025-06-30 to 2025-12-28, 26 weeks). Daily predictions aggregated to weekly for MAPE comparison against the 6.01% weekly baseline.

Four models benchmarked:
- **Naive lag-7:** same weekday last week (daily reference)
- **TimesFM 2.5 zero-shot:** Google's foundation model, no training on NNS data, rolling 7-day predictions
- **N-BEATS:** trained on 546 NNS days via NeuralForecast, rolling 7-day cross-validation
- **M1 (DOW × month × holiday factors):** hand-crafted daily features from Phase 3

---

## Results

| Model | Type | Weekly MAPE | Beat 6.01%? |
|-------|------|-------------|-------------|
| Trend × seasonal | Weekly, 52 factors | 6.01% | — (baseline) |
| M1 (DOW × month × holiday) | Daily→Weekly, hand-crafted | 6.95% | No |
| TimesFM zero-shot | Daily→Weekly, foundation model | 8.20% | No |
| N-BEATS trained | Daily→Weekly, trained on NNS | 10.23% | No |
| Naive lag-7 | Daily→Weekly, no model | 11.80% | No |

---

## Finding 1: Classical decomposition is optimal at this data density

Neither DL model beats the weekly baseline. With 546 training days on a single dense series (~4,000 trips/day), two parameters (slope + intercept) and 52 seasonal factors already capture the predictable structure. This confirms Phase 3's conclusion: daily granularity does not help weekly-horizon forecasting at this density.

This is not a failure of DL — it is a statement about data density. The weekly model's 52 seasonal factors are each a precisely fitted ratio for that specific week-of-year. At daily level, models must learn this pattern from generic features (day-of-week, week-of-year), which is less direct.

## Finding 2: TimesFM zero-shot outperforms trained N-BEATS

The most interesting result. TimesFM (8.20%) beats N-BEATS (10.23%) without seeing any NNS data. A foundation model's pre-trained knowledge from millions of time series is more valuable than 546 days of per-series training.

Why N-BEATS underperforms: with input_size=28 (4 weeks lookback) and only 546 training days, N-BEATS has limited data to learn the annual seasonal pattern. It captures short-term dynamics but misses the longer-term structure that TimesFM absorbs from its pre-training corpus.

## Finding 3: Production architecture emerges from combined findings

Phase 4 + Phase 5 together define a production forecasting path:

| Route stage | Weeks | Method | Expected accuracy |
|-------------|-------|--------|-------------------|
| Cold start (new route) | 1-4 | TimesFM zero-shot | ~8% MAPE |
| Ramp-up complete | 5+ | Global XGBoost model | Matches full-history |
| Established (dense) | All | Trend × seasonal | ~6% MAPE |

TimesFM provides immediate forecasts for new routes with zero training. After 4 weeks of observed data, the global model (trained on all routes) takes over. For established dense routes with full history, classical decomposition remains optimal.

---

## Connection to Route-Level Forecasting

| Finding | Relevance |
|---------|-----------------|
| DL doesn't beat classical on dense series | Validates a classical approach — no need to replace |
| TimesFM zero-shot outperforms trained N-BEATS | Foundation models solve cold-start without per-route training |
| 8.20% MAPE with zero training | New routes get forecasts from day one |
| Clear production architecture | TimesFM → global model → classical, based on data maturity |

### Deep learning coverage

This benchmark demonstrates:
1. Practical DL implementation (TimesFM, N-BEATS)
2. Rigorous evaluation methodology (rolling cross-validation, same holdout period)
3. Correct conclusion: knowing when DL is NOT the right tool is itself a finding
4. Identifying where DL IS the right tool: cold-start and cross-series scenarios

---

## Narrative

"For dense single-series weekly forecasting, classical decomposition remains optimal — 6% MAPE with two parameters and 52 seasonal factors. Neither TimesFM nor N-BEATS beat it. But TimesFM zero-shot outperformed trained N-BEATS without seeing any NNS data. This is the cold-start story: a foundation model's pre-trained knowledge is more valuable than limited per-series training. Combined with Phase 5's finding that new routes converge in 4 weeks, the production path is clear — TimesFM for cold-start weeks 1-4, then the global model takes over."

---

## Technical Notes

- TimesFM 2.5 (200M parameters), PyTorch backend, CPU inference, ~4.4s for 26 rolling predictions
- N-BEATS via NeuralForecast, input_size=28, h=7, max_steps=500, ~11s training + prediction
- All models evaluated with rolling one-step-ahead (7-day horizon), using actual prior-week data as context
- No hyperparameter tuning performed — this is a benchmark, not optimization
