# Phases 1-3: Data Collection, Daily Analysis & Feature Validation, Daily Forecast — Conclusions

## Phase 1: Data Collection

### What was collected

| Dataset | Source | Rows | Purpose |
|---------|--------|------|---------|
| area_week.csv | Assignment output | 9,711 | Weekly area-level taxi demand |
| area_day.csv | Raw taxi CSVs (6.9GB) | 67,512 | Daily area-level taxi demand |
| corridor_day.csv | Raw taxi CSVs | 205,665 | Daily O-D corridor demand |
| corridor_characteristics.csv | Raw taxi CSVs | 308 | Corridor features (fare, distance, airport) |
| tnp_area_week.csv | Chicago Data Portal SODA API | 9,394 | Weekly ride-hail by area |
| chicago_weather_weekly.csv | Open-Meteo API | 130 | Weekly weather variables |
| chicago_weather_daily.csv | Open-Meteo API | ~882 | Daily weather variables |
| chicago_sports_games.csv | ESPN public API | 614 | Professional sports schedules |
| holidays_chicago.csv | Calendar rules | ~105 | US/Chicago holidays |
| conventions_chicago.csv | McCormick Place | 18 | Major convention dates + attendees |
| gdelt_chicago_articles.csv | GDELT via BigQuery | 232,145 | Chicago news articles with titles |
| gdelt_chicago_weekly.csv | Aggregated from articles | 128 | Weekly news volume + tone |

All data is public, free, and verifiable. This directly demonstrates an "acquire data, build products" business model.

### Data provenance

| Source | URL | Cost |
|--------|-----|------|
| Chicago Data Portal | data.cityofchicago.org | Free |
| Open-Meteo | open-meteo.com | Free |
| ESPN | site.api.espn.com | Free |
| GDELT | BigQuery (gdelt-bq) | Free tier |

---

## Phase 2: Daily Analysis & Feature Validation

### Core Finding: Day-of-week dominates, most external features are noise

882 NNS daily observations analyzed. 146 anomalous days (>20% deviation) cross-referenced against 5 feature categories.

### What works

| Feature | Coverage of anomalies | Signal | Verdict |
|---------|----------------------|--------|---------|
| Day-of-week | 100% (structural) | 40% swing (Sun -26.6% to Thu +14.4%) | **Dominant feature** |
| Holidays (major US + St Patrick's) | 32% of anomalies | -20% to -75% | **Confirmed** |
| Conventions (McCormick Place) | 11% of anomalies | +12% to +24% | **Confirmed** |

### What doesn't work

| Feature | Signal | Verdict | Evidence |
|---------|--------|---------|----------|
| Daily weather | r ≈ 0 | **Rejected** | Near-zero correlation after DOW control |
| Sports home games | +0.9% avg | **Rejected** | Games occur almost every day — no discriminating signal |
| Day-of-month (payday) | ±10% | **Rejected** | Confounded with DOW |

87% of anomalous days explained by at least one validated feature. 13% remain unexplained — genuine noise or unobservable local events.

### Holiday Impact Table (key deliverable)

Learned from daily M0 baseline (DOW × trend × month seasonality):

| Holiday | Factor | Impact |
|---------|--------|--------|
| Christmas | 0.254 | -74.6% |
| Thanksgiving | 0.390 | -61.0% |
| Day After Christmas | 0.482 | -51.8% |
| July 4th | 0.540 | -46.0% |
| Christmas Eve | 0.628 | -37.2% |
| Memorial Day | 0.637 | -36.3% |
| Black Friday | 0.639 | -36.1% |
| Labor Day | 0.698 | -30.2% |
| St Patrick's Parade | 1.460 | +46.0% |
| St Patrick's Day | 1.294 | +29.4% |

### Convention Impact Table (key deliverable)

| Convention | Factor | Impact |
|------------|--------|--------|
| NRA Show | 1.236 | +23.6% |
| RSNA | 1.179 | +17.9% |
| ProMat | 1.165 | +16.5% |
| PACK EXPO | 1.123 | +12.3% |
| ASCO | 1.121 | +12.1% |

These tables feed directly into Phase 7's insight layer (Layer 1 rule-based explanation) and Phase 5's daily corridor model.

### Weekly Residual Analysis

- W40 2024 was the largest backtest error (16.1%). Initially hypothesized as Rosh Hashanah. Cross-year validation: 2025 Rosh Hashanah showed only -0.5% impact on NNS. **Effect is variable, not a reliable feature.**
- Fixed holidays (Thanksgiving, Christmas, July 4th) fall in the same ISO week both years — not causing forecast errors.
- Trend acceleration confirmed: 2024→2025 taxi growth +12.9%, 2025→2026 taxi growth +21.0% (Phase 6 figures, cross-referenced with ride-hail). Linear trend is too conservative.
- Lag-1 ACF = 0.463 (significant) — errors persist week-to-week.

### Talking point

"I tested every plausible external signal — weather, sports, holidays, conventions, payday effects. Two worked, three didn't. Knowing what NOT to include is as important as what to include. The rejected features save the model from fitting noise. The validated features — holidays and conventions — became the foundation for the insight layer."

---

## Phase 3: Daily Forecast Model

### Core Finding: Daily ML does not beat the weekly baseline for weekly-horizon forecasting

| Version | Approach | Weekly MAPE |
|---------|----------|-------------|
| v1 | XGBoost raw demand, binary flags, lag_7 | 6.10% |
| v2 | XGBoost residual (M1 without month), lag_7 | 6.10% |
| v3 | M0 × month × holiday factors, no lags | 6.95% |
| v3+v1 | XGBoost raw demand, continuous factors, lag_7 | 7.49% |
| Cleaned + AR(1) | Outlier cleaning + residual correction | 4.78%* |
| **Weekly baseline** | **Trend × 52 seasonal factors** | **6.01%** |

*The 4.78% result used retrospective outlier correction (cleaning W40, W41, W44 after observing their errors). This is not generalizable — it requires knowing which weeks are outliers before forecasting them. Abandoned as a model, retained as a diagnostic insight: the 6.01% baseline is driven by a few large errors, not systematic bias.

Four approaches tested. None beat 6.01%. The weekly model's 52 seasonal factors are each a precisely fitted ratio for that specific week-of-year. At daily level, the model must learn this from generic features (month, week_of_year), which is less precise.

### Why this matters (the Phase 3 → Phase 5 connection)

Phase 3's "failure" is actually the setup for Phase 5's success:

| Phase | Data | Training rows | ML beats simple? |
|-------|------|---------------|------------------|
| Phase 3 | Single series (NNS), 882 days | 882 | **No** |
| Phase 5 daily | 183 corridors, daily | 94,000 | **Yes** — 47% MAPE reduction on dense (18.2%→9.7%; MAE 407→271) |

The difference isn't the model — it's the data regime. ML's advantage scales with data volume and cross-series pooling, not model complexity. Phase 3 proves this by contrast: at 882 rows, ML can't beat a well-fitted simple model. At 94,000 rows, ML dramatically outperforms.

### Methodological lesson

After v1 showed 6.10% (matching baseline), three more iterations tried to make it work. The right approach: check if something CAN work before iterating on HOW to make it work. This is a Senior DS judgment call — knowing when to stop and reframe is more valuable than incremental optimization.

### What Phase 3 produced that fed later phases

1. **Holiday/convention factor tables** — directly used in Phase 5 daily corridor model (2.9% feature importance) and Phase 7 insight layer (Layer 1 rule-based explanation)
2. **Rejected features list with evidence** — prevents downstream phases from repeating failed experiments
3. **"Weekly decomposition is near-optimal at this density" finding** — frames the DL benchmark (Phase 4) and the cross-corridor argument (Phase 5)
4. **DOW pattern quantification** — 40% daily swing, confirmed as the dominant signal at daily granularity

### Talking point

"I ran several iterations trying to beat 6% with daily ML. None succeeded. The right conclusion wasn't 'try harder' — it was 'change the data regime.' With 882 rows on one series, simple methods win. With 94,000 rows across 183 corridors, ML reduces MAPE by 47% (18.2%→9.7%; MAE 407→271). Phase 3's failure is what makes Phase 5's success credible — I proved the boundary between where simple methods win and where ML takes over."

---

## Cross-Phase Connection

Phases 1-3 establish the foundation for everything that follows:

| Phase 1-3 Output | Used In |
|-------------------|---------|
| Holiday factor table | Phase 5 daily corridor model, Phase 7 Layer 1 insight |
| Convention factor table | Phase 5 daily corridor model, Phase 7 Layer 1 insight |
| Rejected features (weather, sports) | Phase 5 feature selection (not repeated) |
| "882 rows is not enough" finding | Phase 5 daily (94K rows), Phase 4 narrative |
| DOW = 40% swing | Phase 5 daily model feature |
| W40 unexplained error | Phase 7 "system knows when it doesn't know" demo |
| Trend acceleration diagnosis | Phase 6 confirms: taxi growth is mode-share shift (+12.9% taxi vs +2.1% total) |
| area_day.csv + corridor data | Phase 4 DL benchmark, Phase 5 corridor models |
| GDELT articles | Phase 7 Layer 2 FAISS retrieval |
| TNP ride-hail data | Phase 6 total market view — reveals NNS +30% is mode-share recapture, not real growth |

Nothing from Phases 1-3 is wasted. Every finding, including the rejected hypotheses and failed models, feeds a later phase.
