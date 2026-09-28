# Chicago Demand Intelligence — Project Design & Plan (v2)

## 1. Context

### 1.1 Background
Portfolio project extending a completed Chicago taxi market analysis toward route-level travel-demand forecasting.

### 1.2 Assignment Summary
Analyzed 16M Chicago taxi trips (2024–2026). Recommended Near North Side (area 8) as market entry point: 22.8% of trips, +30% like-for-like growth, no incumbent above 19% share. Forecast: ~35K trips/week for H2 2026. Backtest MAPE: 6.0% (trend × seasonal), vs seasonal naive 14.2%, flat mean 18.0%.

### 1.3 What This Demo Does
Four capabilities that engage with the forecasting and AI cost/latency problems typical of route-level demand products:
1. **Improved forecasting** — diagnose and fix baseline errors, daily-level modeling
2. **O-D corridor forecasting** — global model with cross-corridor learning (the core route-level problem)
3. **AI-powered insight** — automatically explain forecast anomalies
4. **Automated actions** — rule-based recommendations from forecast signals

### 1.4 Repository Structure
```
chicago-demand-intelligence/
├── data/
│   ├── from_assignment/          # Outputs from the baseline analysis
│   │   ├── area_week.csv         # 9,711 rows — area × week aggregated taxi data
│   │   ├── backtest_8.csv        # 27 rows — H2 2025 holdout predictions
│   │   └── forecast_8.csv        # 155 rows — historical + forecast data
│   ├── external/                 # New data this project collects
│   │   ├── chicago_weather_weekly.csv   # Weekly weather (Open-Meteo)
│   │   ├── chicago_weather_daily.csv    # Daily weather (Open-Meteo)
│   │   ├── tnp_area_week.csv            # Ride-hail trips by area × week (SODA API)
│   │   ├── chicago_sports_games.csv     # 614 home games (ESPN API)
│   │   ├── holidays_chicago.csv         # Holidays/events (generated)
│   │   ├── conventions_chicago.csv      # McCormick Place conventions (curated)
│   │   ├── gdelt_chicago_weekly.csv     # News article counts/tone (GDELT API)
│   │   └── gdelt_chicago_articles.csv   # Per-article data for embedding
│   └── derived/
│       ├── area_day.csv          # 67,512 rows — area × day taxi trips
│       └── corridor_day.csv      # TBD — (origin, dest, date) → trips
├── scripts/
│   ├── 0_get_weather.py          # Weekly weather
│   ├── 0_get_weather_daily.py    # Daily weather
│   ├── 0_get_tnp.py              # Ride-hail (month-by-month for 2024)
│   ├── 0_get_gdelt.py            # GDELT news (120s delay, resume capability)
│   ├── 0_get_sports.py           # ESPN sports schedules
│   ├── 0_get_holidays.py         # Holiday calendar generation
│   ├── 0_aggregate_daily.py      # Raw taxi CSVs → area_day.csv
│   ├── 0_aggregate_corridors.py  # TBD — Raw taxi CSVs → corridor_day.csv
│   ├── 1_daily_analysis.py       # Day-of-week, trend, ACF, date impacts
│   ├── 1b_daily_anomalies.py     # All anomalies with known-event tagging
│   └── 1c_cross_reference.py     # Cross-reference anomalies vs features
├── outputs/
├── slides/
├── PROJECT_PLAN.md
├── .gitignore
└── README.md
```

---

## 2. Design Principles

### Priorities
1. **Business thinking over technical skill.** Every module must have a clear "so what."
2. **Data rigor and self-purchased data strategy.** Acquire data, build products. All data must be public, verifiable, clearly sourced.
3. **Communication and explainability.** Deliverable must be presentation-ready.
4. **AI vision — but pragmatic.** "AI is expensive and slow" is the common pain point. Show capability but emphasize cost-efficiency and reliability.
5. **Forecasting is the foundation.** Position improvements as extensions, not corrections.
6. **Spontaneous problem-solving.** The demo itself demonstrates proactive investigation.

### Terminology
Use "route-level" not "O-D pairs" unless they say "O-D" first.

---

## 3. Key Requirements

### Requirement 1: Must improve over 6% MAPE baseline
Core Senior DS value proposition. Approach: diagnose error sources first, then target fixes — not "try different models and hope."

### Requirement 2: Must include DL
DL should be benchmarked explicitly. If DL doesn't beat classical, the finding itself is valuable.

### Requirement 3: Must demonstrate O-D corridor forecasting
This maps directly to travel-demand products — route-level demand prediction with thousands of routes of varying density.

---

## 4. Diagnostic Findings

### 4.1 Weekly Residual Analysis (Completed)

Backtest: train 2024 W02 – 2025 W26, holdout 2025 W27–W52. MAPE: 6.01%.

Error concentration: 50% of weeks have <5% error. 6 weeks (23%) have >10% error. Top 3 worst weeks = 27% of total error.

Root cause investigation:
- **W40 2024 (16.1% error):** City-wide dip. Initially hypothesized as Rosh Hashanah. Cross-year validation: 2025 Rosh Hashanah (W39) showed only -0.5% dip in NNS. **Conclusion: Rosh Hashanah may contribute but is NOT the sole or consistent cause. The causal explanation is weak.**
- **Trend acceleration in Q4:** Model under-predicts by 4.3% in W40-W52 vs -1.6% in W27-W39. YoY growth 2024→2025: +8.2%, 2025→2026: +20.6%.
- **Autocorrelation:** Lag-1 ACF = 0.463 (significant). AR(1) correction is a mechanical win.
- **Single-year H2 seasonal factors:** With only one H2 year (2024), any anomaly dominates the seasonal factor. This is a structural data limitation.

### 4.2 H1 Backtest Validation (Completed)

Tested whether 2-sample seasonal factors outperform 1-sample:
- SF from 2024 only: MAPE 6.87%
- SF from 2025 only: MAPE 7.66%
- SF from both years: MAPE 5.70%

**Conclusion: Averaging two years saves 1.2-2.0pp. More H2 data (e.g., 2023) would improve H2 forecasts by smoothing out single-year anomalies automatically.**

### 4.3 Weekly Model Improvement Results (Completed)

Best result: **6.01% → 4.78% MAPE** (cleaning W40+W41+W44 seasonal factors + AR(1)).

| Config | MAPE | Δ |
|---|---|---|
| Baseline | 6.01% | — |
| Clean W40 + AR(1) | 5.33% | -0.68% |
| Clean W40+W41 + AR(1) | 5.03% | -0.98% |
| Clean W40+W41+W44 + AR(1) | 4.78% | -1.23% |

Limitation: W44 cleaning is less defensible (no confirmed cause). W40/W41 have partial causal support. The improvement is empirically valid but retrospective — it's outlier correction, not a generalizable feature.

### 4.4 Daily Analysis Findings (Completed)

Shifted to daily granularity (882 NNS days) for better pattern detection.

**Day-of-week: 40% swing.** Sunday (-26.6%) to Thursday (+14.4%). This is the dominant feature, far bigger than any holiday or trend effect. Weekday mean: 4,302. Weekend mean: 3,393. Weekend/weekday ratio: 0.789.

**Weekend growth 2x faster than weekday.** 2025→2026: weekday +10.8%, weekend +21.7%. A single trend line is wrong — the model needs separate weekday and weekend trends.

**Autocorrelation dual structure:** Lag-1 (0.745) and lag-7 (0.663) both very strong. XGBoost should use both yesterday and same-day-last-week as lag features.

**Monthly pattern:** Jan lowest (-25%), May-Jun highest (+17%). Strong annual seasonality.

**Month boundary effects:** Day-of-month 18-19 show +10-11% vs average, but confounded with day-of-week. Not a reliable feature.

### 4.5 Feature Cross-Reference Results (Completed)

Tested five feature categories against 146 anomalous days (>20% deviation):

| Feature | Coverage | Avg Impact | Verdict |
|---|---|---|---|
| Holidays (all types) | 32% | -20% to -70% | **CONFIRMED** — consistent across years |
| Conventions (McCormick Place) | 11% | +20% to +44% | **CONFIRMED** — known in advance |
| Sports home games | 61% | +0.9% | **REJECTED** — 5 teams = games almost every day, no signal |
| Daily weather | 32% | near zero correlation | **REJECTED** — no signal at daily level |
| Day-of-week seasonality | — | — | **CONFIRMED** — captures remaining patterns |

87% of anomalous days explained by at least one feature. 19 days (13%) still unexplained, mostly in December holiday season (structural seasonality).

Key holiday impacts (consistent across years):
- Christmas: -70%, Thanksgiving: -63%, New Year's: -58%
- St Patrick's Parade: +52%, St Patrick's Day: +39%
- July 4th: -47%, Memorial Day: -45%, Black Friday: -49%

Key convention impacts:
- Home & Housewares Show: +44%, ASCO: +41%, NRA Show: +28%

---

## 5. Architecture — Modules

### Module 1: Daily Forecast Model + DL Benchmark
**Goal:** Build improved daily forecast and benchmark classical vs DL.

**Feature set for daily model:**
- Day-of-week (7 categories)
- Day-of-year / month (annual seasonality)
- Separate weekday and weekend trends
- Lag features: lag-1, lag-7, lag-14
- Rolling statistics: 7-day mean, 7-day std
- Holiday flags (binary, by type)
- Convention flags (binary, with attendee count)
- Week-of-year (residual seasonality)

**Models to benchmark:**
- Trend × day-of-week seasonal (daily baseline)
- XGBoost with all features
- TimesFM zero-shot (daily, aggregated to weekly for comparison)
- N-BEATS (pure DL)

### Module 2: O-D Corridor Forecasting (NEW — highest product relevance)
**Goal:** Demonstrate global model with cross-corridor learning outperforms per-corridor models on sparse routes.

**Data:** Aggregate raw taxi CSVs to `(pickup_area, dropoff_area, date) → daily_trip_count`.
- Top 4 origins (NNS, Loop, O'Hare, Near West Side) × top 5 destinations each ≈ 20 corridors
- Include reverse corridors (NNS→Loop AND Loop→NNS) to show bidirectional learning

**Two approaches:**
- **Per-corridor:** Each corridor gets independent model. Works for dense corridors, fails on sparse.
- **Global:** All corridors in one XGBoost. Corridor characteristics as features (distance, fare, airport flag, corridor_id). Dense corridors teach sparse corridors.

**Expected finding:** Sparser the corridor, more cross-corridor learning helps.

**Why this matters for route-level forecasting:**
1. 10,000 routes, only 500 have enough data. Global model handles the other 9,500.
2. NNS→O'Hare and O'Hare→NNS share patterns (bidirectional). Same for airline outbound/return.
3. New route (cold start) only needs its features defined. No retraining from scratch.

### Module 3: Total Market View (Taxi + Ride-hail)
**Goal:** Answer the assignment's biggest limitation: "Is NNS +30% growth real, or mode-share shift?"

**Data:** Chicago TNP (ride-hail) — 122 weeks, 2024-2026. NNS ride-hail ~234K trips/week vs taxi ~28K — 8:1 ratio.

**Analysis:**
- Compare NNS taxi trend vs ride-hail trend
- Total mobility demand = taxi + ride-hail
- Test: is total demand more predictable than taxi alone?

### Module 4: AI Insight Layer
**Goal:** When the forecast shows an anomaly, automatically explain why.

**Layer 1 (rule-based, deterministic):**
- Events knowledge base: holidays + conventions from confirmed features
- Forecast identifies anomaly → code matches event by date → deterministic explanation

**Layer 2 (NLP, demonstrating AI capability):**
- GDELT → sentence-transformers → FAISS → semantic retrieval
- Only triggered on anomaly days outside Layer 1 coverage
- Falls back to Layer 1 if coverage thin

### Module 5: Action Engine (Rule-based)
**Goal:** Recommend specific operational actions from forecast signals.

All rules derived from assignment analysis. LLM only formats structured output — no judgment.

### Module 6: News-as-Features Experiment (stretch goal)
**Goal:** Empirical test — does adding news features improve accuracy?

Deprioritized. GDELT data needed, and the feature cross-reference already showed holidays and conventions are the high-value features.

---

## 6. Addressing Common AI Pain Points

### "AI is expensive" → Solved
Provider-agnostic, budget-tier LLM for formatting only.

### "AI is slow" → Solved
All reasoning is deterministic Python functions. LLM called once at the end for text formatting. Total: ~1-2 seconds.

### Orchestration: No LangGraph, No ReAct
Pipeline is linear. Plain Python function chain. Modular tools can later be registered for LLM routing with zero code changes.

---

## 7. Data Collection Status

| Data | Source | Script | Status |
|------|--------|--------|--------|
| Taxi weekly | Assignment | — | ✅ Done (9,711 rows) |
| Taxi daily | Raw CSVs | `0_aggregate_daily.py` | ✅ Done (67,512 rows) |
| Weather weekly | Open-Meteo | `0_get_weather.py` | ✅ Done (130 weeks) |
| Weather daily | Open-Meteo | `0_get_weather_daily.py` | ✅ Done |
| TNP ride-hail | Chicago Data Portal | `0_get_tnp.py` | ✅ Done (122 weeks, 2024-2026) |
| Sports games | ESPN API | `0_get_sports.py` | ✅ Done (614 home games) |
| Holidays | Generated | `0_get_holidays.py` | ✅ Done |
| Conventions | Hand-curated | CSV | ✅ Done (incomplete but usable) |
| GDELT news | GDELT DOC API | `0_get_gdelt.py` | ⏳ Running overnight |
| Corridor daily | Raw CSVs | `0_aggregate_corridors.py` | 📝 Script needed |

### Data Provenance

| Dataset | Source | Access | Cost |
|---|---|---|---|
| Taxi trips | Chicago Data Portal (data.cityofchicago.org) | SODA API + CSV export | Free |
| TNP ride-hail | Chicago Data Portal | SODA API | Free |
| Daily weather | Open-Meteo Archive API (open-meteo.com) | REST API, no key | Free |
| Sports games | ESPN public API (site.api.espn.com) | REST API, no key | Free |
| Holidays | Generated from calendar rules | Code | — |
| Conventions | McCormick Place website, ConventionCalendar.com | Manual research | — |
| News (GDELT) | GDELT Project DOC API (api.gdeltproject.org) | REST API, no key | Free |

---

## 8. Implementation Plan

### Phase 1: Data Collection ✅ (mostly complete)
- [x] All taxi data (weekly, daily)
- [x] Weather (weekly, daily)
- [x] TNP ride-hail
- [x] Sports, holidays, conventions
- [ ] GDELT (overnight)
- [ ] Corridor daily aggregation (script needed)

### Phase 2: Daily Analysis & Feature Validation ✅
### Phase 2: Daily Analysis & Feature Validation ✅
- [x] Day-of-week pattern analysis (40% swing, Sunday -27% to Thursday +14%)
- [x] Trend by day-of-week (weekend growth 2x faster: +21.7% vs +10.8%)
- [x] Autocorrelation structure (lag-1=0.745, lag-7=0.663 — dual structure)
- [x] Monthly seasonality (Jan -25%, May-Jun +17%)
- [x] Anomaly detection — 146 days with >20% deviation catalogued
- [x] Specific date impact testing (holidays verified cross-year, consistent direction)
- [x] Feature cross-reference — 5 categories tested:
    - Holidays (all types): CONFIRMED — consistent -20% to -70%
    - Conventions (McCormick Place): CONFIRMED — +20% to +44%
    - Sports home games: REJECTED — 61% coverage but zero signal
    - Daily weather: REJECTED — near-zero correlation
    - Day-of-month: REJECTED — confounded with day-of-week
- [x] Jewish holiday hypothesis tested — effect exists but variable, not reliably predictable
- [x] Month boundary (payday) effect checked — not significant

### Phase 3: Daily Forecast Model (next)
- [ ] Build daily baseline (trend × day-of-week × month seasonal)
- [ ] Add confirmed features (holidays, conventions)
- [ ] Add lag features (lag-1, lag-7)
- [ ] XGBoost with full feature set
- [ ] Benchmark table (daily MAPE + aggregated-to-weekly MAPE)

### Phase 4: DL Benchmark
- [ ] TimesFM zero-shot on daily data → aggregate to weekly
- [ ] N-BEATS benchmark
- [ ] Combined benchmark table

### Phase 5: O-D Corridor Forecasting
- [ ] Aggregate raw data to corridor_day.csv
- [ ] Per-corridor models (baseline)
- [ ] Global model with corridor features
- [ ] Compare by corridor density tier
- [ ] Bidirectional corridor analysis

### Phase 6: Total Market View
- [ ] NNS taxi vs ride-hail trend comparison
- [ ] Total mobility demand analysis
- [ ] Test: is total demand more predictable?

### Phase 7: AI Insight Layer
- [ ] Layer 1: rule-based event matching (holidays + conventions)
- [ ] Layer 2: GDELT → embedding → FAISS → retrieval (if GDELT data available)
- [ ] Test on anomaly days from holdout

### Phase 8: Action Engine
- [ ] Rule functions
- [ ] LLM formatting layer
- [ ] Cost/latency benchmark

### Phase 9: Integration & Deliverables
- [ ] End-to-end pipeline for example days
- [ ] Slide deck (5-7 pages)
- [ ] Code repo cleanup and README

### Priority Order
1. **Phase 3 (Daily model)** — core DS value prop, MAPE improvement
2. **Phase 5 (O-D corridors)** — highest product relevance
3. **Phase 4 (DL benchmark)** — required by job spec
4. **Phase 6 (Total market)** — answers assignment limitation
5. **Phase 7 (AI insight)** — demonstrates AI capability
6. **Phase 8 (Action engine)** — decision-engine vision
7. **Phase 9 (Deliverables)** — everything else done first

### Dependencies
- Phase 3 can start immediately (all data available)
- Phase 5 needs corridor_day.csv (one more aggregation script)
- Phase 4 needs area_day.csv (done) + TimesFM/N-BEATS libraries
- Phase 7 needs GDELT (running overnight)
- Phase 9 needs all others done

---

## 9. Deliverable Format

**NOT a live demo.** A concise slide deck + working code in repo:

- Architecture diagram showing the full pipeline
- Daily model: diagnosis → feature validation → improvement (the "Senior DS story")
- O-D corridor: global vs per-corridor benchmark by density tier
- DL benchmark table
- Total market view chart (taxi vs ride-hail)
- AI insight: 3 real examples with actual output
- "Applied to travel routes" slide: area-level → route-level with cross-route learning

### Narrative
"After the assignment, I explored four extensions. First, I shifted to daily forecasting — diagnosed the error structure, tested five feature categories, confirmed two (holidays and conventions), and rejected three with evidence (sports, weather, day-of-month). Second, I built route-level forecasting showing that a global model with cross-route learning outperforms per-route models on sparse routes — directly applicable to route-level travel products. Third, I cross-referenced ride-hail data to determine if the observed growth is real. Fourth, I built an insight pipeline that automatically explains forecast anomalies."

---

## 10. Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| DL doesn't beat improved classical | Expected finding. "Classical is sufficient at this density; DL value comes with cross-series training." |
| O-D corridor global model doesn't help on sparse routes | Unlikely with XGBoost, but even a null result is informative. |
| Ride-hail shows mode-shift not real growth | MORE valuable. "Assignment's biggest limitation, now addressed." |
| GDELT coverage thin | Layer 1 (rule-based) is primary. Layer 2 is architecture demo. |
| Time pressure | Priority order ensures highest-value items done first. |

---

## 11. Technical Stack

- **Python** (primary)
- **XGBoost** (daily model, corridor model, news experiment)
- **PyTorch** (N-BEATS)
- **TimesFM** (Google foundation model)
- **SHAP** (feature importance)
- **sentence-transformers** (embedding, for Layer 2)
- **FAISS** (vector search, for Layer 2)
- **LLM API** (formatting only — provider-agnostic, budget tier)

---

## 12. Working Guidelines

### Methodology
1. **Diagnose before building.** Examine errors first, then hypothesize causes, then verify cross-year.
2. **Every hypothesis must be validated.** If a pattern appears in 2024, it must repeat in 2025 to be confirmed.
3. **Collect all data before building models.** No partial builds.
4. **Features must earn their place.** Test each feature category; reject those without signal.
5. **Daily > weekly for pattern detection.** Events are daily impacts; weekly aggregation dilutes them.

### Presentation Principles
- Frame findings as value regardless of outcome
- Respect the existing baseline while extending it
- Use "route-level" not "O-D pairs"
- Show when NOT to add complexity (rejected features, no LangGraph)

### Lessons Learned
- Don't refuse to write partial data — always save what succeeded
- Set API rate limits conservatively from the start
- Don't push forward without complete data
- Verify cross-year before claiming causal relationships
- Sports games appear correlated (61% coverage) but have zero signal — always check causation, not just coverage
