# Chicago Demand Intelligence — Project Summary

## What This Is

A portfolio demo project on demand forecasting for route-level travel products. Extends a completed analysis (Chicago taxi market analysis, 16M trips, 2024-2026) to engage with the forecasting and AI cost/latency problems common in prediction products: cross-corridor forecasting, total market analysis, AI-powered insight, and automated action recommendations.

## The Two-Part Story

### Part 1: Where does ML actually help?

Six phases of analysis converged on one finding: **ML's value depends on data density, not model sophistication.**

| Data regime | Training rows | Best method | MAPE |
|-------------|---------------|-------------|------|
| Single dense series (NNS) | 882 days | Classical trend × seasonal | 6.01% |
| Single dense series (NNS) | 882 days | XGBoost with 20+ features | 6.10% |
| Single dense series (NNS) | 882 days | TimesFM zero-shot | 8.20% |
| Cross-corridor (183 routes) | 94,000 days | Global XGBoost | 7.10% (NNS-equivalent) |
| Cross-corridor daily | 94,000 days | Global XGBoost | 47% MAPE reduction on dense corridors (18.2%→9.7%; MAE 407→271) |

On a single series, simple methods win. At cross-corridor scale — which is the regime of route-level travel products — ML dramatically outperforms. The difference is not the algorithm, it's the data volume. Phase 3's "failure" (ML can't beat 6%) and Phase 5's success (47% MAPE reduction on dense corridors) tell the same story from opposite sides.

### Part 2: How does prediction become revenue?

An AI pipeline converts forecasts into dollar-valued action recommendations for a 150-car taxi company entering Near North Side.

Six action types, each data-validated and revenue-connected:

| Action Type | Example | Revenue Impact |
|-------------|---------|---------------|
| DEMAND_DROP | Christmas: park 112 cars | $22,400/day saved |
| RESOURCE_OPTIMIZE | Thanksgiving: redirect 28 idle cars to airport $46 fare | $16,664/day benefit |
| DEMAND_SURGE | RSNA convention: system rejects adding cars (math shows net loss) | $6,577 loss avoided |
| COMPETITIVE_SHIFT | Ride-hail surge hours: capture 100 extra trips/week | $72,332/year |
| GROWTH_THRESHOLD | NNS→Loop corridor sustained demand | $208,000/year potential |
| ANOMALY_ESCALATION | 16.1% error, no explanation: escalate, don't guess | Trust preserved |

---

## Phase-by-Phase Findings

### Phases 1-3: Data Collection, Feature Validation, Daily Forecast

**Data:** 16M taxi trips, 9.4K ride-hail weekly records, 232K news articles, 614 sports games, 105 holidays, 18 conventions. All public, all free, all verifiable.

**Feature validation:** Holidays confirmed (-20% to -75%). Conventions confirmed (+12% to +24%). Weather rejected (r≈0). Sports rejected (zero signal). Day-of-month rejected (confounded with DOW).

**Daily forecast:** Four XGBoost approaches tested. None beat the 6.01% weekly baseline. Conclusion: at 882 rows on a single series, simple methods are near-optimal. This finding defines the boundary between where simple methods win and where ML takes over.

**Key deliverables:** Holiday/convention factor tables (fed Phase 5 daily model and Phase 7 insight layer). Rejected features list (prevented downstream phases from repeating failed experiments).

### Phase 4: DL Benchmark

| Model | Type | Weekly MAPE |
|-------|------|-------------|
| Trend × seasonal | Classical, fitted | 6.01% |
| M1 (DOW × month × holiday) | Feature-engineered, daily | 6.95% |
| TimesFM 2.5 zero-shot | Foundation model, no training | 8.20% |
| N-BEATS trained | DL, trained on NNS | 10.23% |

Neither DL model beat classical. But TimesFM zero-shot outperformed trained N-BEATS — a foundation model's pre-trained knowledge is more valuable than 546 days of per-series training. This directly informs the cold-start architecture.

### Phase 5: O-D Corridor Forecasting

**Four experiments, four findings:**

**1. Global model beats per-corridor model across all tiers.** Weekly: ~15% MAE reduction. Daily: 47% MAPE reduction on dense corridors (18.2%→9.7%; MAE 407→271). The mechanism is shared parameter estimation — 94,000 rows across 183 corridors produce more robust splitting rules than 74 rows from one corridor.

**2. Per-corridor XGBoost overfits on sparse routes.** MAE 9.8 vs trailing mean 8.1. With ~74 training rows and 16 features, the model fits noise. Global pooling is not optional — it is necessary.

**3. Cold-start converges in 4 weeks.** 20 corridors held out entirely from training. Cold-start predictions match full-history predictions from week 5 onward. Medium/sparse corridors show exact convergence.

**4. Global daily XGBoost (7.10%) beats TimesFM (8.20%).** Cross-corridor learning from local data outperforms foundation model pre-training. Holiday features contribute 2.9% importance at daily level (vs ~0.8% at weekly). The daily data regime (94K rows) is where ML earns its advantage.

**Production architecture:**

| Route stage | Weeks | Method |
|-------------|-------|--------|
| Cold start | 1-4 | TimesFM zero-shot (~8% MAPE) |
| Ramp-up complete | 5+ | Global XGBoost (~7% MAPE) |
| Established dense | Full history | Classical (~6% MAPE) |

No route ever goes without a forecast.

### Phase 6: Total Market View

**Core finding: NNS +30% taxi growth is mode-share shift, not real demand growth.**

| Period | Taxi | Ride-hail | Total | Taxi Share |
|--------|------|-----------|-------|------------|
| 2024→2025 | +12.9% | +0.9% | +2.1% | 9.9% → 10.9% |
| 2025→2026 | +21.0% | -4.7% | -2.2% | 9.5% → 11.8% |

Total demand is flat to declining. Taxi is recapturing share from ride-hail (8.4:1 ratio). O'Hare is the mirror image: taxi -5.2% but total +6.2%.

Forecasting total demand (MAPE 3.9%) is more accurate than taxi-only (6.3%). Product implication: acquire ride-hail data for total mobility demand modeling.

### Phases 7-8: AI Insight Layer + Action Engine

**Architecture:** All reasoning is deterministic (Python functions, FAISS search, rule-based actions). LLM called only for routing (which tools?) and formatting. No LLM judgment, no hallucination risk.

**Five tools:** get_forecast, search_events, search_news (232K articles in FAISS), suggest_action, get_market_comparison. Same tools work in two modes: deterministic pipeline (<0.1s, $0) and ReAct agent (2-10s, ~$0.001).

**FAISS pipeline:** 232K Chicago news articles from 10 local outlets (GDELT via BigQuery). Embedded with all-MiniLM-L6-v2 (384D). Semantic retrieval fills gaps where rule-based lookup has no coverage.

**Action engine highlights:**
- DEMAND_SURGE: system calculates adding cars would lose money at $13.91 avg fare, recommends maximizing existing fleet instead. The math decides, not the model.
- ANOMALY_ESCALATION: 16.1% forecast error with no validated cause. System escalates instead of fabricating an explanation. Trust over automation.
- Action config externalized to JSON. Same engine, different company profile per vertical.

**Cost/latency benchmark:**

| Approach | Latency | Cost |
|----------|---------|------|
| Full-LLM reasoning (est.) | 10-30s | $0.05-0.10 |
| Our ReAct (GPT-4o-mini + tools) | 2-10s | ~$0.001 |
| Our deterministic pipeline | <0.1s | $0 (no API) |

*Baseline estimated; ReAct and deterministic rows measured.*

---

## Connection to Route-Level Travel Forecasting

| Demo Component | Product Mapping |
|---------------|----------------------|
| Cross-corridor global model | Pool route data; don't model each independently |
| 94K rows is where ML wins | Travel products have thousands of routes × daily — same regime |
| Cold-start 4-week convergence | New routes forecastable within one month |
| TimesFM for day-1 forecasts | Foundation models solve cold-start without training |
| Total market (taxi + ride-hail) | Buy data, combine sources — data-acquisition business model |
| Holiday/convention factors | Known-future inputs — what TFT is designed for |
| Deterministic insight (<0.1s) | "AI is slow" pain point → solved |
| $0.001 per ReAct query | "AI is expensive" pain point → solved |
| Structured action packages | Decision-engine vision: clients take actions on platform |
| FAISS + sentence-transformers | Embeddings, vector DBs, retrieval |
| Action config as JSON | Configurable per industry vertical |

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Core ML | XGBoost, scikit-learn |
| Deep learning | TimesFM 2.5, N-BEATS (NeuralForecast) |
| NLP | sentence-transformers (all-MiniLM-L6-v2), FAISS |
| News data | GDELT via Google BigQuery |
| LLM routing | GPT-4o-mini (OpenAI tool calling) |
| Data sources | Chicago Data Portal (SODA API), Open-Meteo, ESPN, McCormick Place |
| Demo | Streamlit, Plotly |
| All data | Public, free, verifiable |

---

## Key Narratives

**On forecasting:** "For a single dense series, classical is unbeatable at 6%. But at route-network scale — thousands of routes, daily data — ML reduces MAPE by 47% (18.2%→9.7%; MAE 407→271). Phase 3 proves the boundary. Phase 5 proves the opportunity."

**On DL:** "TimesFM zero-shot beats trained N-BEATS without seeing any data. Foundation models solve cold-start. Combined with the 4-week ramp-up finding, no route ever goes without a forecast."

**On total market:** "The +30% taxi growth is mode-share shift, not real demand. Total demand is flat. O'Hare is the mirror. This is the kind of insight you only get by cross-referencing multiple data sources."

**On AI pipeline:** "Most agentic systems are expensive because they use LLMs for reasoning, and slow because each step waits for a response. We invert this: all reasoning is deterministic, LLM only routes and formats. 50-100x cheaper. The deterministic mode runs in under 100 milliseconds with zero API cost — two orders of magnitude faster than full-LLM approaches."

**On actions:** "The system doesn't just explain — it recommends. And it checks the math: when adding cars loses money, it says so. When it can't explain an anomaly, it escalates instead of guessing. Every recommendation has a dollar amount and an evidence trail."

**On methodology:** "I tested every plausible feature and rejected three. I ran four daily model variants and concluded they can't beat weekly at this density. Knowing what doesn't work — and stopping — is part of the result."

---

## Repository Structure

```
chicago-demand-intelligence/
├── app.py                          # Streamlit demo (single file)
├── config/
│   └── action_config.json          # Action types + company profile
├── data/
│   ├── baseline/                   # Baseline weekly analysis outputs
│   │   ├── area_week.csv
│   │   ├── backtest_8.csv
│   │   └── forecast_8.csv
│   ├── external/                   # Collected data
│   │   ├── chicago_weather_weekly.csv
│   │   ├── chicago_weather_daily.csv
│   │   ├── tnp_area_week.csv
│   │   ├── chicago_sports_games.csv
│   │   ├── holidays_chicago.csv
│   │   ├── conventions_chicago.csv
│   │   ├── gdelt_chicago_articles.csv
│   │   └── gdelt_chicago_weekly.csv
│   └── derived/                    # Aggregated + model artifacts
│       ├── area_day.csv
│       ├── corridor_day.csv
│       ├── corridor_characteristics.csv
│       ├── gdelt_faiss.index
│       └── gdelt_articles_meta.pkl
├── scripts/
│   ├── 0_get_weather.py            # Data collection
│   ├── 0_get_weather_daily.py
│   ├── 0_get_tnp.py
│   ├── 0_get_gdelt_bq.py
│   ├── 0_get_sports.py
│   ├── 0_get_holidays.py
│   ├── 0_aggregate_daily.py
│   ├── 0_aggregate_corridors.py
│   ├── 1_daily_analysis.py         # Analysis
│   ├── 1b_daily_anomalies.py
│   ├── 1c_cross_reference.py
│   ├── 1d_weekly_residual_analysis.py
│   ├── 2_daily_forecast.py         # Phase 3 daily models
│   ├── 3_corridor_eda.py           # Phase 5 corridor
│   ├── 3b_corridor_forecast.py
│   ├── 3c_corridor_ablation.py
│   ├── 3d_cold_start_rampup.py
│   ├── 3e_corridor_daily.py
│   ├── 4_total_market.py           # Phase 6
│   ├── 5_dl_benchmark.py           # Phase 4 DL
│   ├── 6_build_faiss.py            # Phase 7 FAISS
│   └── 7_ai_pipeline.py            # Phase 7+8 AI pipeline
├── outputs/
│   ├── phase1_3_conclusions.md
│   ├── phase4_conclusions.md
│   ├── phase5_conclusions.md
│   ├── phase6_conclusions.md
│   ├── phase7_8_conclusions.md
│   ├── phase7_mode1_results.json
│   ├── phase4_dl_benchmark.csv
│   ├── phase5_weekly_tier_comparison.csv
│   ├── phase5_weekly_corridor_detail.csv
│   ├── phase5_daily_corridor_detail.csv
│   ├── phase5_rampup_detail.csv
│   ├── phase5_ablation_corridor_detail.csv
│   └── phase6_nns_total_market.csv
└── README.md
```

---

## What Was Not Built (and Why)

| Item | Reason |
|------|--------|
| News-as-features experiment | Phase 3 proved features don't help dense single series. Same conclusion expected. |
| Weather/sports context signals | Rejected in Phase 2. Can be surfaced manually if relevant. |
| LSTM | 124 weeks on one series — insufficient to beat classical. Dropped early. |
| TFT (Temporal Fusion Transformer) | Insufficient data for multi-horizon training. With thousands of routes, TFT is the right next step. |
| LangGraph / multi-agent orchestration | Pipeline is linear. Adding frameworks where functions suffice signals over-engineering. |
| Live demo execution | Too slow/fragile to run end-to-end interactively. Pre-computed results in Streamlit. ReAct is live but scoped. |
| Mode 2 production (LLM routing to tools) | Architecture demonstrated in ReAct. "Same tools, different orchestrator." Not built as standalone product. |
