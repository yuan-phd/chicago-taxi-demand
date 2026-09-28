# Phase 6: Total Market View — Conclusions

## Core Question

Is NNS's +30% taxi growth (2024→2026) real demand growth, or mode-share shift from ride-hail?

**Answer: Mode-share shift.** Taxi is recapturing share from ride-hail. Total mobility demand is roughly flat.

---

## Evidence

### NNS (Area 8) Growth Decomposition

| Period | Taxi | Ride-hail | Total Demand | Taxi Share |
|--------|------|-----------|-------------|------------|
| 2024→2025 (52 weeks) | +12.9% | +0.9% | +2.1% | 9.9% → 10.9% |
| 2025→2026 (18 weeks) | +21.0% | -4.7% | -2.2% | 9.5% → 11.8% |

Total demand grew marginally in year 1 and declined in year 2. Taxi growth is entirely explained by mode-share recapture (9.8% → 11.8% over 2.5 years). Ride-hail dominates NNS at ~236K trips/week vs taxi ~28K — an 8.4:1 ratio. Even small mode-share shifts produce large percentage swings in taxi volumes.

### Reconciliation with Assignment Figures

The assignment reported +30.2% like-for-like growth comparing 2024 vs 2026 using W02-W21 (20 weeks). Phase 6 shows +12.9% (2024→2025) and +21.0% (2025→2026), cumulating to approximately +36.6% over two years. The difference (30.2% vs 36.6%) is explained by different week samples. Both are correct. Both are taxi-only.

### Mode-Share Trend

| Half-Year | Taxi Share | Taxi/wk | Ride-hail/wk |
|-----------|-----------|---------|-------------|
| 2024 H1 | 9.8% | 25,128 | 231,887 |
| 2024 H2 | 10.0% | 27,004 | 243,383 |
| 2025 H1 | 10.4% | 27,291 | 235,221 |
| 2025 H2 | 11.4% | 31,560 | 244,540 |
| 2026 H1 | 11.8% | 29,468 | 220,342 |

Taxi share is rising steadily. Ride-hail is flat or declining. The taxi growth story is a share story, not a volume story.

### O'Hare Reversal

The assignment noted O'Hare taxi declining (-5.2% YoY). Phase 6 shows O'Hare total demand grew +6.2%, with ride-hail up +10.0%. O'Hare is not a declining market — taxis are losing share to ride-hail. This is the mirror image of NNS.

| Area | Taxi | Ride-hail | Total | Share Δ |
|------|------|-----------|-------|---------|
| Near North Side | +12.9% | +0.9% | +2.1% | +1.0pp |
| O'Hare | -5.2% | +10.0% | +6.2% | -2.7pp |
| Loop | +6.9% | +1.1% | +1.8% | +0.7pp |
| Near West Side | +14.9% | +2.8% | +3.8% | +0.8pp |
| Near South Side | +10.9% | +3.4% | +4.3% | +0.7pp |

Pattern: taxis are gaining share in urban areas (NNS, Loop, Near West/South Side) and losing share at the airport. This suggests a structural shift in competitive dynamics, not random fluctuation.

### Forecasting Test (NNS)

| Model | MAPE | Note |
|-------|------|------|
| Taxi-only (trend × seasonal) | 6.3% | Standard assignment method |
| Total demand (trend × seasonal) | 3.9% | More predictable |
| Derived taxi (total × fixed share) | 10.0% | Fails — share is trending |

Total demand is more predictable (CV 0.106 vs 0.133) because mode-share noise is removed. However, converting total back to taxi using a fixed share ratio fails because the share is continuously shifting. A dynamic share model (trending or rolling share) would be needed.

---

## Impact on Assignment Recommendations

### What changes

1. **Growth narrative:** NNS taxi growth is mode-share recapture, not market expansion. The underlying market is flat to declining.
2. **Growth ceiling:** Taxi share cannot increase indefinitely. At 11.8% share (current), there is room to grow, but the ceiling is perhaps 15-20%, not unbounded.
3. **Competitive framing:** The real competitor is not other taxi companies (HHI was already low in the assignment) but ride-hail. Entry strategy should consider differentiation from ride-hail.
4. **O'Hare reframing:** Not a declining market. Taxis are losing to ride-hail. An airport strategy should account for ride-hail dominance.

### What does not change

1. **NNS is still the best taxi entry point.** Taxis are gaining share there, regardless of cause.
2. **The forecast methodology is valid.** Taxi-only trend × seasonal achieves 6.3% MAPE on taxi volumes. The model predicts what taxis will do, not why.
3. **Corridor analysis (Phase 5) remains valid.** It models taxi corridors, and taxi corridors are growing in NNS.

### New recommendation

**Product implication:** forecasting total mobility demand (3.9% MAPE) is more accurate than taxi-only (6.3%). The product implication: acquire ride-hail data alongside taxi data. Total demand is the more stable signal; mode-share is the business intelligence layer on top. This directly demonstrates a data-acquisition model — combining data sources creates more value than either alone.

---

## Caveats

1. **2025→2026 comparison uses 18 weeks only (H1).** Directional but not definitive. Could reflect seasonal patterns. The 2024→2025 full-year comparison is more robust.
2. **Mode-share trend may not continue.** Taxi share recovery could plateau, reverse, or accelerate. Without understanding the causal mechanism (pricing? regulation? service quality?), extrapolation is speculative.
3. **TNP data is pickup-area only.** No O-D pairs for ride-hail, so corridor-level mode-share analysis is not possible.

---

## Narrative

"The ride-hail data revealed that NNS's +30% taxi growth is primarily a mode-share shift — taxi recapturing share from ride-hail — not total market growth. Total mobility demand is roughly flat. This doesn't invalidate the NNS recommendation — it's still the best taxi entry point — but it reframes the growth ceiling and competitive dynamics. The real competitor isn't other taxi companies, it's ride-hail. O'Hare tells the opposite story: total demand is growing 6%, but taxis are losing share. This is exactly the kind of insight you can only get by cross-referencing multiple data sources — the core of a data-acquisition business model."

---

## Connection to Route-Level Forecasting

| Finding | Relevance |
|---------|-----------------|
| Total demand more predictable than taxi-only | Buy ride-hail data → more accurate forecasts |
| Mode-share is a business intelligence signal | Separate product layer: demand forecast + share forecast |
| Cross-referencing taxi + ride-hail reveals hidden dynamics | Demonstrates the "acquire data, build products" model |
| O'Hare reversal only visible with combined data | Single data source gives wrong conclusions |
