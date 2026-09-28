"""
Chicago Taxi Market Intelligence — Streamlit Demo
===================================================
Single-file Streamlit demo app.
Tab 1: Analysis (methodology + findings)
Tab 2: Decision Engine (AI pipeline + ReAct)

Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import html
import os
import time
from pathlib import Path
from datetime import timedelta, date
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Chicago Taxi Market Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
HOLIDAY_FACTORS = {
    "Christmas": 0.254, "Thanksgiving": 0.390, "Day After Christmas": 0.482,
    "July 4th": 0.540, "Christmas Eve": 0.628, "Memorial Day": 0.637,
    "Black Friday": 0.639, "Labor Day": 0.698, "St Patrick's Parade Chicago": 1.460,
    "St Patrick's Day": 1.294, "St Patrick's Sunday": 1.294,
    "New Year's Day": 0.700, "Day After NY": 0.750,
}
CONVENTION_FACTORS = {
    "National Restaurant Association Show": 1.236, "RSNA Annual Meeting": 1.179,
    "ProMat": 1.165, "PACK EXPO International": 1.123, "ASCO Annual Meeting": 1.121,
}
AREA_NAMES = {8: "Near North Side", 32: "Loop", 76: "O'Hare",
              28: "Near West Side", 33: "Near South Side"}

COMPETITIVE_SHIFT_WEEKS = {
    (2025, 37): {
        "direction": "gaining",
        "taxi_share_trend": "9.8% → 11.8% over 2.5 years (+2.0pp)",
        "taxi_growth_yoy": "+12.9%", "ridehail_growth_yoy": "+0.9%",
        "total_growth_yoy": "+2.1%",
        "opportunity": "Fri-Sat 22:00-02:00 ride-hail charges $18-25 (illustrative), taxi $13.91",
        "extra_trips_per_week": 100,
    },
}
GROWTH_THRESHOLD_WEEKS = {
    (2025, 42): {
        "corridor": "NNS→Loop",
        "corridor_share_of_nns": "20%",
        "consecutive_weeks_above": 4,
        "company_corridor_trips_week": 282,
        "breakeven_trips_week": 840,
        "corridor_fare": 8.42,
        "corridor_seconds": 544,
        "corridor_revenue_per_hr": 55.7,
        "internal_fare": 8.85,
        "internal_seconds": 367,
        "internal_revenue_per_hr": 86.8,
    },
}

# Typical non-event baselines — median NNS weekly actual (backtest_8.csv) of the
# preceding 3 CLEAN weeks, excluding any week containing a validated holiday
# factor <0.7 or convention factor >1.1. Used for each card's forward-looking
# expected_line. Clean weeks used per demo week:
#   W49 / W48 : W45,W46,W47  (30,955 / 30,139 / 31,643)  -> median 30,955
#   W52       : W47,W50,W51  (31,643 / 37,950 / 36,116)  -> median 36,116
#   W42       : W39,W40,W41  (34,128 / 34,272 / 34,737)  -> median 34,272
#   W40       : W37,W38,W39  (31,718 / 34,318 / 34,128)  -> median 34,128
#   W37       : W33,W34,W35  (32,434 / 29,540 / 29,669)  -> median 29,669
# typical DAILY value = typical_week / 7.
TYPICAL_BASELINES = {
    (2025, 49): 30955, (2025, 48): 30955, (2025, 52): 36116,
    (2025, 42): 34272, (2025, 40): 34128, (2025, 37): 29669,
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=20, r=20, t=40, b=20),
    font=dict(size=12),
    height=380,
)

# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    .action-card {
        border: 1px solid #e0e0e0; border-radius: 10px;
        padding: 1.2rem; margin: 0.5rem 0;
        background: linear-gradient(135deg, #fafbfc 0%, #f5f7fa 100%);
    }
    .action-type {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em;
        padding: 0.2rem 0.6rem; border-radius: 4px; display: inline-block;
    }
    .type-demand-drop {background: #fde8e8; color: #c0392b;}
    .type-resource-optimize {background: #e8f4fd; color: #2980b9;}
    .type-demand-surge {background: #e8fde8; color: #27ae60;}
    .type-growth-threshold {background: #fdf5e8; color: #e67e22;}
    .type-competitive-shift {background: #f0e8fd; color: #8e44ad;}
    .type-anomaly-escalation {background: #fde8e8; color: #e74c3c;}
    .metric-row {display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.5rem 0;}
    .metric-item {
        background: #f8f9fa; border-radius: 6px; padding: 0.4rem 0.8rem;
        font-size: 0.9rem;
    }
    div[data-testid="stExpander"] details summary p {font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CACHED DATA LOADING
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_mode1_results():
    with open("outputs/phase7_mode1_results.json") as f:
        return json.load(f)

@st.cache_data
def load_total_market():
    df = pd.read_csv("outputs/phase6_nns_total_market.csv")
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df

@st.cache_data
def load_backtest():
    return pd.read_csv("data/from_assignment/backtest_8.csv")

@st.cache_data
def load_holidays():
    df = pd.read_csv("data/external/holidays_chicago.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_conventions():
    df = pd.read_csv("data/external/conventions_chicago.csv")
    df["date_start"] = pd.to_datetime(df["date_start"])
    df["date_end"] = pd.to_datetime(df["date_end"])
    return df

@st.cache_data
def load_config():
    with open("config/action_config.json") as f:
        return json.load(f)

@st.cache_data
def load_taxi_weekly():
    return pd.read_csv("data/from_assignment/area_week.csv")

@st.cache_data
def load_tnp_weekly():
    return pd.read_csv("data/external/tnp_area_week.csv")

@st.cache_resource
def load_faiss_index():
    import faiss
    return faiss.read_index("data/derived/gdelt_faiss.index")

@st.cache_resource
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.encode([
        "Chicago Christmas Eve Christmas Day After Christmas travel demand",
        "Chicago Thanksgiving Black Friday RSNA Annual Meeting travel demand",
        "Chicago RSNA Annual Meeting travel demand",
        "Chicago travel demand",
    ], normalize_embeddings=True)
    return model

@st.cache_data
def load_articles_meta():
    return pd.read_pickle("data/derived/gdelt_articles_meta.pkl")


# ══════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS (for Mode 2 ReAct)
# ══════════════════════════════════════════════════════════════════════
def get_forecast(area: int, year: int, week: int) -> dict:
    bt = load_backtest()
    if area != 8:
        return {"error": "Only area 8 (NNS) in this demo."}
    row = bt[(bt["iso_year"] == year) & (bt["iso_week"] == week)]
    if len(row) == 0:
        return {"error": f"No data for {year} W{week}. Range: W27-W52."}
    r = row.iloc[0]
    actual, predicted = int(r["actual"]), int(r["trend_seasonal"])
    error_pct = round((actual - predicted) / actual * 100, 1)
    return {"area": area, "area_name": "Near North Side", "year": year, "week": week,
            "actual_trips": actual, "forecast_trips": predicted, "error_pct": error_pct,
            "direction": "under-forecast" if error_pct > 0 else "over-forecast",
            "is_anomaly": abs(error_pct) > 10}

def search_events(year: int, week: int) -> dict:
    hol = load_holidays()
    conv = load_conventions()
    jan4 = date(year, 1, 4)
    w1_start = jan4 - timedelta(days=jan4.isoweekday() - 1)
    ws = pd.Timestamp(w1_start + timedelta(weeks=week - 1))
    we = ws + pd.Timedelta(days=6)
    events = []
    for _, h in hol[(hol["date"] >= ws) & (hol["date"] <= we)].iterrows():
        f = HOLIDAY_FACTORS.get(h["event_name"])
        events.append({"type": "holiday", "name": h["event_name"],
                       "date": h["date"].strftime("%Y-%m-%d"), "category": h["category"],
                       "factor": f, "impact_pct": f"{(f-1)*100:+.0f}%" if f else "unknown",
                       "source": "Phase 3 validated" if f else "unvalidated"})
    for _, c in conv.iterrows():
        if c["date_start"] <= we and c["date_end"] >= ws:
            f = CONVENTION_FACTORS.get(c["event_name"])
            od = (min(c["date_end"], we) - max(c["date_start"], ws)).days + 1
            events.append({"type": "convention", "name": c["event_name"],
                           "overlap_days": od, "est_attendees": int(c["est_attendees"]),
                           "factor": f, "impact_pct": f"{(f-1)*100:+.0f}%" if f else "unknown",
                           "source": "Phase 3 validated" if f else "unvalidated"})
    return {"year": year, "week": week, "events": events, "n_events": len(events)}

def search_news(query: str, year: int, week: int, top_k: int = 5) -> dict:
    try:
        idx = load_faiss_index()
        model = load_embed_model()
        meta = load_articles_meta()
    except Exception as e:
        return {"error": f"FAISS not available: {e}", "articles": []}
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    dists, idxs = idx.search(q_emb, top_k * 20)
    results = []
    for d, i in zip(dists[0], idxs[0]):
        if i < 0: continue
        r = meta.iloc[i]
        if r["iso_year"] == year and abs(r["iso_week"] - week) <= 1:
            results.append({"title": r["title"], "source": r["source"],
                            "date": str(r["article_date"])[:10],
                            "tone": round(float(r["tone"]), 1), "url": r["url"]})
            if len(results) >= top_k: break
    return {"query": query, "articles": results, "n_found": len(results)}

def suggest_action(area: int, year: int, week: int,
                   forecast_data: dict = None, event_data: dict = None) -> dict:
    """Generate ONE primary action recommendation with revenue impact.
    Self-contained: if forecast_data / event_data are not supplied (e.g. the
    ReAct agent passes only area/year/week), fetch them internally so the
    decision always runs on complete data.
    Priority: GROWTH_THRESHOLD > COMPETITIVE_SHIFT > ANOMALY > DEMAND_DROP > RESOURCE_OPTIMIZE > DEMAND_SURGE > NONE
    """
    if forecast_data is None:
        forecast_data = get_forecast(area=area, year=year, week=week)
    if event_data is None:
        event_data = search_events(year=year, week=week)
    cfg = load_config()
    C = cfg["company_profile"]
    CFG = cfg["action_types"]
    error_pct = forecast_data.get("error_pct", 0)
    events = event_data.get("events", [])

    holiday_events = [e for e in events if e["type"] == "holiday" and e.get("factor")]
    convention_events = [e for e in events if e["type"] == "convention" and e.get("factor")]
    neg_holidays = [h for h in holiday_events if h["factor"] < CFG["RESOURCE_OPTIMIZE"]["holiday_threshold"]]
    pos_conventions = [c for c in convention_events
                       if c["factor"] > CFG["DEMAND_SURGE"]["surge_threshold"]
                       and c.get("est_attendees", 0) >= CFG["DEMAND_SURGE"]["min_attendees"]]
    has_validated_cause = bool(neg_holidays or pos_conventions)

    # Actual-vs-predicted moves OUT of the headline (except ANOMALY_ESCALATION)
    # and INTO evidence. Each card leads with a forward-looking expected_line;
    # this validation string is appended to every card's evidence.
    _actual = forecast_data.get("actual_trips")
    _predicted = forecast_data.get("forecast_trips")
    _model_val = (
        f" Model validation: {_actual:,} actual vs {_predicted:,} predicted "
        f"({error_pct:+.1f}%)."
        if _actual is not None and _predicted is not None else ""
    )

    # Priority 1: GROWTH_THRESHOLD
    gt = GROWTH_THRESHOLD_WEEKS.get((year, week))
    if gt:
        return {
            "action_type": "GROWTH_THRESHOLD",
            "headline": f"{gt['corridor']} sustained demand — expansion NOT recommended",
            "expected_line": (
                f"Corridor demand above threshold {gt['consecutive_weeks_above']} "
                f"consecutive weeks ({gt['corridor']}, {gt['corridor_share_of_nns']} of NNS trips)"
            ),
            "trigger": f"{gt['corridor']} ({gt['corridor_share_of_nns']} of NNS trips) above threshold for {gt['consecutive_weeks_above']} consecutive weeks",
            "what_to_do": [
                f"Do NOT deploy dedicated fleet to {gt['corridor'].split('→')[1]}",
                f"Volume: {gt['company_corridor_trips_week']} trips/week vs {gt['breakeven_trips_week']} breakeven — insufficient for dedicated deployment",
                f"Revenue/hr: ${gt['corridor_revenue_per_hr']}/occupied-hr ({gt['corridor']}) vs ${gt['internal_revenue_per_hr']}/occupied-hr (NNS internal) — even dispatching existing fleet is inferior",
                "Maintain corridor watchlist — revisit if volume doubles",
            ],
            "revenue_impact": {
                "expansion_rejected": f"{gt['company_corridor_trips_week']} trips/wk ≪ {gt['breakeven_trips_week']} breakeven",
                "opportunity_cost": f"${gt['internal_revenue_per_hr']}/hr internal vs ${gt['corridor_revenue_per_hr']}/hr Loop",
            },
            "evidence": (
                f"{gt['corridor']} avg fare ${gt['corridor_fare']} / {gt['corridor_seconds']}s = "
                f"${gt['corridor_revenue_per_hr']}/occupied-hr. "
                f"NNS internal ${gt['internal_fare']} / {gt['internal_seconds']}s = "
                f"${gt['internal_revenue_per_hr']}/occupied-hr. "
                f"Loop→NNS return exists (849 trips/day) but round-trip still inferior."
                + _model_val
            ),
            "confidence": "HIGH — based on actual corridor economics from Phase 5",
            "requires_approval": False,
        }

    # Priority 2: COMPETITIVE_SHIFT
    cs = COMPETITIVE_SHIFT_WEEKS.get((year, week))
    if cs:
        extra_trips = cs["extra_trips_per_week"]
        weekly_rev = extra_trips * C["avg_fare"]
        annual_rev = weekly_rev * 52
        return {
            "action_type": "COMPETITIVE_SHIFT",
            "headline": "Mode-share momentum — capture ride-hail surge hours",
            "expected_line": "Taxi share 9.8% → 11.8% over 2.5 years; ride-hail flat-to-declining",
            "trigger": f"Taxi share trend: {cs['taxi_share_trend']}",
            "what_to_do": [
                f"Competitive pricing during ride-hail surge hours ({cs['opportunity']})",
                "Target Fri-Sat 22:00-02:00 when ride-hail prices peak",
                f"Capture estimated {extra_trips} additional trips/week at zero fleet cost",
            ],
            "revenue_impact": {
                "weekly_gain": f"${weekly_rev:,.0f}/week ({extra_trips} trips × ${C['avg_fare']})",
                "annual_gain": f"${annual_rev:,.0f}/year",
                "fleet_cost": "$0 — use existing idle night-shift capacity",
            },
            "evidence": (
                f"Phase 6: taxi growth {cs['taxi_growth_yoy']}, ride-hail {cs['ridehail_growth_yoy']}, "
                f"total {cs['total_growth_yoy']}. Total demand flat — taxi growth is share capture. "
                f"Taxi share: {cs['taxi_share_trend']}."
                + _model_val
            ),
            "confidence": "Trend: HIGH (2.5 years data). Capture estimate: ASSUMPTION (100 trips/week unvalidated)",
            "requires_approval": True,
        }

    # Priority 3: ANOMALY_ESCALATION
    if abs(error_pct) > CFG["ANOMALY_ESCALATION"]["error_threshold_pct"] and not has_validated_cause:
        unvalidated = [e for e in events if not e.get("factor")]
        return {
            "action_type": "ANOMALY_ESCALATION",
            "headline": f"Forecast deviation {error_pct:+.1f}% — no validated cause",
            # ANOMALY_ESCALATION keeps actual-vs-predicted as its headline story,
            # so it carries no forward-looking expected_line.
            "expected_line": None,
            "trigger": f"W{week} error {error_pct:+.1f}% exceeds ±{CFG['ANOMALY_ESCALATION']['error_threshold_pct']}% threshold",
            "what_to_do": [
                "Flag for analyst review",
                "Check local conditions not in event calendar",
                "Review whether trend model is systematically under-predicting Q4",
            ],
            "context": [
                f"Unvalidated events detected: {', '.join(e['name'] for e in unvalidated)}"
                if unvalidated else "No events detected in calendar",
            ],
            "revenue_impact": "Unknown — requires investigation",
            "evidence": (
                f"Forecast error {error_pct:+.1f}% with no Phase 3 validated event. "
                "System escalates instead of guessing."
                + _model_val
            ),
            "confidence": "N/A — escalation, not recommendation",
            "requires_approval": False,
        }

    # Priority 4: DEMAND_DROP (extreme holiday, factor < 0.3)
    if neg_holidays:
        strongest = min(neg_holidays, key=lambda h: h["factor"])
        factor = strongest["factor"]
        reduction_pct = round((1 - factor) * 100)
        cars_to_park = round(C["fleet_size"] * (1 - factor))
        cars_active = C["fleet_size"] - cars_to_park
        daily_saving = cars_to_park * C["cost_per_car_day"]

        if factor < CFG["DEMAND_DROP"]["extreme_threshold"]:
            _tb = TYPICAL_BASELINES.get((year, week))
            _expected = (
                f"Expected on Dec 25: ~{_tb / 7 * factor:,.0f} vs ~{_tb / 7:,.0f} "
                f"typical day (−75%, Christmas factor)"
            ) if _tb else None
            return {
                "action_type": "DEMAND_DROP",
                "headline": f"Park {cars_to_park} of {C['fleet_size']} cars — {strongest['name']} reduces demand {reduction_pct}%",
                "expected_line": _expected,
                "trigger": f"{strongest['name']} (factor {factor}, {strongest['impact_pct']})",
                "what_to_do": [
                    f"Reduce fleet from {C['fleet_size']} to {cars_active} cars ({reduction_pct}% reduction)",
                    f"Park {cars_to_park} cars for the holiday period",
                    "Schedule full fleet redeployment for post-holiday recovery",
                ],
                "revenue_impact": {
                    "daily_saving": f"${daily_saving:,.0f}",
                    "calculation": f"{cars_to_park} parked × ${C['cost_per_car_day']}/car/day",
                    "period": strongest.get("date", f"W{week}"),
                },
                "evidence": (
                    f"{strongest['name']} factor {factor} ({strongest['impact_pct']}). "
                    "Phase 3 validated. Cross-year consistent direction."
                    + _model_val
                ),
                "confidence": "HIGH",
                "requires_approval": True,
            }

        # Priority 5: RESOURCE_OPTIMIZE (moderate holiday + airport)
        else:
            redirect_pct = CFG["RESOURCE_OPTIMIZE"]["airport_redirect_pct"]
            trips_per_car = CFG["RESOURCE_OPTIMIZE"]["airport_trips_per_car_day"]
            redirect_cars = round(cars_to_park * redirect_pct)
            remaining_idle = cars_to_park - redirect_cars
            airport_revenue = redirect_cars * trips_per_car * C["avg_airport_fare"]
            idle_saving = remaining_idle * C["cost_per_car_day"]
            total_daily = airport_revenue + idle_saving
            _tb = TYPICAL_BASELINES.get((year, week))
            _expected = (
                f"Thanksgiving −61% (Nov 27) + RSNA +18% overlap → redirect idle "
                f"capacity to airport premium. Expected on Nov 27: "
                f"~{_tb / 7 * factor:,.0f} vs ~{_tb / 7:,.0f} typical day"
            ) if _tb else None
            return {
                "action_type": "RESOURCE_OPTIMIZE",
                "headline": f"Redirect {redirect_cars} idle cars to airport premium routes (${C['avg_airport_fare']} avg fare)",
                "expected_line": _expected,
                "trigger": f"{strongest['name']} (factor {factor}, {strongest['impact_pct']}) + airport corridor available",
                "what_to_do": [
                    f"Reduce fleet from {C['fleet_size']} to {cars_active} cars ({reduction_pct}% reduction)",
                    f"Redirect {redirect_cars} cars ({redirect_pct:.0%} of idle) to airport dispatch",
                    f"Park remaining {remaining_idle} idle cars",
                    f"Airport fare: ${C['avg_airport_fare']} avg vs ${C['avg_fare']} standard ({C['avg_airport_fare']/C['avg_fare']:.1f}x premium)",
                ],
                "revenue_impact": {
                    "airport_revenue": f"${airport_revenue:,.0f}/day ({redirect_cars} × {trips_per_car} trips × ${C['avg_airport_fare']})",
                    "idle_saving": f"${idle_saving:,.0f}/day ({remaining_idle} parked × ${C['cost_per_car_day']})",
                    "total_daily_benefit": f"${total_daily:,.0f}/day",
                    "vs_no_action": f"Without action: {cars_to_park} cars idle at ${C['cost_per_car_day']}/day = ${cars_to_park * C['cost_per_car_day']:,.0f}/day wasted",
                },
                "evidence": (
                    f"{strongest['name']} factor {factor} ({strongest['impact_pct']}). "
                    f"Assumed {trips_per_car} airport legs/car/day (~2hr cycle: 39-min trip + empty return, pickups permit-gated); "
                    f"breakeven {round(C['cost_per_car_day'] / C['avg_airport_fare'], 1)} legs/day. "
                    f"Airport fare ${C['avg_airport_fare']} vs ${C['avg_fare']} standard "
                    "(Phase 5: 5.8% of NNS trips = 19% of revenue)."
                    + _model_val
                ),
                "confidence": "HIGH",
                "requires_approval": True,
            }

    # Priority 6: DEMAND_SURGE (convention)
    if pos_conventions:
        strongest = max(pos_conventions, key=lambda c: c["factor"])
        factor = strongest["factor"]
        extra_cars = round(C["fleet_size"] * (factor - 1))
        overlap_days = strongest.get("overlap_days", 3)
        trips_per_car = CFG["DEMAND_SURGE"]["trips_per_temp_car_day"]
        daily_extra_revenue = extra_cars * trips_per_car * C["avg_fare"]
        daily_extra_cost = extra_cars * C["cost_per_car_day"]
        daily_net = daily_extra_revenue - daily_extra_cost
        total_net = daily_net * overlap_days
        breakeven_trips = round(C["cost_per_car_day"] / C["avg_fare"])

        _tb = TYPICAL_BASELINES.get((year, week))
        _expected = (
            f"Expected: ~{_tb * factor:,.0f} for the week vs ~{_tb:,.0f} typical "
            f"({(factor - 1) * 100:+.0f}%, RSNA, {strongest.get('est_attendees', '?'):,} attendees)"
        ) if _tb else None

        if daily_net > 0:
            return {
                "action_type": "DEMAND_SURGE",
                "headline": f"Add {extra_cars} temporary cars for {strongest['name']} ({strongest.get('est_attendees', '?'):,} attendees)",
                "expected_line": _expected,
                "trigger": f"{strongest['name']} (factor {factor}, {strongest['impact_pct']}, {strongest.get('est_attendees', '?'):,} attendees)",
                "what_to_do": [
                    f"Hire or lease {extra_cars} additional cars for {overlap_days} days",
                    "Deploy near convention venue (McCormick Place / NNS hotels)",
                    f"Increase fleet to {C['fleet_size'] + extra_cars} total",
                ],
                "revenue_impact": {
                    "daily_extra_revenue": f"${daily_extra_revenue:,.0f} ({extra_cars} × {trips_per_car} trips × ${C['avg_fare']})",
                    "daily_extra_cost": f"${daily_extra_cost:,.0f} ({extra_cars} × ${C['cost_per_car_day']})",
                    "daily_net_gain": f"${daily_net:,.0f}",
                    "total_net_gain": f"${total_net:,.0f} over {overlap_days} days",
                },
                "evidence": (
                    f"{strongest['name']} factor {factor} ({strongest['impact_pct']}). "
                    f"Est. attendees: {strongest.get('est_attendees', '?'):,}. "
                    f"Assumed {trips_per_car} trips/car/day; breakeven is {breakeven_trips}."
                    + _model_val
                ),
                "confidence": "MEDIUM — limited historical occurrences per convention",
                "requires_approval": True,
            }
        else:
            return {
                "action_type": "DEMAND_SURGE",
                "headline": f"Maximize existing fleet during {strongest['name']} — adding cars not profitable",
                "expected_line": _expected,
                "trigger": f"{strongest['name']} (factor {factor}, {strongest['impact_pct']}, {strongest.get('est_attendees', '?'):,} attendees)",
                "what_to_do": [
                    f"Do NOT add temporary cars — break-even requires {breakeven_trips} trips/car/day at ${C['avg_fare']} avg fare",
                    f"Maximize utilization of existing {C['fleet_size']} cars during convention hours",
                    "Extend shift hours to capture convention-driven evening demand",
                    "Deploy near McCormick Place / NNS hotels during convention days",
                ],
                "revenue_impact": {
                    "analysis": f"Adding {extra_cars} temp cars would cost ${daily_extra_cost:,.0f}/day but generate only ${daily_extra_revenue:,.0f}/day",
                    "net_loss_avoided": f"${-total_net:,.0f} loss avoided over {overlap_days} days",
                    "recommendation": "Capture surge with existing fleet — zero marginal cost",
                },
                "evidence": (
                    f"{strongest['name']} factor {factor} ({strongest['impact_pct']}). "
                    f"Assumed {trips_per_car} trips/car/day; breakeven is {breakeven_trips}. "
                    f"Conclusion robust unless temp cars exceed {breakeven_trips} trips/day — implausible at +{round((factor-1)*100)}% surge."
                    + _model_val
                ),
                "confidence": "HIGH — math-based rejection",
                "requires_approval": False,
            }

    # Priority 7: NONE
    return {
        "action_type": "NONE",
        "headline": "Normal operations — no action needed",
        "trigger": "No anomaly, no validated event",
        "what_to_do": ["Maintain current fleet deployment"],
        "revenue_impact": "N/A",
        "evidence": "Forecast within normal range, no events detected.",
        "confidence": "N/A",
        "requires_approval": False,
    }

def _market_verdict(tg, ng, totg):
    """Deterministic per-area verdict from the growth numbers (pp = points).
    Rules: |total| < 3% -> flat; taxi - total > 5pp -> mode-share shift.
    Computed here so the LLM never has to infer the verdict itself."""
    flat = abs(totg) < 3
    if flat and (tg - totg) > 5:
        return (f"total {totg:+.1f}% ~ flat; taxi {tg:+.1f}% is mode-share "
                f"capture from ride-hail, not market growth")
    if tg < 0 and totg > 0:
        return (f"taxi {tg:+.1f}% declining but total {totg:+.1f}% growing "
                f"— ride-hail taking share")
    trend = "flat" if flat else ("growing" if totg > 0 else "declining")
    if (ng - totg) > 5:
        return (f"total {totg:+.1f}% {trend}; ride-hail {ng:+.1f}% outpacing "
                f"taxi {tg:+.1f}% — taxi losing share")
    return (f"total {totg:+.1f}% {trend}; taxi {tg:+.1f}% and ride-hail "
            f"{ng:+.1f}% moving together")


def _overall_verdict(rows):
    """rows: list of (name, tg, ng, totg). Deterministic cross-area summary."""
    taxi_shift = [n for (n, tg, ng, totg) in rows if abs(totg) < 3 and (tg - totg) > 5]
    rh_shift = [n for (n, tg, ng, totg) in rows if tg < 0 and totg > 0]
    if taxi_shift and rh_shift:
        return (f"Mixed mode-share dynamics — {', '.join(taxi_shift)}: taxi capturing "
                f"share on flat demand; {', '.join(rh_shift)}: ride-hail capturing share "
                f"as total grows. Neither reflects organic taxi-market growth.")
    if taxi_shift:
        return (f"{', '.join(taxi_shift)}: taxi growth is mode-share capture from "
                f"ride-hail on flat total demand, not market growth.")
    if rh_shift:
        return (f"{', '.join(rh_shift)}: total demand growing but taxi losing share "
                f"to ride-hail.")
    return "No dominant mode-share shift; taxi and total demand moving together."


def get_market_comparison(areas: list) -> dict:
    taxi = load_taxi_weekly()
    tnp = load_tnp_weekly()
    merged = taxi[["pickup_community_area","iso_year","iso_week","trip_count"]].rename(
        columns={"trip_count":"taxi"}).merge(
        tnp.rename(columns={"trip_count":"tnp"}),
        on=["pickup_community_area","iso_year","iso_week"], how="inner")
    results = []
    numeric = []
    for a in areas:
        ad = merged[merged["pickup_community_area"]==a]
        y1,y2 = ad[ad["iso_year"]==2024], ad[ad["iso_year"]==2025]
        ow = set(y1["iso_week"]) & set(y2["iso_week"])
        if not ow: continue
        y1o,y2o = y1[y1["iso_week"].isin(ow)], y2[y2["iso_week"].isin(ow)]
        tg = (y2o["taxi"].sum()/y1o["taxi"].sum()-1)*100
        ng = (y2o["tnp"].sum()/y1o["tnp"].sum()-1)*100
        totg = ((y2o["taxi"].sum()+y2o["tnp"].sum())/(y1o["taxi"].sum()+y1o["tnp"].sum())-1)*100
        name = AREA_NAMES.get(a,f"Area {a}")
        results.append({"area": a, "name": name,
                        "taxi_growth": f"{tg:+.1f}%", "ridehail_growth": f"{ng:+.1f}%",
                        "total_growth": f"{totg:+.1f}%",
                        "verdict": _market_verdict(tg, ng, totg)})
        numeric.append((name, tg, ng, totg))
    return {"comparison": results, "overall_verdict": _overall_verdict(numeric)}


# ══════════════════════════════════════════════════════════════════════
# HELPER: RENDER ACTION CARD
# ══════════════════════════════════════════════════════════════════════
def render_action_card(result):
    """Render a Mode 1 result as a styled action card."""
    # Streamlit markdown treats paired "$" as LaTeX delimiters, which mangles
    # any string with two dollar signs. Escape "$" before passing to markdown/
    # caption/info. (st.metric value= doesn't parse markdown, so it's left as-is.)
    esc = lambda s: str(s).replace("$", "\\$")
    action = result["action"]
    forecast = result["forecast"]
    events = result["events"]
    news = result["news"]
    atype = action["action_type"]

    type_class = f"type-{atype.lower().replace('_', '-')}"

    # Header
    st.markdown(f'<span class="action-type {type_class}">{atype}</span> &nbsp; '
                f'**{result["week"]}**', unsafe_allow_html=True)

    # Card face 📊 line: lead with the forward-looking expected_line when the
    # card has one; ANOMALY_ESCALATION (expected_line is None) keeps its
    # actual-vs-predicted forecast line as the face headline. Either way the
    # actual-vs-predicted numbers remain in the Evidence expander.
    expected_line = action.get("expected_line")
    if expected_line:
        st.markdown(f"📊 **{esc(expected_line)}**")
    else:
        st.markdown(f"📊 **Forecast:** {forecast['actual_trips']:,} actual vs "
                    f"{forecast['forecast_trips']:,} predicted "
                    f"({forecast['error_pct']:+.1f}%)")

    # Events
    validated = [e for e in events.get("events", []) if e.get("factor")]
    unvalidated = [e for e in events.get("events", []) if not e.get("factor")]
    if validated:
        ev_str = " · ".join(f"{e['name']} ({e.get('impact_pct','')})" for e in validated)
        st.markdown(f"📅 **Events:** {ev_str}")
    if unvalidated:
        uv_str = ", ".join(e["name"] for e in unvalidated)
        st.caption(f"Unvalidated: {uv_str}")

    # News
    articles = news.get("articles", [])
    if articles:
        with st.expander(f"📰 News ({len(articles)} articles)", expanded=False):
            for a in articles[:3]:
                st.markdown(f"- [{a['source']}] {a['title'][:90]}...")

    st.divider()

    # Recommendation
    st.markdown(f"💡 **{esc(action['headline'])}**")
    if action.get("trigger"):
        st.caption(f"Trigger: {esc(action['trigger'])}")
    if isinstance(action.get("what_to_do"), list):
        for step in action["what_to_do"]:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {esc(step)}")

    # Revenue impact
    ri = action.get("revenue_impact", {})
    if isinstance(ri, dict):
        n_cols = min(len(ri), 3)
        cols = st.columns(n_cols)
        for i, (k, v) in enumerate(ri.items()):
            with cols[i % n_cols]:
                label = k.replace("_", " ").title()
                v_str = str(v)
                # Uniform renderer: caption label + wrapping value at a fixed
                # size so nothing truncates regardless of value length. Value is
                # inside raw HTML, so use html.escape (NOT the markdown-oriented
                # esc, whose "\$" would render literally here).
                st.caption(label)
                st.markdown(
                    f"<div style='font-size:1.5rem;font-weight:600;"
                    f"line-height:1.3'>{html.escape(v_str)}</div>",
                    unsafe_allow_html=True)
    elif isinstance(ri, str):
        st.info(f"💰 {esc(ri)}")

    # Evidence (expander — shows assumption labels and data sources)
    evidence = action.get("evidence")
    if evidence:
        with st.expander("📋 Evidence & assumptions", expanded=False):
            st.markdown(esc(evidence))

    # Footer
    conf = action.get("confidence", "N/A")
    latency = result.get("latency_s", 0)
    st.caption(f"Confidence: {conf} · Pipeline: {latency}s · Cost: $0")

    # Buttons — Approve/Modify/Reject for actions, Acknowledge for rejections/escalations
    if action.get("action_type") == "ANOMALY_ESCALATION":
        st.info("📋 Notification — flagged for analyst review")
    elif action.get("requires_approval", True):
        c1, c2, c3, _ = st.columns([1, 1, 1, 3])
        with c1:
            if st.button("✅ Approve", key=f"approve_{result['week']}"):
                st.toast("Action logged for execution.", icon="✅")
        with c2:
            if st.button("✏️ Modify", key=f"modify_{result['week']}"):
                st.toast("Opening modification panel...", icon="✏️")
        with c3:
            if st.button("❌ Reject", key=f"reject_{result['week']}"):
                st.toast("Action rejected.", icon="❌")
    else:
        # Rejection cards (GROWTH_THRESHOLD, DEMAND_SURGE rejection)
        if st.button("✅ Acknowledge", key=f"ack_{result['week']}"):
            st.toast("Acknowledged.", icon="✅")


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.title("Chicago Taxi Market Intelligence")
st.caption("150-car fleet entering Near North Side · Target: 5% market share · $13.91 avg fare")

# Pre-load FAISS + embed model so first ReAct query doesn't pay load cost
try:
    load_faiss_index()
    load_embed_model()
    load_articles_meta()
except Exception:
    pass  # Mode 2 will show fallback message if these fail

tab1, tab2 = st.tabs(["📊 Analysis", "⚡ Decision Engine"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tab1:

    # ── Section 1: Forecast Performance ──────────────────────────
    st.subheader("Forecast Performance")

    bench = pd.DataFrame({
        "Model": ["Trend × seasonal", "M1 (DOW×month×holiday)",
                   "Global XGB (183 corridors)", "TimesFM zero-shot", "N-BEATS trained"],
        "MAPE": [6.01, 6.95, 7.10, 8.20, 10.23],
        "Scope": ["NNS weekly", "NNS daily→weekly", "Corridors daily→weekly",
                   "NNS daily→weekly", "NNS daily→weekly"],
    })

    fig = px.bar(bench, y="Model", x="MAPE", orientation="h",
                 text=bench["MAPE"].apply(lambda x: f"{x:.2f}%"),
                 color="MAPE", color_continuous_scale=["#2ecc71", "#e74c3c"],
                 hover_data=["Scope"])
    fig.add_vline(x=6.01, line_dash="dash", line_color="#c0392b", line_width=1.5,
                  annotation_text="6.01% baseline", annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed"), xaxis_title="Weekly MAPE (%)")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("On a single dense series (882 days), classical decomposition is near-optimal. "
               "ML's advantage emerges at cross-corridor scale — the data regime of route-level travel products.")

    # ── Section 2: Corridor Intelligence ─────────────────────────
    st.subheader("Corridor Intelligence")
    col_corr, col_cold = st.columns(2)

    with col_corr:
        st.markdown("**Global vs Per-Corridor Model (weekly)**")
        corr_data = pd.DataFrame({
            "Tier": ["Dense (23)"]*3 + ["Medium (28)"]*3 + ["Sparse (132)"]*3,
            "Model": ["Trailing Mean", "Per-corridor XGB", "Global XGB"]*3,
            "MAE": [408.6, 367.6, 335.5, 53.5, 50.1, 41.9, 8.1, 9.8, 7.4],
        })
        colors = {"Trailing Mean": "#bdc3c7", "Per-corridor XGB": "#95a5a6", "Global XGB": "#2980b9"}
        fig2 = px.bar(corr_data, x="Tier", y="MAE", color="Model", barmode="group",
                      color_discrete_map=colors, text_auto=".1f")
        fig2.update_layout(**PLOTLY_LAYOUT, yaxis_title="MAE (trips/week)",
                           legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Per-corridor XGB **overfits** on sparse routes (worse than trailing mean). "
                   "Global model wins everywhere. Daily granularity: **47% MAPE reduction** on dense "
                   "corridors (18.2%→9.7%; MAE 407→271).")

    with col_cold:
        st.markdown("**Cold-Start Ramp-Up**")
        gap_data = pd.DataFrame({
            "Week": list(range(1, 27)),
            "Gap": [1413.2, 1648.7, 92.0, 97.2, -0.1, 0.3, 1.2, -0.4,
                    -3.3, -2.8, 0.7, 1.3, -3.7, -0.3, 1.3, -1.0,
                    -0.7, 1.6, 0.4, 0.7, -0.2, 0.1, 1.9, -0.8, -0.1, 0.1],
        })
        # Clip for readability
        gap_display = gap_data.copy()
        gap_display["Gap_clipped"] = gap_display["Gap"].clip(-20, 120)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=gap_display["Week"], y=gap_display["Gap_clipped"],
                                   mode="lines+markers", line=dict(color="#2980b9", width=2),
                                   marker=dict(size=5)))
        fig3.add_hline(y=0, line_dash="dash", line_color="#27ae60", line_width=1)
        fig3.add_annotation(x=5, y=0, text="Week 5: converged ✓",
                            showarrow=True, arrowhead=2, ay=-40, font=dict(color="#27ae60"))
        fig3.update_layout(**PLOTLY_LAYOUT, yaxis_title="MAE gap vs full-history (%)",
                           xaxis_title="Weeks since route launch")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("New route → 4 weeks to converge. **TimesFM for days 1-4, global model from week 5.** "
                   "No route ever goes without a forecast.")

    # ── Section 3: Total Market View ─────────────────────────────
    st.subheader("Total Market View")
    col_chart, col_metrics = st.columns([3, 2])

    with col_chart:
        tm = load_total_market().sort_values("week_start").reset_index(drop=True)
        # Exclude the trailing incomplete ISO week: the ride-hail feed cuts off
        # mid-week, leaving a partial final week with anomalously low volume.
        # Only the last row is a candidate — never drop interior weeks.
        min_complete = 0.6 * tm["total_trips"].median()
        if len(tm) > 1 and tm["total_trips"].iloc[-1] < min_complete:
            tm = tm.iloc[:-1]

        # Rolling 8-week YoY growth: trailing 8-week trip sum vs the same 8-week
        # window 52 weeks earlier. Weekly rows are contiguous (52 ISO weeks/year).
        for col, label in [("taxi_trips", "Taxi"), ("tnp_trips", "Ride-hail"),
                           ("total_trips", "Total")]:
            roll8 = tm[col].rolling(8).sum()
            tm[label] = (roll8 / roll8.shift(52) - 1) * 100
        yoy = tm.dropna(subset=["Taxi", "Ride-hail", "Total"])

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=yoy["week_start"], y=yoy["Taxi"], name="Taxi",
                                   line=dict(color="#2980b9", width=2.5)))
        fig4.add_trace(go.Scatter(x=yoy["week_start"], y=yoy["Ride-hail"], name="Ride-hail",
                                   line=dict(color="#e74c3c", width=2)))
        fig4.add_trace(go.Scatter(x=yoy["week_start"], y=yoy["Total"], name="Total",
                                   line=dict(color="#7f8c8d", width=2, dash="dot")))
        fig4.add_hline(y=0, line_dash="dash", line_color="#7f8c8d", line_width=1)
        fig4.update_layout(**PLOTLY_LAYOUT, yaxis_title="Rolling 8-week YoY growth (%)",
                           legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig4, use_container_width=True)

    with col_metrics:
        st.markdown("**NNS 2024→2025**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Taxi", "+12.9%")
        m2.metric("Ride-hail", "+0.9%")
        m3.metric("Total", "+2.1%")

        st.markdown("**NNS 2025→2026** (18 weeks)")
        m4, m5, m6 = st.columns(3)
        m4.metric("Taxi", "+21.0%")
        m5.metric("Ride-hail", "-4.7%", delta_color="inverse")
        m6.metric("Total", "-2.2%", delta_color="inverse")

        st.error("**Verdict: MODE-SHARE SHIFT.** Total demand flat. "
                 "Taxi gaining share from ride-hail, not real growth.")

        st.markdown("**O'Hare (mirror image)**")
        o1, o2, o3 = st.columns(3)
        o1.metric("Taxi", "-5.2%", delta_color="inverse")
        o2.metric("Ride-hail", "+10.0%")
        o3.metric("Total", "+6.2%")

    st.caption("Forecasting total demand (MAPE 3.9%) is more accurate than taxi-only (6.3%). "
               "Product implication: acquire ride-hail data to model total mobility demand.")

    # ── Section 4: Feature Validation ────────────────────────────
    st.subheader("Feature Validation")
    col_yes, col_no = st.columns(2)

    with col_yes:
        st.success("**✅ Confirmed Features**")
        st.markdown("**Holidays:** -20% to -75% impact (cross-year validated)\n\n"
                    "**Conventions:** +12% to +24% impact (McCormick Place)")
        with st.expander("Show holiday factor table"):
            hf = pd.DataFrame(
                [(name, factor, f"{(factor - 1) * 100:+.1f}%")
                 for name, factor in sorted(HOLIDAY_FACTORS.items(), key=lambda kv: kv[1])],
                columns=["Holiday", "Factor", "Impact"])
            st.dataframe(hf, hide_index=True, use_container_width=True)

        with st.expander("Show convention factor table"):
            cf = pd.DataFrame([
                ("Nat'l Restaurant Assoc. Show", 1.236, "+23.6%", 8), ("RSNA", 1.179, "+17.9%", 5),
                ("ProMat", 1.165, "+16.5%", 2), ("PACK EXPO", 1.123, "+12.3%", 4),
                ("ASCO", 1.121, "+12.1%", 10),
            ], columns=["Convention", "Factor", "Impact", "N"])
            st.dataframe(cf, hide_index=True, use_container_width=True)

    with col_no:
        st.error("**❌ Rejected Features**")
        st.markdown("**Weather:** r ≈ 0 after DOW control (882 days tested)\n\n"
                    "**Sports:** games occur 61% of days — zero discriminating signal\n\n"
                    "**Day-of-month:** confounded with day-of-week")


# ══════════════════════════════════════════════════════════════════════
# TAB 2: DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════
with tab2:

    # ── Mode 1: Deterministic ────────────────────────────────────
    st.subheader("Action Recommendations")

    mode1 = load_mode1_results()
    # Map results by label for button lookup. Button order follows the JSON
    # (DEMO_WEEKS) order — not hardcoded.
    week_map = {r["label"]: r for r in mode1}
    labels = list(week_map.keys())
    # Icons keyed by action type so each icon tracks its card regardless of
    # the week ordering.
    ACTION_ICONS = {
        "DEMAND_SURGE": "🏛️", "RESOURCE_OPTIMIZE": "✈️", "DEMAND_DROP": "🎄",
        "GROWTH_THRESHOLD": "📈", "COMPETITIVE_SHIFT": "🔄", "ANOMALY_ESCALATION": "⚠️",
    }

    # Button row
    cols = st.columns(len(labels))
    for i, label in enumerate(labels):
        r = week_map[label]
        icon = ACTION_ICONS.get(r["action"]["action_type"], "•")
        with cols[i]:
            if st.button(f"{icon} {r['week']}", key=f"btn_{label}", use_container_width=True):
                st.session_state.selected_week = label

    # Show selected action card
    selected = st.session_state.get("selected_week", labels[0])
    if selected in week_map:
        render_action_card(week_map[selected])

    # ── Latency Benchmark ────────────────────────────────────────
    st.divider()
    st.subheader("Latency & Cost Benchmark")

    bench_data = pd.DataFrame({
        "Approach": ["Full-LLM reasoning (est.)", "Our ReAct (GPT-4o-mini)", "Our Deterministic"],
        "Latency (s)": [20, 6, 0.05],
        "Cost": ["$0.05-0.10", "~$0.001", "$0 (no API)"],
    })

    fig5 = px.bar(bench_data, y="Approach", x="Latency (s)", orientation="h",
                  text=bench_data.apply(lambda r: f"{r['Latency (s)']}s · {r['Cost']}", axis=1),
                  color="Latency (s)",
                  color_continuous_scale=["#27ae60", "#e74c3c"])
    fig5.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False,
                       xaxis_title="Latency (seconds)",
                       yaxis=dict(autorange="reversed"))
    fig5.update_traces(textposition="outside")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Baseline estimated; ReAct and deterministic rows measured.")

    # ── Mode 2: ReAct ────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Ask the Analyst")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("Agent requires OPENAI_API_KEY. Deterministic analysis above works without it.")
    else:
        SUGGESTED = [
            "What should we do about Christmas week?",
            "Compare NNS with O'Hare — which is growing?",
            "Is our growth real or mode-share shift?",
            "What's the revenue impact of the RSNA convention?",
        ]

        # Suggested question buttons
        scols = st.columns(len(SUGGESTED))
        for i, q in enumerate(SUGGESTED):
            with scols[i]:
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.pending_q = q
                    st.rerun()

        # Chat messages are a mix of dicts (from chat_history) and OpenAI
        # message objects (from the live ReAct loop). Normalize access.
        def get_content(m):
            return m.get("content") if isinstance(m, dict) else m.content

        def get_tool_calls(m):
            return m.get("tool_calls") if isinstance(m, dict) else m.tool_calls

        # Init chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history (assistant messages replay tool calls + metrics)
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    tcs = msg.get("tool_calls") or []
                    if tcs:
                        st.caption("🔧 " + "  ·  ".join(tcs))
                    st.markdown(msg["content"])
                    if msg.get("latency_s") is not None:
                        st.caption(f"Latency: {msg['latency_s']:.2f}s · "
                                   f"Tokens: {msg.get('tokens', 0):,} · "
                                   f"Cost: ${msg.get('cost', 0):.4f}")
                else:
                    st.markdown(msg["content"])

        # Get input
        user_input = st.chat_input("Type your question...")
        if "pending_q" in st.session_state:
            user_input = st.session_state.pop("pending_q")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                TOOL_SCHEMAS = [
                    {"type": "function", "function": {"name": "get_forecast",
                     "description": "Get forecast vs actual for NNS (area 8), weeks 27-52 of 2025.",
                     "parameters": {"type": "object", "properties": {
                         "area": {"type": "integer"}, "year": {"type": "integer"}, "week": {"type": "integer"}},
                         "required": ["area", "year", "week"]}}},
                    {"type": "function", "function": {"name": "search_events",
                     "description": "Find holidays/conventions in a specific ISO week.",
                     "parameters": {"type": "object", "properties": {
                         "year": {"type": "integer"}, "week": {"type": "integer"}},
                         "required": ["year", "week"]}}},
                    {"type": "function", "function": {"name": "search_news",
                     "description": "Search Chicago news articles for a week.",
                     "parameters": {"type": "object", "properties": {
                         "query": {"type": "string"}, "year": {"type": "integer"},
                         "week": {"type": "integer"}, "top_k": {"type": "integer"}},
                         "required": ["query", "year", "week"]}}},
                    {"type": "function", "function": {"name": "suggest_action",
                     "description": "Generate action recommendation with revenue impact. Fetches forecast + events internally — pass only area/year/week.",
                     "parameters": {"type": "object", "properties": {
                         "area": {"type": "integer"}, "year": {"type": "integer"},
                         "week": {"type": "integer"}},
                         "required": ["area", "year", "week"]}}},
                    {"type": "function", "function": {"name": "get_market_comparison",
                     "description": "Compare taxi vs ride-hail across areas (8=NNS, 76=O'Hare, 32=Loop).",
                     "parameters": {"type": "object", "properties": {
                         "areas": {"type": "array", "items": {"type": "integer"}}},
                         "required": ["areas"]}}},
                ]
                REGISTRY = {"get_forecast": get_forecast, "search_events": search_events,
                            "search_news": search_news, "suggest_action": suggest_action,
                            "get_market_comparison": get_market_comparison}

                SYSTEM = ("You are a Chicago taxi market analyst for a 150-car company entering NNS. "
                          "Use tools to answer. Ground claims in tool output. Include revenue impact. "
                          "Keep responses under 200 words. "
                          "Convention impact weeks: RSNA Annual Meeting -> 2025-W49 (not W48). "
                          "If outside scope, say: 'This demo covers NNS taxi demand for H2 2025.'")

                from openai import OpenAI
                client = OpenAI(timeout=15)

                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user_input}]

                t0 = time.time()
                total_tokens = 0
                tool_calls_log = []
                msg = None
                agent_failed = False

                with st.status("🔍 Analyzing...", expanded=True) as status:
                    try:
                        for iteration in range(5):
                            response = client.chat.completions.create(
                                model="gpt-4o-mini", messages=messages,
                                tools=TOOL_SCHEMAS, tool_choice="auto")

                            msg = response.choices[0].message
                            total_tokens += response.usage.total_tokens

                            if not get_tool_calls(msg):
                                break

                            messages.append(msg)
                            for tc in get_tool_calls(msg):
                                fname = tc.function.name
                                args = json.loads(tc.function.arguments)
                                tool_calls_log.append(f"{fname}({json.dumps(args, default=str)[:60]})")
                                st.write(f"→ **{fname}**({json.dumps(args, default=str)[:60]})")
                                try:
                                    result = REGISTRY[fname](**args)
                                except Exception as e:
                                    result = {"error": str(e)}
                                messages.append({"role": "tool", "tool_call_id": tc.id,
                                                 "content": json.dumps(result, default=str)})

                        status.update(label="Complete", state="complete")
                    except Exception:
                        # Connection/API failure — degrade gracefully, no traceback.
                        agent_failed = True
                        status.update(label="Agent unavailable", state="error")

                elapsed = time.time() - t0
                cost = total_tokens * 0.0003 / 1000

                if agent_failed:
                    warning_msg = ("Agent unavailable — no API connection. "
                                   "Deterministic analysis above works offline.")
                    st.warning(warning_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": warning_msg,
                        "tool_calls": tool_calls_log,
                        "latency_s": None, "tokens": None, "cost": None,
                    })
                else:
                    content = get_content(msg)
                    if content:
                        if tool_calls_log:
                            st.caption("🔧 " + "  ·  ".join(tool_calls_log))
                        st.markdown(content)
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": content,
                            "tool_calls": tool_calls_log,
                            "latency_s": elapsed, "tokens": total_tokens, "cost": cost,
                        })
                    st.caption(f"Latency: {elapsed:.2f}s · Tokens: {total_tokens:,} · "
                               f"Cost: ${cost:.4f}")
