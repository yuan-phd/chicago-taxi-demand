"""
Model Comparison: gpt-4o-mini vs gpt-5.5 on ReAct agent
=========================================================
Tests whether model choice affects latency/cost when all
reasoning is in deterministic tools.

Run from project root:
    python scripts/8_model_comparison.py
"""

import json
import re
import time
import random
import os
import pandas as pd
from pathlib import Path
from datetime import timedelta, date
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
Path("outputs").mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS + DATA
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
        "corridor": "NNS→Loop", "corridor_share_of_nns": "20%",
        "consecutive_weeks_above": 4,
        "company_corridor_trips_week": 282, "breakeven_trips_week": 840,
        "corridor_fare": 8.42, "corridor_seconds": 544, "corridor_revenue_per_hr": 55.7,
        "internal_fare": 8.85, "internal_seconds": 367, "internal_revenue_per_hr": 86.8,
    },
}

backtest = pd.read_csv("data/from_assignment/backtest_8.csv")
holidays_df = pd.read_csv("data/external/holidays_chicago.csv")
holidays_df["date"] = pd.to_datetime(holidays_df["date"])
conventions_df = pd.read_csv("data/external/conventions_chicago.csv")
conventions_df["date_start"] = pd.to_datetime(conventions_df["date_start"])
conventions_df["date_end"] = pd.to_datetime(conventions_df["date_end"])
taxi_weekly = pd.read_csv("data/from_assignment/area_week.csv")
tnp_weekly = pd.read_csv("data/external/tnp_area_week.csv")

with open("config/action_config.json") as f:
    _config = json.load(f)
COMPANY = _config["company_profile"]
ACTION_CONFIG = _config["action_types"]

# ══════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def get_forecast(area: int, year: int, week: int) -> dict:
    if area != 8:
        return {"error": "Only area 8 (NNS) in this demo."}
    row = backtest[(backtest["iso_year"] == year) & (backtest["iso_week"] == week)]
    if len(row) == 0:
        return {"error": f"No data for {year} W{week}."}
    r = row.iloc[0]
    actual, predicted = int(r["actual"]), int(r["trend_seasonal"])
    error_pct = round((actual - predicted) / actual * 100, 1)
    return {"area": area, "area_name": "Near North Side", "year": year, "week": week,
            "actual_trips": actual, "forecast_trips": predicted, "error_pct": error_pct,
            "direction": "under-forecast" if error_pct > 0 else "over-forecast",
            "is_anomaly": abs(error_pct) > 10}

def search_events(year: int, week: int) -> dict:
    jan4 = date(year, 1, 4)
    w1_start = jan4 - timedelta(days=jan4.isoweekday() - 1)
    ws = pd.Timestamp(w1_start + timedelta(weeks=week - 1))
    we = ws + pd.Timedelta(days=6)
    events = []
    for _, h in holidays_df[(holidays_df["date"] >= ws) & (holidays_df["date"] <= we)].iterrows():
        f = HOLIDAY_FACTORS.get(h["event_name"])
        events.append({"type": "holiday", "name": h["event_name"],
                       "date": h["date"].strftime("%Y-%m-%d"),
                       "factor": f, "impact_pct": f"{(f-1)*100:+.0f}%" if f else "unknown"})
    for _, c in conventions_df.iterrows():
        if c["date_start"] <= we and c["date_end"] >= ws:
            f = CONVENTION_FACTORS.get(c["event_name"])
            od = (min(c["date_end"], we) - max(c["date_start"], ws)).days + 1
            events.append({"type": "convention", "name": c["event_name"],
                           "overlap_days": od, "est_attendees": int(c["est_attendees"]),
                           "factor": f, "impact_pct": f"{(f-1)*100:+.0f}%" if f else "unknown"})
    return {"year": year, "week": week, "events": events, "n_events": len(events)}

def suggest_action(area: int, year: int, week: int,
                   forecast_data: dict, event_data: dict) -> dict:
    C = COMPANY
    CFG = ACTION_CONFIG
    error_pct = forecast_data.get("error_pct", 0)
    events = event_data.get("events", [])
    holiday_events = [e for e in events if e["type"] == "holiday" and e.get("factor")]
    convention_events = [e for e in events if e["type"] == "convention" and e.get("factor")]
    neg_holidays = [h for h in holiday_events if h["factor"] < CFG["RESOURCE_OPTIMIZE"]["holiday_threshold"]]
    pos_conventions = [c for c in convention_events
                       if c["factor"] > CFG["DEMAND_SURGE"]["surge_threshold"]
                       and c.get("est_attendees", 0) >= CFG["DEMAND_SURGE"]["min_attendees"]]
    has_validated_cause = bool(neg_holidays or pos_conventions)

    gt = GROWTH_THRESHOLD_WEEKS.get((year, week))
    if gt:
        return {"action_type": "GROWTH_THRESHOLD",
                "headline": f"{gt['corridor']} — expansion NOT recommended",
                "revenue_impact": {"expansion_rejected": f"{gt['company_corridor_trips_week']} trips/wk ≪ {gt['breakeven_trips_week']} breakeven"},
                "requires_approval": False}

    cs = COMPETITIVE_SHIFT_WEEKS.get((year, week))
    if cs:
        wr = cs["extra_trips_per_week"] * C["avg_fare"]
        return {"action_type": "COMPETITIVE_SHIFT", "headline": "Capture ride-hail surge hours",
                "revenue_impact": {"weekly_gain": f"${wr:,.0f}/week", "annual_gain": f"${wr*52:,.0f}/year"},
                "requires_approval": True}

    if abs(error_pct) > CFG["ANOMALY_ESCALATION"]["error_threshold_pct"] and not has_validated_cause:
        return {"action_type": "ANOMALY_ESCALATION",
                "headline": f"Forecast deviation {error_pct:+.1f}% — no validated cause",
                "revenue_impact": "Unknown — requires investigation", "requires_approval": False}

    if neg_holidays:
        strongest = min(neg_holidays, key=lambda h: h["factor"])
        factor = strongest["factor"]
        reduction_pct = round((1 - factor) * 100)
        cars_to_park = round(C["fleet_size"] * (1 - factor))
        cars_active = C["fleet_size"] - cars_to_park
        daily_saving = cars_to_park * C["cost_per_car_day"]

        if factor < CFG["DEMAND_DROP"]["extreme_threshold"]:
            return {"action_type": "DEMAND_DROP",
                    "headline": f"Park {cars_to_park} of {C['fleet_size']} cars — {strongest['name']} reduces demand {reduction_pct}%",
                    "what_to_do": [f"Reduce fleet from {C['fleet_size']} to {cars_active} cars",
                                   f"Park {cars_to_park} cars for the holiday period"],
                    "revenue_impact": {"daily_saving": f"${daily_saving:,.0f}",
                                       "calculation": f"{cars_to_park} parked × ${C['cost_per_car_day']}/car/day"},
                    "requires_approval": True}
        else:
            redirect_cars = round(cars_to_park * CFG["RESOURCE_OPTIMIZE"]["airport_redirect_pct"])
            trips = CFG["RESOURCE_OPTIMIZE"]["airport_trips_per_car_day"]
            ar = redirect_cars * trips * C["avg_airport_fare"]
            idle_saving = (cars_to_park - redirect_cars) * C["cost_per_car_day"]
            return {"action_type": "RESOURCE_OPTIMIZE",
                    "headline": f"Redirect {redirect_cars} idle cars to airport (${C['avg_airport_fare']} fare)",
                    "revenue_impact": {"airport_revenue": f"${ar:,.0f}/day",
                                       "idle_saving": f"${idle_saving:,.0f}/day",
                                       "total_daily": f"${ar+idle_saving:,.0f}/day"},
                    "requires_approval": True}

    if pos_conventions:
        s = max(pos_conventions, key=lambda c: c["factor"])
        extra = round(C["fleet_size"] * (s["factor"] - 1))
        trips = CFG["DEMAND_SURGE"]["trips_per_temp_car_day"]
        rev = extra * trips * C["avg_fare"]
        cost = extra * C["cost_per_car_day"]
        if rev > cost:
            return {"action_type": "DEMAND_SURGE", "headline": f"Add {extra} temp cars",
                    "revenue_impact": {"daily_net": f"${rev-cost:,.0f}/day"}, "requires_approval": True}
        else:
            return {"action_type": "DEMAND_SURGE",
                    "headline": "Maximize existing fleet — adding cars not profitable",
                    "revenue_impact": {"loss_avoided": f"${(cost-rev)*s.get('overlap_days',3):,.0f}"},
                    "requires_approval": False}

    return {"action_type": "NONE", "headline": "Normal operations", "requires_approval": False}

def get_market_comparison(areas: list) -> dict:
    merged = taxi_weekly[["pickup_community_area","iso_year","iso_week","trip_count"]].rename(
        columns={"trip_count":"taxi"}).merge(
        tnp_weekly.rename(columns={"trip_count":"tnp"}),
        on=["pickup_community_area","iso_year","iso_week"], how="inner")
    results = []
    for a in areas:
        ad = merged[merged["pickup_community_area"]==a]
        y1,y2 = ad[ad["iso_year"]==2024], ad[ad["iso_year"]==2025]
        ow = set(y1["iso_week"]) & set(y2["iso_week"])
        if not ow: continue
        y1o,y2o = y1[y1["iso_week"].isin(ow)], y2[y2["iso_week"].isin(ow)]
        tg = (y2o["taxi"].sum()/y1o["taxi"].sum()-1)*100
        ng = (y2o["tnp"].sum()/y1o["tnp"].sum()-1)*100
        results.append({"area": a, "name": AREA_NAMES.get(a,f"Area {a}"),
                        "taxi_growth": f"{tg:+.1f}%", "ridehail_growth": f"{ng:+.1f}%"})
    return {"comparison": results}

# ══════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS + REGISTRY
# ══════════════════════════════════════════════════════════════════════
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
    {"type": "function", "function": {"name": "suggest_action",
     "description": "Generate revenue-connected action recommendation based on forecast + events. Returns action type, what to do, and revenue impact.",
     "parameters": {"type": "object", "properties": {
         "area": {"type": "integer"}, "year": {"type": "integer"}, "week": {"type": "integer"},
         "forecast_data": {"type": "object"}, "event_data": {"type": "object"}},
         "required": ["area", "year", "week", "forecast_data", "event_data"]}}},
    {"type": "function", "function": {"name": "get_market_comparison",
     "description": "Compare taxi vs ride-hail across areas (8=NNS, 76=O'Hare).",
     "parameters": {"type": "object", "properties": {
         "areas": {"type": "array", "items": {"type": "integer"}}},
         "required": ["areas"]}}},
]

REGISTRY = {"get_forecast": get_forecast, "search_events": search_events,
            "suggest_action": suggest_action, "get_market_comparison": get_market_comparison}

SYSTEM = ("You are a Chicago taxi market analyst for a 150-car company entering NNS. "
          "Use tools to answer. Ground claims in tool output. Include revenue impact. "
          "Keep responses under 200 words.")

QUERIES = [
    ("christmas", "What should we do about Christmas week in NNS? How much money is at stake?"),
    ("comparison", "Compare the market dynamics between NNS and O'Hare. Which area is actually growing?"),
]

EXPECTED = {
    "christmas": ["112", "22,400"],
    "comparison": ["12.9", "-5.2"],
}

# Decline cues that let a signed-negative expected token (e.g. "-5.2") match a
# bare number the model phrased in words ("down by 5.2%", "5.2% decrease").
_NEG_CUES = r"(?:declin|decreas|down|drop|-|−)"

def token_matches(text, token):
    """Substring match, but tolerant of worded negatives. A token like "-5.2"
    also matches a bare "5.2" when a decline cue sits within a few tokens on
    either side — avoids flagging a semantically-correct answer as wrong."""
    if token in text:
        return True
    m = re.fullmatch(r"-([\d.,]+)", token)
    if m:
        numb = rf"(?<!\d){re.escape(m.group(1))}(?!\d)"
        gap = r".{0,25}?"  # a few tokens
        pat = rf"{_NEG_CUES}{gap}{numb}|{numb}{gap}{_NEG_CUES}"
        return re.search(pat, text, re.IGNORECASE) is not None
    return False

MODELS = ["gpt-4o-mini", "gpt-5.5"]
RUNS_PER = 5

# ══════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════════
print("Pre-flight checks...")
client = OpenAI(timeout=60)

# Check model names exist
for m in MODELS:
    try:
        client.models.retrieve(m)
        print(f"  ✓ {m} — available")
    except Exception as e:
        print(f"  ✗ {m} — {e}")
        print("Aborting. Fix model name.")
        exit(1)

# Single trial of gpt-5.5 to check API compatibility (only when it's under test)
if "gpt-5.5" in MODELS:
    print(f"\n  Testing gpt-5.5 single call with tools...")
    try:
        test_resp = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "Say hello"}],
            tools=TOOL_SCHEMAS[:1],
            tool_choice="auto",
        )
        print(f"  ✓ gpt-5.5 tool calling works. Tokens: {test_resp.usage.total_tokens}")
    except Exception as e:
        print(f"  ✗ gpt-5.5 tool calling failed: {e}")
        print("  Check if gpt-5.5 needs different API params (e.g. Responses API).")
        exit(1)

print("\nPre-flight passed.\n")

# ══════════════════════════════════════════════════════════════════════
# TEST MATRIX
# ══════════════════════════════════════════════════════════════════════
trials = []
for model in MODELS:
    for q_key, q_text in QUERIES:
        for run_idx in range(RUNS_PER):
            trials.append({"model": model, "q_key": q_key, "q_text": q_text, "run": run_idx})

random.seed(42)
random.shuffle(trials)

print(f"Test matrix: {len(trials)} calls ({len(MODELS)} models × {len(QUERIES)} queries × {RUNS_PER} runs)")
print()

# ══════════════════════════════════════════════════════════════════════
# RUN TRIALS
# ══════════════════════════════════════════════════════════════════════
results = []

for i, trial in enumerate(trials):
    model = trial["model"]
    q_key = trial["q_key"]
    q_text = trial["q_text"]

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": q_text}]

    t0 = time.time()
    total_tokens = 0
    n_tool_calls = 0
    final_content = ""
    exhausted = False

    try:
        for iteration in range(5):
            response = client.chat.completions.create(
                model=model, messages=messages,
                tools=TOOL_SCHEMAS, tool_choice="auto",
            )
            msg = response.choices[0].message
            total_tokens += response.usage.total_tokens

            if not msg.tool_calls:
                final_content = msg.content or ""
                break

            messages.append(msg)
            for tc in msg.tool_calls:
                n_tool_calls += 1
                fname = tc.function.name
                args = json.loads(tc.function.arguments)
                result = REGISTRY[fname](**args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, default=str)})
        else:
            exhausted = True

        elapsed = time.time() - t0
        correct = all(token_matches(final_content, exp) for exp in EXPECTED[q_key])

        results.append({
            "model": model, "query": q_key, "run": trial["run"],
            "latency_s": round(elapsed, 2),
            "total_tokens": total_tokens,
            "n_tool_calls": n_tool_calls,
            "correct": correct,
            "exhausted": exhausted,
            "final_content": final_content[:500],
        })

        status = "✓" if correct else ("⚠ exhausted" if exhausted else "✗")
        print(f"  [{i+1}/{len(trials)}] {model:<15} {q_key:<12} "
              f"{elapsed:.2f}s  {total_tokens:>5} tok  {n_tool_calls} calls  {status}")

    except Exception as e:
        print(f"  [{i+1}/{len(trials)}] {model:<15} {q_key:<12} ERROR: {e}")
        results.append({
            "model": model, "query": q_key, "run": trial["run"],
            "latency_s": None, "total_tokens": None,
            "n_tool_calls": None, "correct": False, "exhausted": False,
            "error": str(e),
        })

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

df = pd.DataFrame(results)
df_valid = df.dropna(subset=["latency_s"])

print(f"\n{'Model':<15} {'Query':<12} {'Med Lat':>8} {'Min':>6} {'Max':>6} "
      f"{'Med Tok':>8} {'Correct':>8}")
print("-" * 67)

for model in MODELS:
    for q_key, _ in QUERIES:
        sub = df_valid[(df_valid["model"] == model) & (df_valid["query"] == q_key)]
        if len(sub) == 0:
            continue
        print(f"{model:<15} {q_key:<12} {sub['latency_s'].median():>7.2f}s "
              f"{sub['latency_s'].min():>5.2f}s {sub['latency_s'].max():>5.2f}s "
              f"{sub['total_tokens'].median():>7.0f} {sub['correct'].sum()}/{len(sub)}")

print(f"\n{'Model':<15} {'Med Latency':>12} {'Med Tokens':>12} {'Correct':>8}")
print("-" * 50)
for model in MODELS:
    sub = df_valid[df_valid["model"] == model]
    print(f"{model:<15} {sub['latency_s'].median():>11.2f}s {sub['total_tokens'].median():>11.0f} "
          f"{sub['correct'].sum()}/{len(sub)}")

with open("outputs/phase7_model_comparison.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSaved: outputs/phase7_model_comparison.json")
print("Note: GPT-5.5 pricing not hardcoded. Compute cost manually from token counts.")
print("Done.")
