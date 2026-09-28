"""
Cross-reference daily demand anomalies against external features.
Identifies which features explain the big daily swings.

Inputs:
  data/derived/area_day.csv
  data/external/chicago_weather_daily.csv
  data/external/chicago_sports_games.csv
  data/external/holidays_chicago.csv
  data/external/conventions_chicago.csv

Usage: python scripts/1c_cross_reference.py
"""

import csv
from datetime import date, timedelta
from collections import defaultdict

THRESHOLD = 20  # percent deviation to count as anomalous

# =============================================================
# LOAD ALL DATA
# =============================================================

def load_taxi():
    rows = {}
    with open("data/derived/area_day.csv") as f:
        for r in csv.DictReader(f):
            if int(r["pickup_community_area"]) == 8:
                d = date.fromisoformat(r["date"])
                trips = int(r["trip_count"])
                if trips >= 100:
                    rows[d] = trips
    return rows

def load_weather():
    rows = {}
    with open("data/external/chicago_weather_daily.csv") as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["date"])
            rows[d] = {
                "tmax": float(r["tmax"]) if r["tmax"] else None,
                "tmin": float(r["tmin"]) if r["tmin"] else None,
                "precip_mm": float(r["precip_mm"]),
                "snow_mm": float(r["snow_mm"]),
                "rain_mm": float(r["rain_mm"]),
                "wind_gust": float(r["wind_gust_max"]),
                "extreme_cold": int(r["extreme_cold"]),
                "extreme_heat": int(r["extreme_heat"]),
                "heavy_precip": int(r["heavy_precip"]),
                "heavy_snow": int(r["heavy_snow"]),
            }
    return rows

def load_sports():
    games = defaultdict(list)
    with open("data/external/chicago_sports_games.csv") as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["date"])
            games[d].append(f"{r['team']} vs {r['opponent']}")
    return games

def load_holidays():
    events = defaultdict(list)
    with open("data/external/holidays_chicago.csv") as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["date"])
            events[d].append({
                "name": r["event_name"],
                "category": r["category"],
                "impact": r["expected_impact"],
            })
    return events

def load_conventions():
    events = []
    with open("data/external/conventions_chicago.csv") as f:
        for r in csv.DictReader(f):
            events.append({
                "start": date.fromisoformat(r["date_start"]),
                "end": date.fromisoformat(r["date_end"]),
                "name": r["event_name"],
                "attendees": int(r["est_attendees"]),
            })
    return events

def convention_on_date(conventions, d):
    """Return list of conventions active on a given date."""
    return [c for c in conventions if c["start"] <= d <= c["end"]]

# =============================================================
# COMPUTE ANOMALIES
# =============================================================

taxi = load_taxi()
weather = load_weather()
sports = load_sports()
holidays = load_holidays()
conventions = load_conventions()

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def same_dow_neighbors(d, weeks=2):
    vals = []
    for w in range(-weeks, weeks + 1):
        if w == 0:
            continue
        nd = d + timedelta(weeks=w)
        if nd in taxi:
            vals.append(taxi[nd])
    return vals

# Compute all anomalies
anomalies = []
for d in sorted(taxi.keys()):
    neighbors = same_dow_neighbors(d)
    if len(neighbors) < 2:
        continue
    avg_n = sum(neighbors) / len(neighbors)
    impact = (taxi[d] - avg_n) / avg_n * 100
    if abs(impact) >= THRESHOLD:
        # Gather all features for this day
        w = weather.get(d, {})
        s = sports.get(d, [])
        h = holidays.get(d, [])
        c = convention_on_date(conventions, d)

        # Determine weather severity
        weather_flags = []
        if w.get("heavy_snow"):
            weather_flags.append(f"heavy snow ({w['snow_mm']}mm)")
        if w.get("heavy_precip"):
            weather_flags.append(f"heavy precip ({w['precip_mm']}mm)")
        if w.get("extreme_cold"):
            weather_flags.append(f"extreme cold ({w['tmin']}°C)")
        if w.get("extreme_heat"):
            weather_flags.append(f"extreme heat ({w['tmax']}°C)")
        if w.get("wind_gust", 0) > 80:
            weather_flags.append(f"high wind ({w['wind_gust']}km/h)")
        # Moderate weather (not extreme but notable)
        if not weather_flags:
            if w.get("snow_mm", 0) > 5:
                weather_flags.append(f"snow ({w['snow_mm']}mm)")
            if w.get("precip_mm", 0) > 10:
                weather_flags.append(f"rain ({w['precip_mm']}mm)")
            if w.get("tmin") is not None and w["tmin"] < -10:
                weather_flags.append(f"cold ({w['tmin']}°C)")

        # Build explanation
        explanations = []
        if h:
            explanations.extend([f"Holiday: {e['name']}" for e in h])
        if c:
            explanations.extend([f"Convention: {e['name']} ({e['attendees']:,})" for e in c])
        if s:
            explanations.extend([f"Game: {g}" for g in s])
        if weather_flags:
            explanations.extend([f"Weather: {f}" for f in weather_flags])

        explained = len(explanations) > 0

        anomalies.append({
            "date": d,
            "day": DAY_NAMES[d.weekday()],
            "trips": taxi[d],
            "neighbor_avg": avg_n,
            "impact": impact,
            "direction": "spike" if impact > 0 else "dip",
            "explanations": explanations,
            "explained": explained,
            "has_holiday": len(h) > 0,
            "has_convention": len(c) > 0,
            "has_game": len(s) > 0,
            "has_weather": len(weather_flags) > 0,
        })

# =============================================================
# OUTPUT
# =============================================================

total = len(anomalies)
explained_count = sum(1 for a in anomalies if a["explained"])
unexplained_count = total - explained_count

print(f"Anomalous days (>{THRESHOLD}% deviation): {total}")
print(f"Explained by at least one feature: {explained_count} ({explained_count/total*100:.0f}%)")
print(f"Still unexplained: {unexplained_count} ({unexplained_count/total*100:.0f}%)")

# By feature type
has_holiday = sum(1 for a in anomalies if a["has_holiday"])
has_convention = sum(1 for a in anomalies if a["has_convention"])
has_game = sum(1 for a in anomalies if a["has_game"])
has_weather = sum(1 for a in anomalies if a["has_weather"])
print(f"\nFeature coverage:")
print(f"  Holiday/event:  {has_holiday}/{total} ({has_holiday/total*100:.0f}%)")
print(f"  Convention:     {has_convention}/{total} ({has_convention/total*100:.0f}%)")
print(f"  Sports game:    {has_game}/{total} ({has_game/total*100:.0f}%)")
print(f"  Weather:        {has_weather}/{total} ({has_weather/total*100:.0f}%)")

# =============================================================
# FEATURE IMPACT ANALYSIS
# =============================================================

print(f"\n{'='*80}")
print(f"FEATURE IMPACT ANALYSIS")
print(f"{'='*80}")

# For each feature type, what's the average impact direction?
print(f"\n--- HOLIDAYS ---")
holiday_impacts = defaultdict(list)
for a in anomalies:
    for h in holidays.get(a["date"], []):
        holiday_impacts[h["name"]].append(a["impact"])

if holiday_impacts:
    print(f"{'Event':<30} {'Count':>6} {'Avg Impact':>11} {'Consistent?'}")
    print("-" * 60)
    for name, impacts in sorted(holiday_impacts.items(), key=lambda x: abs(sum(x[1])/len(x[1])), reverse=True):
        avg = sum(impacts) / len(impacts)
        # Consistent if all same sign
        consistent = "Yes" if all(i > 0 for i in impacts) or all(i < 0 for i in impacts) else "Mixed"
        if len(impacts) == 1:
            consistent = "1 sample"
        print(f"{name:<30} {len(impacts):>6} {avg:>+10.1f}% {consistent}")

print(f"\n--- CONVENTIONS ---")
conv_impacts = defaultdict(list)
for a in anomalies:
    for c in convention_on_date(conventions, a["date"]):
        conv_impacts[c["name"]].append(a["impact"])

if conv_impacts:
    print(f"{'Convention':<40} {'Days':>5} {'Avg Impact':>11}")
    print("-" * 60)
    for name, impacts in sorted(conv_impacts.items(), key=lambda x: abs(sum(x[1])/len(x[1])), reverse=True):
        avg = sum(impacts) / len(impacts)
        print(f"{name:<40} {len(impacts):>5} {avg:>+10.1f}%")

print(f"\n--- SPORTS GAMES ---")
# Compare: anomalous days WITH games vs WITHOUT
game_days_impact = [a["impact"] for a in anomalies if a["has_game"]]
no_game_days_impact = [a["impact"] for a in anomalies if not a["has_game"]]
if game_days_impact:
    print(f"Anomalous days with home game: {len(game_days_impact)}")
    print(f"  Avg impact: {sum(game_days_impact)/len(game_days_impact):+.1f}%")
    print(f"  Positive spikes: {sum(1 for i in game_days_impact if i > 0)}")
    print(f"  Negative dips: {sum(1 for i in game_days_impact if i < 0)}")
if no_game_days_impact:
    print(f"Anomalous days without home game: {len(no_game_days_impact)}")
    print(f"  Avg impact: {sum(no_game_days_impact)/len(no_game_days_impact):+.1f}%")

# Game count vs impact
print(f"\nImpact by number of simultaneous home games:")
game_count_impact = defaultdict(list)
for d in sorted(taxi.keys()):
    neighbors = same_dow_neighbors(d)
    if len(neighbors) < 2:
        continue
    avg_n = sum(neighbors) / len(neighbors)
    impact = (taxi[d] - avg_n) / avg_n * 100
    n_games = len(sports.get(d, []))
    game_count_impact[n_games].append(impact)

for n_games in sorted(game_count_impact.keys()):
    impacts = game_count_impact[n_games]
    avg = sum(impacts) / len(impacts)
    print(f"  {n_games} games: avg impact {avg:+.1f}% ({len(impacts)} days)")

print(f"\n--- WEATHER ---")
# Impact on extreme weather days
print(f"Impact on weather-flagged anomalous days:")
weather_dips = [a for a in anomalies if a["has_weather"] and a["impact"] < 0]
weather_spikes = [a for a in anomalies if a["has_weather"] and a["impact"] > 0]
print(f"  Dips with bad weather: {len(weather_dips)}")
print(f"  Spikes with bad weather: {len(weather_spikes)} (likely coincidental)")

# Correlation: all days (not just anomalous)
print(f"\nWeather correlation with daily demand (all days, not just anomalous):")
all_impacts = []
all_precip = []
all_snow = []
all_tmin = []
for d in sorted(taxi.keys()):
    neighbors = same_dow_neighbors(d)
    if len(neighbors) < 2:
        continue
    avg_n = sum(neighbors) / len(neighbors)
    impact = (taxi[d] - avg_n) / avg_n * 100
    w = weather.get(d, {})
    if w:
        all_impacts.append(impact)
        all_precip.append(w.get("precip_mm", 0))
        all_snow.append(w.get("snow_mm", 0))
        all_tmin.append(w.get("tmin", 0) if w.get("tmin") is not None else 0)

def correlation(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    sx = (sum((xi - mx)**2 for xi in x) / n) ** 0.5
    sy = (sum((yi - my)**2 for yi in y) / n) ** 0.5
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0

if all_impacts:
    print(f"  Precip vs impact: r = {correlation(all_precip, all_impacts):+.3f}")
    print(f"  Snow vs impact:   r = {correlation(all_snow, all_impacts):+.3f}")
    print(f"  Tmin vs impact:   r = {correlation(all_tmin, all_impacts):+.3f}")

# =============================================================
# FULL TABLE: ALL ANOMALOUS DAYS WITH EXPLANATIONS
# =============================================================

print(f"\n{'='*80}")
print(f"ALL ANOMALOUS DAYS WITH CROSS-REFERENCED FEATURES")
print(f"{'='*80}")
print(f"{'Date':<12} {'Day':<5} {'Trips':>7} {'Impact':>8} {'Explanations'}")
print("-" * 90)

current_month = ""
for a in anomalies:
    month = a["date"].strftime("%Y-%m")
    if month != current_month:
        if current_month:
            print()
        current_month = month

    expl = "; ".join(a["explanations"]) if a["explanations"] else "???"
    # Truncate long explanations
    if len(expl) > 55:
        expl = expl[:52] + "..."
    print(f"{str(a['date']):<12} {a['day']:<5} {a['trips']:>7,} {a['impact']:>+7.1f}%  {expl}")

# =============================================================
# STILL UNEXPLAINED — INVESTIGATION NEEDED
# =============================================================

print(f"\n{'='*80}")
print(f"STILL UNEXPLAINED (>{THRESHOLD}% deviation, no matching features)")
print(f"{'='*80}")
unexplained = [a for a in anomalies if not a["explained"]]
print(f"{'Date':<12} {'Day':<5} {'Trips':>7} {'Impact':>8}")
print("-" * 38)
for a in sorted(unexplained, key=lambda x: abs(x["impact"]), reverse=True):
    print(f"{str(a['date']):<12} {a['day']:<5} {a['trips']:>7,} {a['impact']:>+7.1f}%")

print(f"\nTotal unexplained: {len(unexplained)}/{total}")
