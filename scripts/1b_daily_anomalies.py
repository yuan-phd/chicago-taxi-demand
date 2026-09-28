"""
List all NNS daily anomalies (>15% deviation from same-day-of-week neighbors).
Auto-tags known US holidays. Remaining unknowns are investigation targets.

Usage: python scripts/1b_daily_anomalies.py
Input:  data/derived/area_day.csv
"""

import csv
from datetime import date, timedelta
from collections import defaultdict

# =============================================================
# KNOWN HOLIDAYS / EVENTS (auto-tag these)
# =============================================================

def build_known_events():
    """Build dict of date → event label for known holidays."""
    events = {}

    # US Federal / Major Holidays
    fixed = [
        (date(2024,1,1), "New Year's Day"),
        (date(2025,1,1), "New Year's Day"),
        (date(2026,1,1), "New Year's Day"),
        (date(2024,1,15), "MLK Day"),
        (date(2025,1,20), "MLK Day"),
        (date(2026,1,19), "MLK Day"),
        (date(2024,2,14), "Valentine's Day"),
        (date(2025,2,14), "Valentine's Day"),
        (date(2026,2,14), "Valentine's Day"),
        (date(2024,2,19), "Presidents Day"),
        (date(2025,2,17), "Presidents Day"),
        (date(2026,2,16), "Presidents Day"),
        (date(2024,5,27), "Memorial Day"),
        (date(2025,5,26), "Memorial Day"),
        (date(2024,6,19), "Juneteenth"),
        (date(2025,6,19), "Juneteenth"),
        (date(2024,7,4), "July 4th"),
        (date(2025,7,4), "July 4th"),
        (date(2024,9,2), "Labor Day"),
        (date(2025,9,1), "Labor Day"),
        (date(2024,11,11), "Veterans Day"),
        (date(2025,11,11), "Veterans Day"),
        (date(2024,11,28), "Thanksgiving"),
        (date(2024,11,29), "Black Friday"),
        (date(2024,11,30), "Thanksgiving wknd"),
        (date(2025,11,27), "Thanksgiving"),
        (date(2025,11,28), "Black Friday"),
        (date(2025,11,29), "Thanksgiving wknd"),
        (date(2024,12,24), "Christmas Eve"),
        (date(2024,12,25), "Christmas"),
        (date(2024,12,26), "Day after Christmas"),
        (date(2025,12,24), "Christmas Eve"),
        (date(2025,12,25), "Christmas"),
        (date(2025,12,26), "Day after Christmas"),
        (date(2024,12,31), "New Year's Eve"),
        (date(2025,12,31), "New Year's Eve"),
    ]
    for d, label in fixed:
        events[d] = label

    # Day after major holidays (often affected)
    day_after = [
        (date(2024,1,2), "Day after NY"),
        (date(2025,1,2), "Day after NY"),
        (date(2026,1,2), "Day after NY"),
        (date(2024,7,5), "Day after July 4th"),
        (date(2025,7,5), "Day after July 4th"),
    ]
    for d, label in day_after:
        events[d] = label

    # St Patrick's Day (huge in Chicago)
    # Chicago parade is usually the Saturday before March 17
    events[date(2024,3,16)] = "St Patrick's (parade)"
    events[date(2024,3,17)] = "St Patrick's Day"
    events[date(2025,3,15)] = "St Patrick's (parade)"
    events[date(2025,3,17)] = "St Patrick's Day"
    events[date(2026,3,14)] = "St Patrick's (parade)"
    events[date(2026,3,16)] = "St Patrick's (near parade)"
    events[date(2026,3,17)] = "St Patrick's Day"

    # Easter
    events[date(2024,3,29)] = "Good Friday"
    events[date(2024,3,31)] = "Easter Sunday"
    events[date(2025,4,18)] = "Good Friday"
    events[date(2025,4,20)] = "Easter Sunday"

    # Jewish holidays
    # Rosh Hashanah
    for d in [date(2024,10,2), date(2024,10,3), date(2024,10,4)]:
        events[d] = "Rosh Hashanah"
    for d in [date(2025,9,22), date(2025,9,23), date(2025,9,24)]:
        events[d] = "Rosh Hashanah"
    # Yom Kippur
    events[date(2024,10,11)] = "Yom Kippur"
    events[date(2024,10,12)] = "Yom Kippur"
    events[date(2025,10,1)] = "Yom Kippur"
    events[date(2025,10,2)] = "Yom Kippur"

    # Halloween
    events[date(2024,10,31)] = "Halloween"
    events[date(2025,10,31)] = "Halloween"

    # Chinese New Year (Chicago has a large celebration)
    events[date(2024,2,10)] = "Chinese New Year"
    events[date(2025,1,29)] = "Chinese New Year"
    events[date(2026,2,17)] = "Chinese New Year"

    # Lollapalooza (Grant Park, usually late July/early Aug)
    for d in [date(2024,8,1), date(2024,8,2), date(2024,8,3), date(2024,8,4)]:
        events[d] = "Lollapalooza"
    for d in [date(2025,7,31), date(2025,8,1), date(2025,8,2), date(2025,8,3)]:
        events[d] = "Lollapalooza"

    # Chicago Marathon (October)
    events[date(2024,10,13)] = "Chicago Marathon"
    events[date(2025,10,12)] = "Chicago Marathon"

    # Super Bowl Sunday (not in Chicago but affects viewing/going out)
    events[date(2024,2,11)] = "Super Bowl Sunday"
    events[date(2025,2,9)] = "Super Bowl Sunday"
    events[date(2026,2,8)] = "Super Bowl Sunday"

    return events

# =============================================================
# LOAD DATA
# =============================================================

daily = []
with open("data/derived/area_day.csv") as f:
    for r in csv.DictReader(f):
        if int(r["pickup_community_area"]) == 8:
            d = date.fromisoformat(r["date"])
            trips = int(r["trip_count"])
            if trips < 100: continue
            daily.append({"date": d, "trips": trips,
                         "weekday": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d.weekday()]})
daily.sort(key=lambda x: x["date"])

date_lookup = {r["date"]: r["trips"] for r in daily}
known_events = build_known_events()

# =============================================================
# COMPUTE ALL ANOMALIES
# =============================================================

def same_dow_neighbors(d, weeks=2):
    vals = []
    for w in range(-weeks, weeks+1):
        if w == 0: continue
        nd = d + timedelta(weeks=w)
        if nd in date_lookup:
            vals.append(date_lookup[nd])
    return vals

THRESHOLD = 15  # percent

anomalies = []
for r in daily:
    neighbors = same_dow_neighbors(r["date"])
    if len(neighbors) < 2: continue
    avg_n = sum(neighbors) / len(neighbors)
    impact = (r["trips"] - avg_n) / avg_n * 100
    if abs(impact) >= THRESHOLD:
        event = known_events.get(r["date"], "")
        # Also check ±1 day for event proximity
        if not event:
            for offset in [-1, 1]:
                nearby = known_events.get(r["date"] + timedelta(days=offset))
                if nearby:
                    event = f"Near: {nearby}"
                    break
        anomalies.append({
            "date": r["date"],
            "day": r["weekday"],
            "trips": r["trips"],
            "neighbor_avg": avg_n,
            "impact": impact,
            "event": event,
            "explained": bool(event),
        })

anomalies.sort(key=lambda x: x["date"])

# =============================================================
# OUTPUT: ALL ANOMALIES, CHRONOLOGICAL
# =============================================================

print(f"Total days analyzed: {len(daily)}")
print(f"Anomalous days (>{THRESHOLD}% deviation): {len(anomalies)}")

explained = [a for a in anomalies if a["explained"]]
unexplained = [a for a in anomalies if not a["explained"]]
print(f"  Explained (known events): {len(explained)}")
print(f"  UNEXPLAINED: {len(unexplained)}")

# Positive vs negative
pos = [a for a in anomalies if a["impact"] > 0]
neg = [a for a in anomalies if a["impact"] < 0]
print(f"  Positive spikes: {len(pos)}")
print(f"  Negative dips: {len(neg)}")

# === UNEXPLAINED ANOMALIES (investigation targets) ===
print(f"\n{'='*80}")
print(f"UNEXPLAINED ANOMALIES — INVESTIGATION TARGETS")
print(f"{'='*80}")
print(f"{'Date':<12} {'Day':<5} {'Trips':>7} {'NbrAvg':>8} {'Impact':>8}")
print("-" * 45)
for a in sorted(unexplained, key=lambda x: abs(x["impact"]), reverse=True):
    print(f"{str(a['date']):<12} {a['day']:<5} {a['trips']:>7,} "
          f"{a['neighbor_avg']:>8,.0f} {a['impact']:>+7.1f}%")

# === ALL ANOMALIES CHRONOLOGICAL ===
print(f"\n{'='*80}")
print(f"ALL ANOMALIES (chronological, >{THRESHOLD}% deviation)")
print(f"{'='*80}")
print(f"{'Date':<12} {'Day':<5} {'Trips':>7} {'NbrAvg':>8} {'Impact':>8} {'Event'}")
print("-" * 80)

current_month = ""
for a in anomalies:
    month = a["date"].strftime("%Y-%m")
    if month != current_month:
        if current_month:
            print()
        current_month = month
    tag = a["event"] if a["event"] else "???"
    print(f"{str(a['date']):<12} {a['day']:<5} {a['trips']:>7,} "
          f"{a['neighbor_avg']:>8,.0f} {a['impact']:>+7.1f}%  {tag}")

# === SUMMARY: RECURRING PATTERNS ===
print(f"\n{'='*80}")
print(f"RECURRING PATTERNS (events appearing in multiple years)")
print(f"{'='*80}")

# Group by event name
event_impacts = defaultdict(list)
for a in anomalies:
    if a["event"] and "Near:" not in a["event"]:
        event_impacts[a["event"]].append(a["impact"])

print(f"\n{'Event':<25} {'Count':>6} {'Avg Impact':>11} {'Range'}")
print("-" * 60)
for event, impacts in sorted(event_impacts.items(), key=lambda x: abs(sum(x[1])/len(x[1])), reverse=True):
    avg = sum(impacts) / len(impacts)
    rng = f"{min(impacts):+.0f}% to {max(impacts):+.0f}%"
    print(f"{event:<25} {len(impacts):>6} {avg:>+10.1f}% {rng}")

# === UNEXPLAINED CLUSTERS ===
print(f"\n{'='*80}")
print(f"UNEXPLAINED CLUSTERS (consecutive unexplained anomalies)")
print(f"{'='*80}")

clusters = []
current_cluster = []
for a in sorted(unexplained, key=lambda x: x["date"]):
    if current_cluster and (a["date"] - current_cluster[-1]["date"]).days <= 2:
        current_cluster.append(a)
    else:
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        current_cluster = [a]
if len(current_cluster) >= 2:
    clusters.append(current_cluster)

if clusters:
    for cluster in clusters:
        dates = f"{cluster[0]['date']} to {cluster[-1]['date']}"
        avg_impact = sum(a['impact'] for a in cluster) / len(cluster)
        print(f"\n  {dates} ({len(cluster)} days, avg impact {avg_impact:+.1f}%):")
        for a in cluster:
            print(f"    {a['date']} {a['day']}: {a['trips']:,} trips ({a['impact']:+.1f}%)")
else:
    print("  No clusters found.")
