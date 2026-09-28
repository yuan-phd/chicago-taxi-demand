"""
Download daily Chicago weather and aggregate to ISO weeks.
Uses Open-Meteo free API (no API key needed).
Output: chicago_weather_weekly.csv — ready to merge with area_week.csv

Usage: python get_weather.py
"""

import json
import urllib.request
import csv
from datetime import date, timedelta

# Chicago O'Hare coordinates
LAT, LON = 41.98, -87.91

# Date range matching area_week.csv
START = "2024-01-01"
END = "2026-06-22"  # through W22 2026

# Variables: daily max/min temp, precipitation, snowfall, max wind gusts
VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
    "wind_gusts_10m_max",
]

def fetch_weather():
    """Fetch daily weather from Open-Meteo archive API."""
    params = (
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={START}&end_date={END}"
        f"&daily={','.join(VARIABLES)}"
        f"&temperature_unit=celsius"
        f"&timezone=America/Chicago"
    )
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
    print(f"Fetching: {url[:80]}...")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    daily = data["daily"]
    dates = [date.fromisoformat(d) for d in daily["time"]]

    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": d,
            "iso_year": d.isocalendar()[0],
            "iso_week": d.isocalendar()[1],
            "tmax": daily["temperature_2m_max"][i],
            "tmin": daily["temperature_2m_min"][i],
            "precip_mm": daily["precipitation_sum"][i],
            "snow_mm": daily["snowfall_sum"][i],
            "wind_gust_max": daily["wind_gusts_10m_max"][i],
        })
    print(f"  Got {len(rows)} daily records")
    return rows


def aggregate_weekly(daily_rows):
    """Aggregate daily weather to ISO week level."""
    weeks = {}
    for r in daily_rows:
        key = (r["iso_year"], r["iso_week"])
        if key not in weeks:
            weeks[key] = {
                "tmax_vals": [], "tmin_vals": [],
                "precip_vals": [], "snow_vals": [],
                "wind_vals": [], "days": 0,
            }
        w = weeks[key]
        w["days"] += 1
        if r["tmax"] is not None: w["tmax_vals"].append(r["tmax"])
        if r["tmin"] is not None: w["tmin_vals"].append(r["tmin"])
        if r["precip_mm"] is not None: w["precip_vals"].append(r["precip_mm"])
        if r["snow_mm"] is not None: w["snow_vals"].append(r["snow_mm"])
        if r["wind_gust_max"] is not None: w["wind_vals"].append(r["wind_gust_max"])

    def safe_mean(vals):
        return sum(vals) / len(vals) if vals else None

    def safe_max(vals):
        return max(vals) if vals else None

    out = []
    for (yr, wk), w in sorted(weeks.items()):
        if w["days"] < 5:  # skip partial weeks
            continue
        out.append({
            "iso_year": yr,
            "iso_week": wk,
            "avg_tmax": round(safe_mean(w["tmax_vals"]), 1) if w["tmax_vals"] else "",
            "avg_tmin": round(safe_mean(w["tmin_vals"]), 1) if w["tmin_vals"] else "",
            "total_precip_mm": round(sum(w["precip_vals"]), 1) if w["precip_vals"] else "",
            "total_snow_mm": round(sum(w["snow_vals"]), 1) if w["snow_vals"] else "",
            "max_wind_gust": round(safe_max(w["wind_vals"]), 1) if w["wind_vals"] else "",
            "precip_days": sum(1 for p in w["precip_vals"] if p > 1.0),
            "snow_days": sum(1 for s in w["snow_vals"] if s > 0),
            "extreme_cold": 1 if w["tmin_vals"] and min(w["tmin_vals"]) < -15 else 0,
            "extreme_heat": 1 if w["tmax_vals"] and max(w["tmax_vals"]) > 35 else 0,
        })
    return out


def main():
    daily = fetch_weather()
    weekly = aggregate_weekly(daily)

    outfile = "chicago_weather_weekly.csv"
    fields = list(weekly[0].keys())
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(weekly)

    print(f"  Wrote {len(weekly)} weeks to {outfile}")
    print(f"  Columns: {', '.join(fields)}")


if __name__ == "__main__":
    main()
