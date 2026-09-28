"""
Download daily Chicago weather for 2024-2026.
Uses Open-Meteo free API (no API key needed).
Output: data/external/chicago_weather_daily.csv

Usage: python scripts/0_get_weather_daily.py
"""

import json
import csv
import urllib.request
from datetime import date

LAT, LON = 41.98, -87.91  # O'Hare
START = "2024-01-01"
END = "2026-06-22"

VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
    "wind_gusts_10m_max",
    "rain_sum",
]

def main():
    params = (
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={START}&end_date={END}"
        f"&daily={','.join(VARIABLES)}"
        f"&temperature_unit=celsius"
        f"&timezone=America/Chicago"
    )
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
    print(f"Fetching daily weather...")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    daily = data["daily"]
    dates = daily["time"]

    rows = []
    for i, d_str in enumerate(dates):
        d = date.fromisoformat(d_str)
        iso = d.isocalendar()
        tmax = daily["temperature_2m_max"][i]
        tmin = daily["temperature_2m_min"][i]
        precip = daily["precipitation_sum"][i] or 0
        snow = daily["snowfall_sum"][i] or 0
        rain = daily["rain_sum"][i] or 0
        wind = daily["wind_gusts_10m_max"][i] or 0

        rows.append({
            "date": d_str,
            "iso_year": iso[0],
            "iso_week": iso[1],
            "weekday": d.weekday(),
            "tmax": round(tmax, 1) if tmax is not None else "",
            "tmin": round(tmin, 1) if tmin is not None else "",
            "precip_mm": round(precip, 1),
            "rain_mm": round(rain, 1),
            "snow_mm": round(snow, 1),
            "wind_gust_max": round(wind, 1),
            "extreme_cold": 1 if tmin is not None and tmin < -15 else 0,
            "extreme_heat": 1 if tmax is not None and tmax > 35 else 0,
            "heavy_precip": 1 if precip > 20 else 0,
            "heavy_snow": 1 if snow > 10 else 0,
        })

    outfile = "data/external/chicago_weather_daily.csv"
    fields = list(rows[0].keys())
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} days to {outfile}")

    # Quick stats
    heavy_days = sum(1 for r in rows if r["heavy_precip"])
    snow_days = sum(1 for r in rows if r["heavy_snow"])
    cold_days = sum(1 for r in rows if r["extreme_cold"])
    print(f"Heavy precip days (>20mm): {heavy_days}")
    print(f"Heavy snow days (>10mm): {snow_days}")
    print(f"Extreme cold days (<-15C): {cold_days}")


if __name__ == "__main__":
    main()
