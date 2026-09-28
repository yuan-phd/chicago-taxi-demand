"""
Aggregate raw Chicago taxi trip CSVs to corridor-level daily trip counts.
Outputs: data/derived/corridor_day.csv

Corridors: top 4 origin areas × their top destinations, plus reverse corridors.
Origins: NNS (8), Loop (32), O'Hare (76), Near West Side (28).

Usage: python scripts/0_aggregate_corridors.py <csv_files...>

Example:
  python scripts/0_aggregate_corridors.py \
    data/from_assignment/raw/taxi_2024_h1.csv \
    data/from_assignment/raw/taxi_2024_h1b.csv \
    ...
"""

import csv
import sys
from datetime import date
from collections import defaultdict

# Column name aliases (Portal CSV → SODA API format)
COLUMN_MAP = {
    "Trip Start Timestamp": "trip_start_timestamp",
    "Pickup Community Area": "pickup_community_area",
    "Dropoff Community Area": "dropoff_community_area",
    "Trip ID": "trip_id",
    "Fare": "fare",
    "Trip Miles": "trip_miles",
    "Trip Seconds": "trip_seconds",
}

# Origin areas of interest
ORIGINS = {8, 32, 76, 28}


def detect_format(header):
    if "trip_start_timestamp" in header:
        return "soda"
    elif "Trip Start Timestamp" in header:
        return "portal"
    return None


def normalize_header(header):
    return [COLUMN_MAP.get(col, col) for col in header]


def parse_date(timestamp_str, fmt):
    try:
        if fmt == "soda":
            return timestamp_str[:10]
        else:
            parts = timestamp_str.split(" ")[0]
            month, day, year = parts.split("/")
            return f"{year}-{int(month):02d}-{int(day):02d}"
    except (ValueError, IndexError, AttributeError):
        return None


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def process_file(filepath, counts, fare_sums, mile_sums, sec_sums, seen_ids):
    rows_read = 0
    rows_valid = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        fmt = detect_format(raw_header)

        if fmt is None:
            print(f"  WARNING: cannot detect format for {filepath}, skipping")
            return 0, 0

        header = normalize_header(raw_header) if fmt == "portal" else raw_header

        try:
            ts_idx = header.index("trip_start_timestamp")
            pu_idx = header.index("pickup_community_area")
            do_idx = header.index("dropoff_community_area")
        except ValueError:
            print(f"  WARNING: required columns missing in {filepath}, skipping")
            return 0, 0

        id_idx = header.index("trip_id") if "trip_id" in header else None
        fare_idx = header.index("fare") if "fare" in header else None
        miles_idx = header.index("trip_miles") if "trip_miles" in header else None
        secs_idx = header.index("trip_seconds") if "trip_seconds" in header else None

        for row in reader:
            rows_read += 1
            max_idx = max(ts_idx, pu_idx, do_idx)
            if len(row) <= max_idx:
                continue

            # Dedup
            if id_idx is not None and id_idx < len(row):
                tid = row[id_idx]
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)

            # Parse pickup area — only process if it's an origin of interest
            pu_str = row[pu_idx].strip()
            if not pu_str:
                continue
            try:
                pu_area = int(float(pu_str))
            except (ValueError, TypeError):
                continue

            if pu_area not in ORIGINS:
                continue

            # Parse dropoff area
            do_str = row[do_idx].strip()
            if not do_str:
                continue
            try:
                do_area = int(float(do_str))
            except (ValueError, TypeError):
                continue

            # Parse date
            trip_date = parse_date(row[ts_idx], fmt)
            if not trip_date:
                continue

            # Count
            key = (pu_area, do_area, trip_date)
            counts[key] = counts.get(key, 0) + 1
            rows_valid += 1

            # Accumulate fare, miles, seconds for corridor characteristics
            corridor = (pu_area, do_area)
            if fare_idx and fare_idx < len(row):
                f_val = safe_float(row[fare_idx])
                if f_val and 0 < f_val <= 500:
                    fare_sums[corridor] = fare_sums.get(corridor, [0, 0])
                    fare_sums[corridor][0] += f_val
                    fare_sums[corridor][1] += 1

            if miles_idx and miles_idx < len(row):
                m_val = safe_float(row[miles_idx])
                if m_val and 0 < m_val <= 200:
                    mile_sums[corridor] = mile_sums.get(corridor, [0, 0])
                    mile_sums[corridor][0] += m_val
                    mile_sums[corridor][1] += 1

            if secs_idx and secs_idx < len(row):
                s_val = safe_float(row[secs_idx])
                if s_val and 0 < s_val <= 36000:
                    sec_sums[corridor] = sec_sums.get(corridor, [0, 0])
                    sec_sums[corridor][0] += s_val
                    sec_sums[corridor][1] += 1

    return rows_read, rows_valid


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/0_aggregate_corridors.py <csv_files...>")
        sys.exit(1)

    files = sys.argv[1:]
    counts = {}
    fare_sums = {}
    mile_sums = {}
    sec_sums = {}
    seen_ids = set()

    total_read = 0
    total_valid = 0

    print(f"Processing {len(files)} files (filtering origins: {sorted(ORIGINS)})...\n")

    for filepath in files:
        print(f"  {filepath}...")
        read, valid = process_file(filepath, counts, fare_sums, mile_sums, sec_sums, seen_ids)
        total_read += read
        total_valid += valid
        print(f"    {read:,} rows → {valid:,} valid (from target origins)")

    print(f"\nTotals: {total_read:,} read, {total_valid:,} valid corridor trips")
    print(f"Unique (origin, dest, date) cells: {len(counts):,}")

    # Compute corridor characteristics
    corridor_chars = {}
    for corridor in set(list(fare_sums.keys()) + list(mile_sums.keys())):
        chars = {}
        if corridor in fare_sums and fare_sums[corridor][1] > 0:
            chars["avg_fare"] = round(fare_sums[corridor][0] / fare_sums[corridor][1], 2)
        if corridor in mile_sums and mile_sums[corridor][1] > 0:
            chars["avg_miles"] = round(mile_sums[corridor][0] / mile_sums[corridor][1], 2)
        if corridor in sec_sums and sec_sums[corridor][1] > 0:
            chars["avg_seconds"] = round(sec_sums[corridor][0] / sec_sums[corridor][1], 0)
        corridor_chars[corridor] = chars

    # Write corridor_day.csv
    outfile = "data/derived/corridor_day.csv"
    rows_out = []
    for (pu, do, date_str), count in sorted(counts.items()):
        d = date.fromisoformat(date_str)
        iso = d.isocalendar()
        rows_out.append({
            "pickup_community_area": pu,
            "dropoff_community_area": do,
            "date": date_str,
            "iso_year": iso[0],
            "iso_week": iso[1],
            "weekday": d.weekday(),
            "trip_count": count,
        })

    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pickup_community_area", "dropoff_community_area", "date",
            "iso_year", "iso_week", "weekday", "trip_count"
        ])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out):,} rows to {outfile}")

    # Pre-compute corridor totals efficiently
    corridor_totals = defaultdict(lambda: [0, 0])  # [total_trips, n_days]
    for (pu, do, _), count in counts.items():
        corridor_totals[(pu, do)][0] += count
        corridor_totals[(pu, do)][1] += 1

    # Write corridor characteristics
    char_file = "data/derived/corridor_characteristics.csv"
    char_rows = []
    for (pu, do), chars in sorted(corridor_chars.items()):
        total_trips = corridor_totals[(pu, do)][0]
        n_days = corridor_totals[(pu, do)][1]
        char_rows.append({
            "pickup_community_area": pu,
            "dropoff_community_area": do,
            "total_trips": total_trips,
            "n_days": n_days,
            "avg_daily_trips": round(total_trips / n_days, 1) if n_days > 0 else 0,
            "avg_fare": chars.get("avg_fare", ""),
            "avg_miles": chars.get("avg_miles", ""),
            "avg_seconds": chars.get("avg_seconds", ""),
            "is_airport": 1 if 76 in (pu, do) else 0,
        })

    # Sort by total trips descending
    char_rows.sort(key=lambda x: x["total_trips"], reverse=True)

    with open(char_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pickup_community_area", "dropoff_community_area",
            "total_trips", "n_days", "avg_daily_trips",
            "avg_fare", "avg_miles", "avg_seconds", "is_airport"
        ])
        writer.writeheader()
        writer.writerows(char_rows)

    print(f"Wrote {len(char_rows)} corridors to {char_file}")

    # Summary: top corridors by volume
    print(f"\nTop 20 corridors by total trips:")
    print(f"{'Origin':>7} {'Dest':>7} {'Total':>10} {'Days':>6} {'Avg/Day':>8} {'Fare':>7} {'Miles':>6} {'Airport'}")
    print("-" * 65)
    for r in char_rows[:20]:
        airport = "✈" if r["is_airport"] else ""
        print(f"{r['pickup_community_area']:>7} {r['dropoff_community_area']:>7} "
              f"{r['total_trips']:>10,} {r['n_days']:>6} {r['avg_daily_trips']:>8} "
              f"${r['avg_fare']:>6} {r['avg_miles']:>5} {airport}")

    # Density tiers
    print(f"\nCorridor density tiers:")
    dense = [r for r in char_rows if r["avg_daily_trips"] >= 100]
    medium = [r for r in char_rows if 30 <= r["avg_daily_trips"] < 100]
    sparse = [r for r in char_rows if 5 <= r["avg_daily_trips"] < 30]
    very_sparse = [r for r in char_rows if r["avg_daily_trips"] < 5]
    print(f"  Dense (≥100/day):     {len(dense)} corridors")
    print(f"  Medium (30-99/day):   {len(medium)} corridors")
    print(f"  Sparse (5-29/day):    {len(sparse)} corridors")
    print(f"  Very sparse (<5/day): {len(very_sparse)} corridors")


if __name__ == "__main__":
    main()
