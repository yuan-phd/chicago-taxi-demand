"""
Aggregate raw Chicago taxi trip CSVs to daily × area trip counts.
Handles both SODA API format (lowercase) and Portal CSV format (Title Case).
Output: data/derived/area_day.csv

Usage: python 0_aggregate_daily.py <csv_files...>

Example:
  python 0_aggregate_daily.py \
    ~/data/taxi_2024_h1.csv \
    ~/data/taxi_2024_h1b.csv \
    ~/data/Taxi_Trips_202407_202409.csv \
    ~/data/Taxi_Trips_202410_202412.csv \
    ~/data/Taxi_Trips_202501_202503.csv \
    ~/data/Taxi_Trips_202504_202506.csv \
    ~/data/Taxi_Trips_202507_202509.csv \
    ~/data/Taxi_Trips_202510_202512.csv \
    "~/data/Taxi_Trips_(2024-)_20260611.csv"
"""

import csv
import sys
from datetime import date

# Column name aliases (Portal CSV → SODA API format)
COLUMN_MAP = {
    "Trip Start Timestamp": "trip_start_timestamp",
    "Pickup Community Area": "pickup_community_area",
    "Trip ID": "trip_id",
}


def detect_format(header):
    """Detect whether CSV uses SODA (lowercase) or Portal (Title Case) format."""
    if "trip_start_timestamp" in header:
        return "soda"
    elif "Trip Start Timestamp" in header:
        return "portal"
    else:
        return None


def normalize_header(header):
    """Map Portal CSV column names to SODA API equivalents."""
    return [COLUMN_MAP.get(col, col) for col in header]


def parse_date(timestamp_str, fmt):
    """Extract date from timestamp string."""
    try:
        if fmt == "soda":
            # ISO format: "2024-01-15T14:30:00.000"
            return timestamp_str[:10]
        else:
            # Portal format: "01/15/2024 02:30:00 PM" or "1/15/2024 2:30:00 PM"
            parts = timestamp_str.split(" ")[0]  # get date part
            month, day, year = parts.split("/")
            return f"{year}-{int(month):02d}-{int(day):02d}"
    except (ValueError, IndexError, AttributeError):
        return None


def process_file(filepath, counts, seen_ids):
    """Process one CSV file, accumulating trip counts."""
    rows_read = 0
    rows_valid = 0
    rows_dup = 0
    rows_skip = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        fmt = detect_format(raw_header)

        if fmt is None:
            print(f"  WARNING: cannot detect format for {filepath}, skipping")
            return 0, 0, 0, 0

        if fmt == "portal":
            header = normalize_header(raw_header)
        else:
            header = raw_header

        try:
            ts_idx = header.index("trip_start_timestamp")
            area_idx = header.index("pickup_community_area")
        except ValueError:
            print(f"  WARNING: required columns missing in {filepath}, skipping")
            return 0, 0, 0, 0

        # Check for trip_id to deduplicate
        try:
            id_idx = header.index("trip_id")
        except ValueError:
            id_idx = None

        for row in reader:
            rows_read += 1

            if len(row) <= max(ts_idx, area_idx):
                rows_skip += 1
                continue

            # Dedup by trip_id
            if id_idx is not None and id_idx < len(row):
                tid = row[id_idx]
                if tid in seen_ids:
                    rows_dup += 1
                    continue
                seen_ids.add(tid)

            # Parse date
            trip_date = parse_date(row[ts_idx], fmt)
            if not trip_date:
                rows_skip += 1
                continue

            # Parse area
            area_str = row[area_idx].strip()
            if not area_str:
                rows_skip += 1
                continue
            try:
                area = int(float(area_str))
            except (ValueError, TypeError):
                rows_skip += 1
                continue

            # Count
            key = (area, trip_date)
            counts[key] = counts.get(key, 0) + 1
            rows_valid += 1

    return rows_read, rows_valid, rows_dup, rows_skip


def main():
    if len(sys.argv) < 2:
        print("Usage: python 0_aggregate_daily.py <csv_files...>")
        sys.exit(1)

    files = sys.argv[1:]
    counts = {}  # (area, date_str) → trip_count
    seen_ids = set()

    total_read = 0
    total_valid = 0
    total_dup = 0
    total_skip = 0

    print(f"Processing {len(files)} files...\n")

    for filepath in files:
        print(f"  {filepath}...")
        read, valid, dup, skip = process_file(filepath, counts, seen_ids)
        total_read += read
        total_valid += valid
        total_dup += dup
        total_skip += skip
        print(f"    {read:,} rows → {valid:,} valid, {dup:,} dup, {skip:,} skipped")

    print(f"\nTotals: {total_read:,} read, {total_valid:,} valid, "
          f"{total_dup:,} dup, {total_skip:,} skipped")
    print(f"Unique area×day cells: {len(counts):,}")

    # Write output
    outfile = "data/derived/area_day.csv"
    rows_out = []
    for (area, date_str), count in sorted(counts.items()):
        d = date.fromisoformat(date_str)
        iso = d.isocalendar()
        rows_out.append({
            "pickup_community_area": area,
            "date": date_str,
            "iso_year": iso[0],
            "iso_week": iso[1],
            "iso_weekday": iso[2],
            "trip_count": count,
        })

    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pickup_community_area", "date", "iso_year", "iso_week",
            "iso_weekday", "trip_count"
        ])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out):,} rows to {outfile}")

    # Quick NNS summary
    nns_daily = [r for r in rows_out if r["pickup_community_area"] == 8]
    if nns_daily:
        vals = [r["trip_count"] for r in nns_daily]
        print(f"\nNNS (area 8) daily summary:")
        print(f"  Days: {len(vals)}")
        print(f"  Avg daily trips: {sum(vals)/len(vals):,.0f}")
        print(f"  Min: {min(vals):,}  Max: {max(vals):,}")


if __name__ == "__main__":
    main()
