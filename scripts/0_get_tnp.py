"""
Download Chicago TNP (ride-hail) trip counts by area × day.
Uses SODA API server-side aggregation to avoid downloading 100M+ rows.
Output: tnp_area_week.csv — same schema as area_week.csv (subset of columns)

Two datasets cover the time range:
  - 2023-2024: n26f-ihde (we filter to 2024+)
  - 2025-:     6dvr-xwnh

Usage: python 0_get_tnp.py
"""

import json
import csv
import os
import sys
import argparse
import urllib.request
import urllib.parse
from datetime import date, timedelta
import time

# Dataset identifiers on Chicago Data Portal
# 2023-2024 (n26f-ihde): full-table aggregation times out, so we query it one
#   month at a time for 2024 (see fetch_2024_monthly).
# 2025-     (6dvr-xwnh): single aggregation query works, fetched as-is.
DATASET_2024 = "n26f-ihde"
DATASET_2025 = "6dvr-xwnh"

BASE_URL = "https://data.cityofchicago.org/resource/{dataset_id}.json"
PAGE_SIZE = 50000  # SODA API max

OUTFILE = "data/external/tnp_area_week.csv"


def fetch_dataset(dataset_id, label, where_clause=None):
    """Fetch daily area-level trip counts using server-side aggregation."""
    select = (
        "date_trunc_ymd(trip_start_timestamp) as trip_date, "
        "pickup_community_area, "
        "count(*) as trip_count"
    )
    group = "date_trunc_ymd(trip_start_timestamp), pickup_community_area"
    order = "trip_date, pickup_community_area"

    all_rows = []
    offset = 0

    while True:
        params = {
            "$select": select,
            "$group": group,
            "$order": order,
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }
        if where_clause:
            params["$where"] = where_clause

        url = BASE_URL.format(dataset_id=dataset_id)
        full_url = f"{url}?{urllib.parse.urlencode(params)}"

        print(f"  {label}: fetching offset {offset}...")
        req = urllib.request.Request(full_url)
        req.add_header("Accept", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  ERROR at offset {offset}: {e}")
            break

        if not data:
            break

        all_rows.extend(data)
        print(f"    Got {len(data)} rows (total: {len(all_rows)})")

        if len(data) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(1)  # be polite

    return all_rows


def fetch_month_2024(dataset_id, month, timeout):
    """Fetch one month of 2024 daily area-level counts. Returns (rows, error)."""
    select = (
        "date_trunc_ymd(trip_start_timestamp) as trip_date, "
        "pickup_community_area, "
        "count(*) as trip_count"
    )
    group = "date_trunc_ymd(trip_start_timestamp), pickup_community_area"
    order = "trip_date, pickup_community_area"

    start = date(2024, month, 1)
    end = date(2025, 1, 1) if month == 12 else date(2024, month + 1, 1)
    where = (
        f"trip_start_timestamp >= '{start.isoformat()}T00:00:00' "
        f"AND trip_start_timestamp < '{end.isoformat()}T00:00:00'"
    )
    params = {
        "$select": select,
        "$group": group,
        "$order": order,
        "$where": where,
        "$limit": PAGE_SIZE,
    }
    url = BASE_URL.format(dataset_id=dataset_id)
    full_url = f"{url}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(full_url)
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return None, str(e)
    return data, None


def fetch_2024_monthly(dataset_id, label):
    """Fetch all 12 months of 2024 with escalating-timeout retries.

    The full-table aggregation on this dataset times out, so we issue 12
    bounded monthly queries. Failed months are retried automatically with
    progressively longer timeouts (300s -> 600s -> 900s).

    Returns (rows_by_month, failed_months):
      rows_by_month: {month: [rows]} for months that succeeded
      failed_months: sorted list of month numbers still failing after all attempts
    """
    rows_by_month = {}
    pending = list(range(1, 13))

    # (description, timeout) for the initial pass and each retry round
    attempts = [
        ("initial", 300),
        ("retry #1", 600),
        ("retry #2", 900),
    ]

    for desc, timeout in attempts:
        if not pending:
            break
        if desc != "initial":
            failed_str = ", ".join(f"2024-{m:02d}" for m in pending)
            print(f"\n  {label}: {desc} — {len(pending)} failed month(s) "
                  f"at timeout {timeout}s: {failed_str}")

        still_failing = []
        for month in pending:
            mlabel = f"2024-{month:02d}"
            print(f"  {label} [{desc}] {mlabel}: fetching (timeout {timeout}s)...")
            data, error = fetch_month_2024(dataset_id, month, timeout)

            if error is not None:
                print(f"    FAILED {mlabel}: {error}")
                still_failing.append(month)
            elif not data:
                print(f"    FAILED {mlabel}: returned 0 rows")
                still_failing.append(month)
            else:
                rows_by_month[month] = data
                print(f"    {mlabel}: got {len(data)} rows")
            time.sleep(2)  # be polite between months
        pending = still_failing

    failed_months = sorted(pending)

    # Per-month status summary
    print(f"\n  {label} — month status summary:")
    for month in range(1, 13):
        mlabel = f"2024-{month:02d}"
        if month in rows_by_month:
            print(f"    {mlabel}: SUCCESS ({len(rows_by_month[month])} rows)")
        else:
            print(f"    {mlabel}: FAIL")

    return rows_by_month, failed_months


def parse_date_to_iso_week(date_str):
    """Parse SODA date string to (iso_year, iso_week)."""
    # SODA returns dates like "2024-01-15T00:00:00.000"
    d = date.fromisoformat(date_str[:10])
    iso = d.isocalendar()
    return iso[0], iso[1]


def aggregate_to_weekly(daily_rows):
    """Aggregate daily trip counts to ISO week level."""
    weeks = {}
    skipped = 0

    for r in daily_rows:
        area = r.get("pickup_community_area")
        trip_date = r.get("trip_date")
        count = int(r.get("trip_count", 0))

        if not area or not trip_date:
            skipped += 1
            continue

        try:
            area = int(float(area))
        except (ValueError, TypeError):
            skipped += 1
            continue

        iso_year, iso_week = parse_date_to_iso_week(trip_date)
        key = (area, iso_year, iso_week)
        weeks[key] = weeks.get(key, 0) + count

    if skipped:
        print(f"  Skipped {skipped} rows (missing area or date)")

    return weeks


def load_weekly_csv(path):
    """Load an existing weekly CSV into {(area, iso_year, iso_week): count}.

    Returns an empty dict if the file does not exist.
    """
    weekly = {}
    if not os.path.exists(path):
        return weekly
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                key = (
                    int(r["pickup_community_area"]),
                    int(r["iso_year"]),
                    int(r["iso_week"]),
                )
                weekly[key] = weekly.get(key, 0) + int(r["trip_count"])
            except (KeyError, ValueError, TypeError):
                continue
    return weekly


def write_weekly_csv(weekly, path):
    """Write {(area, iso_year, iso_week): count} to CSV, sorted. Returns rows."""
    rows_out = []
    for (area, yr, wk), count in sorted(weekly.items()):
        rows_out.append({
            "pickup_community_area": area,
            "iso_year": yr,
            "iso_week": wk,
            "trip_count": count,
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pickup_community_area", "iso_year", "iso_week", "trip_count"
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def print_nns_summary(rows_out):
    """Print a quick sanity summary for the Near North Side (area 8)."""
    nns = {(r["iso_year"], r["iso_week"]): r["trip_count"]
           for r in rows_out if r["pickup_community_area"] == 8}
    if nns:
        vals = list(nns.values())
        print(f"\nNNS (area 8) summary:")
        print(f"  Weeks: {len(vals)}")
        print(f"  Avg weekly trips: {sum(vals)/len(vals):,.0f}")
        print(f"  Min: {min(vals):,}  Max: {max(vals):,}")


def retry_months(months):
    """Fetch only the given 2024 months (timeout 600s) and merge into OUTFILE.

    New weekly counts are ADDED to whatever is already in the file. This is the
    right behavior for filling in months that previously failed (and so are
    absent from the file, including the days they contribute to boundary weeks
    shared with an adjacent month). Do NOT re-run this for a month that already
    has data in the file — it would double-count.
    """
    label = ", ".join(f"2024-{m:02d}" for m in months)
    print(f"Retrying month(s) {label} (timeout 600s), merging into {OUTFILE}\n")

    new_daily = []
    failed = []
    for m in months:
        mlabel = f"2024-{m:02d}"
        print(f"  {mlabel}: fetching (timeout 600s)...")
        data, error = fetch_month_2024(DATASET_2024, m, 600)
        if error is not None:
            print(f"    FAILED {mlabel}: {error}")
            failed.append(m)
        elif not data:
            print(f"    FAILED {mlabel}: returned 0 rows")
            failed.append(m)
        else:
            print(f"    {mlabel}: got {len(data)} rows")
            new_daily.extend(data)
        time.sleep(2)

    if not new_daily:
        print("\n✗ No data fetched for the requested month(s); leaving file unchanged.")
        sys.exit(1)

    new_weekly = aggregate_to_weekly(new_daily)
    weekly = load_weekly_csv(OUTFILE)
    before = len(weekly)
    for key, cnt in new_weekly.items():
        weekly[key] = weekly.get(key, 0) + cnt

    rows_out = write_weekly_csv(weekly, OUTFILE)
    print(f"\nMerged {len(new_weekly)} area×week rows into {OUTFILE} "
          f"({before} -> {len(rows_out)} total rows).")
    if failed:
        failed_str = ", ".join(f"2024-{m:02d}" for m in failed)
        print(f"⚠ Still failed: {failed_str}")
    print_nns_summary(rows_out)


def main():
    print("Downloading TNP (ride-hail) data from Chicago Data Portal...\n")

    all_daily = []

    # 2024 (n26f-ihde): full-table aggregation times out -> query month by month
    rows_by_month, failed_months = fetch_2024_monthly(DATASET_2024, "TNP 2024")

    for month in sorted(rows_by_month):
        all_daily.extend(rows_by_month[month])
    print(f"  TNP 2024: {len(all_daily)} daily×area rows "
          f"({len(rows_by_month)}/12 months)\n")

    # 2025- (6dvr-xwnh): single aggregation query works, fetched as-is
    rows_2025 = fetch_dataset(DATASET_2025, "TNP 2025-", None)
    all_daily.extend(rows_2025)
    print(f"  TNP 2025-: {len(rows_2025)} daily×area rows\n")

    print(f"Total daily rows: {len(all_daily)}")

    # Aggregate to weekly and always write whatever we have
    weekly = aggregate_to_weekly(all_daily)
    print(f"Weekly aggregated: {len(weekly)} area×week rows")

    rows_out = write_weekly_csv(weekly, OUTFILE)
    print(f"\nWrote {len(rows_out)} rows to {OUTFILE}")

    # Warn about any months missing from the output so they can be filled in
    if failed_months:
        failed_str = " ".join(f"2024-{m:02d}" for m in failed_months)
        print(f"\n⚠ WARNING: {len(failed_months)} of 12 2024 month(s) failed and are "
              f"MISSING from the output: {failed_str.replace(' ', ', ')}")
        print(f"  Fill them in with:")
        print(f"    python scripts/0_get_tnp.py --months {failed_str}")

    print_nns_summary(rows_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Chicago TNP trip counts by area × week."
    )
    parser.add_argument(
        "--months", nargs="+", metavar="YYYY-MM",
        help="Retry only these 2024 month(s) (timeout 600s) and merge into the "
             "existing output instead of a full run. Example: --months 2024-03",
    )
    cli_args = parser.parse_args()

    if cli_args.months:
        months = []
        for s in cli_args.months:
            parts = s.split("-")
            try:
                yr, mo = int(parts[0]), int(parts[1])
            except (IndexError, ValueError):
                parser.error(f"invalid --months value '{s}'; use YYYY-MM, e.g. 2024-03")
            if yr != 2024 or not (1 <= mo <= 12):
                parser.error(f"--months only supports 2024-01 .. 2024-12; got '{s}'")
            months.append(mo)
        retry_months(sorted(set(months)))
    else:
        main()
