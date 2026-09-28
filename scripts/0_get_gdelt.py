"""
Download Chicago-related news from GDELT DOC API, week by week.
Outputs two files:
  - gdelt_chicago_articles.csv  (per-article: title, url, tone, date)
  - gdelt_chicago_weekly.csv    (aggregated: weekly features for modeling)

Uses GDELT DOC 2.0 API (free, no API key needed).
If historical queries fail (API may limit to recent months),
see BigQuery alternative at the bottom of this file.

Usage: python 0_get_gdelt.py
"""

import json
import csv
import os
import urllib.request
import urllib.parse
from datetime import date, timedelta
import time

# Query parameters
QUERY = "Chicago"  # broad — captures all Chicago news, not just transportation
# We cast a wide net; relevance filtering happens later via embeddings
MAX_RECORDS = 250  # API max per request

# Date range matching our taxi data
START_DATE = date(2024, 1, 1)
END_DATE = date(2026, 6, 22)

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Output paths (also used for resume)
ART_FILE = "data/external/gdelt_chicago_articles.csv"
WEEKLY_FILE = "data/external/gdelt_chicago_weekly.csv"

BASE_DELAY = 120  # seconds between requests


def iso_week_mondays(start, end):
    """Generate (monday, sunday) pairs for each ISO week in range."""
    # Find first Monday on or after start
    d = start
    while d.weekday() != 0:  # Monday = 0
        d += timedelta(days=1)

    while d <= end:
        monday = d
        sunday = d + timedelta(days=6)
        if sunday > end:
            sunday = end
        yield monday, sunday
        d += timedelta(weeks=1)


def fetch_week(monday, sunday):
    """Fetch articles for one week from GDELT DOC API."""
    start_dt = monday.strftime("%Y%m%d%H%M%S")
    end_dt = (sunday + timedelta(days=1)).strftime("%Y%m%d%H%M%S")  # exclusive

    params = {
        "query": QUERY,
        "mode": "ArtList",
        "maxrecords": MAX_RECORDS,
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "format": "json",
        "sort": "DateDesc",
    }

    url = f"{API_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return None, str(e)

    articles = data.get("articles", [])
    return articles, None


def fetch_with_retry(monday, sunday):
    """Fetch with exponential backoff on 429: up to 4 retries (120, 240, 480, 960s)."""
    backoffs = [120, 240, 480, 960]
    articles, error = fetch_week(monday, sunday)
    for attempt, wait in enumerate(backoffs):
        if not (error and "429" in str(error)):
            return articles, error
        print(f"    Rate limited, retrying in {wait}s (retry {attempt+1}/{len(backoffs)})...")
        time.sleep(wait)
        articles, error = fetch_week(monday, sunday)
    return articles, error


def write_outputs(all_articles, weekly_by_key, verbose=True):
    """Write per-article and weekly aggregated CSVs. Safe to call repeatedly.

    weekly_by_key is a dict keyed by (iso_year, iso_week); rows are written
    sorted by that key so re-fetching a week overwrites rather than duplicates.
    """
    # Write per-article data
    if all_articles:
        with open(ART_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "iso_year", "iso_week", "date", "title", "url",
                "tone", "domain", "language"
            ])
            writer.writeheader()
            writer.writerows(all_articles)
        if verbose:
            print(f"\nWrote {len(all_articles)} articles to {ART_FILE}")

    # Write weekly aggregated features
    rows = [weekly_by_key[k] for k in sorted(weekly_by_key)]
    with open(WEEKLY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "iso_year", "iso_week", "article_count", "avg_tone",
            "positive_count", "negative_count", "status"
        ])
        writer.writeheader()
        writer.writerows(rows)
    if verbose:
        print(f"Wrote {len(rows)} weeks to {WEEKLY_FILE}")


def load_progress():
    """Resume support: load any existing outputs from a prior (interrupted) run.

    Returns (weekly_by_key, all_articles, done_ok):
      weekly_by_key: {(iso_year, iso_week): row dict} of all prior weekly rows
      all_articles:  list of previously saved article rows (preserved on rewrite)
      done_ok:       set of (iso_year, iso_week) already fetched with status=ok
    """
    weekly_by_key = {}
    all_articles = []
    done_ok = set()

    if os.path.exists(WEEKLY_FILE):
        with open(WEEKLY_FILE, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    key = (int(row["iso_year"]), int(row["iso_week"]))
                except (KeyError, ValueError, TypeError):
                    continue
                weekly_by_key[key] = row
                if row.get("status") == "ok":
                    done_ok.add(key)

    if os.path.exists(ART_FILE):
        with open(ART_FILE, newline="", encoding="utf-8") as f:
            all_articles = list(csv.DictReader(f))

    return weekly_by_key, all_articles, done_ok


def main():
    print("Downloading Chicago news from GDELT DOC API...\n")

    # Resume: load any prior progress and skip weeks already fetched (status=ok)
    weekly_by_key, all_articles, done_ok = load_progress()
    if done_ok:
        print(f"Resuming: {len(done_ok)} weeks already complete — will skip them.\n")

    failures = []

    weeks = list(iso_week_mondays(START_DATE, END_DATE))
    total_weeks = len(weeks)
    keys = [monday.isocalendar()[:2] for monday, _ in weeks]

    for i, (monday, sunday) in enumerate(weeks):
        iso_yr, iso_wk = keys[i]
        key = (iso_yr, iso_wk)
        label = f"W{iso_wk:02d} {iso_yr} ({monday} to {sunday})"

        # Skip weeks already fetched successfully in a prior run
        if key in done_ok:
            print(f"  [{i+1}/{total_weeks}] {label}: already done, skipping")
            continue

        articles, error = fetch_with_retry(monday, sunday)

        if error:
            print(f"  [{i+1}/{total_weeks}] {label}: FAILED — {error}")
            failures.append((iso_yr, iso_wk, error))
            weekly_by_key[key] = {
                "iso_year": iso_yr,
                "iso_week": iso_wk,
                "article_count": 0,
                "avg_tone": "",
                "positive_count": 0,
                "negative_count": 0,
                "status": "failed",
            }
        elif articles is None or len(articles) == 0:
            print(f"  [{i+1}/{total_weeks}] {label}: 0 articles")
            weekly_by_key[key] = {
                "iso_year": iso_yr,
                "iso_week": iso_wk,
                "article_count": 0,
                "avg_tone": "",
                "positive_count": 0,
                "negative_count": 0,
                "status": "empty",
            }
        else:
            # Extract per-article data
            tones = []
            for art in articles:
                tone = art.get("tone", 0)
                try:
                    tone = float(tone)
                except (ValueError, TypeError):
                    tone = 0.0
                tones.append(tone)

                all_articles.append({
                    "iso_year": iso_yr,
                    "iso_week": iso_wk,
                    "date": art.get("seendate", "")[:10],
                    "title": art.get("title", ""),
                    "url": art.get("url", ""),
                    "tone": tone,
                    "domain": art.get("domain", ""),
                    "language": art.get("language", ""),
                })

            avg_tone = sum(tones) / len(tones) if tones else 0
            pos = sum(1 for t in tones if t > 0)
            neg = sum(1 for t in tones if t < 0)

            weekly_by_key[key] = {
                "iso_year": iso_yr,
                "iso_week": iso_wk,
                "article_count": len(articles),
                "avg_tone": round(avg_tone, 3),
                "positive_count": pos,
                "negative_count": neg,
                "status": "ok",
            }
            done_ok.add(key)

            print(f"  [{i+1}/{total_weeks}] {label}: {len(articles)} articles, "
                  f"avg tone={avg_tone:+.2f}")

            # Save after every successful fetch so an interruption loses nothing
            write_outputs(all_articles, weekly_by_key, verbose=False)
            weeks_left = sum(1 for j in range(i + 1, total_weeks) if keys[j] not in done_ok)
            eta_min = weeks_left * BASE_DELAY / 60
            print(f"    ↳ progress saved; ~{eta_min:.0f} min remaining "
                  f"({weeks_left} weeks left to fetch)")

        # Rate limit: be polite to free API
        time.sleep(BASE_DELAY)

    # Final write of all collected data
    write_outputs(all_articles, weekly_by_key)

    # Report failures
    if failures:
        print(f"\n⚠ {len(failures)} weeks failed. If historical coverage is limited,")
        print(f"  use the BigQuery alternative (see bottom of this script).")

    # Coverage summary
    ok_weeks = sum(1 for s in weekly_by_key.values() if s["status"] == "ok")
    print(f"\nCoverage: {ok_weeks}/{total_weeks} weeks with data")


if __name__ == "__main__":
    main()


# =============================================================================
# BIGQUERY ALTERNATIVE (if GDELT DOC API lacks historical coverage)
# =============================================================================
#
# Prerequisites:
#   pip install google-cloud-bigquery
#   Set up GCP project (free tier: 1TB/month queries)
#   Authenticate: gcloud auth application-default login
#
# SQL query for GDELT GKG (Global Knowledge Graph) table:
#
# SELECT
#   DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(date AS STRING))) as article_date,
#   DocumentIdentifier as url,
#   SPLIT(V2Themes, ';') as themes,
#   V2Tone as tone_csv,  -- first value is overall tone
#   Locations as locations
# FROM `gdelt-bq.gdeltv2.gkg_partitioned`
# WHERE
#   _PARTITIONTIME >= '2024-01-01'
#   AND _PARTITIONTIME < '2026-07-01'
#   AND Locations LIKE '%Chicago%'
# ORDER BY article_date
#
# Estimated cost: ~$2-5 (well within free tier)
# Expected rows: ~5,000-20,000 articles
