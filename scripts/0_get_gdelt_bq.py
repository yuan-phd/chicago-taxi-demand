"""
GDELT Chicago Articles — BigQuery Download
============================================
Downloads Chicago local news articles from GDELT GKG via BigQuery.
Source-based filtering (10 Chicago outlets) for best signal quality.

Prerequisites:
    pip install google-cloud-bigquery db-dtypes pyarrow
    gcloud auth application-default login

Output:
    data/external/gdelt_chicago_articles.csv  — per-article data (for embedding)
    data/external/gdelt_chicago_weekly.csv    — weekly aggregation (for features)

Run from project root:
    python scripts/0_get_gdelt_bq.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from google.cloud import bigquery
except ImportError:
    print("ERROR: pip install google-cloud-bigquery db-dtypes pyarrow")
    raise

Path("data/external").mkdir(parents=True, exist_ok=True)

CHICAGO_SOURCES = [
    "chicagotribune.com",
    "chicago.suntimes.com",
    "suntimes.com",
    "fox32chicago.com",
    "abc7chicago.com",
    "nbcchicago.com",
    "wgntv.com",
    "chicagobusiness.com",
    "blockclubchicago.org",
    "dailyherald.com",
]

QUERY = """
SELECT
  DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS article_date,
  SourceCommonName AS source,
  DocumentIdentifier AS url,
  CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
  _PARTITIONTIME >= '2024-01-01'
  AND _PARTITIONTIME < '2026-07-01'
  AND SourceCommonName IN UNNEST(@sources)
ORDER BY article_date
"""

# ── Download ─────────────────────────────────────────────────────────
print("Querying BigQuery (may take 30-60s)...")
client = bigquery.Client()

job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter("sources", "STRING", CHICAGO_SOURCES),
    ]
)

df = client.query(QUERY, job_config=job_config).to_dataframe()
print(f"  Downloaded: {len(df):,} articles")
print(f"  Date range: {df['article_date'].min()} to {df['article_date'].max()}")
print(f"  Sources: {df['source'].nunique()}")

# Drop rows without title (can't embed)
n_before = len(df)
df = df.dropna(subset=["title"])
df = df[df["title"].str.strip() != ""]
print(f"  With valid title: {len(df):,} / {n_before:,}")

# Source breakdown
print("\n  Articles per source:")
for source, count in df["source"].value_counts().items():
    print(f"    {source:<30s} {count:>6,}")

# ── Save per-article data ────────────────────────────────────────────
articles_path = "data/external/gdelt_chicago_articles.csv"
df.to_csv(articles_path, index=False)
print(f"\nSaved: {articles_path} ({len(df):,} rows)")

# ── Weekly aggregation ───────────────────────────────────────────────
print("\nBuilding weekly aggregation...")
df["article_date"] = pd.to_datetime(df["article_date"])
iso = df["article_date"].dt.isocalendar()
df["iso_year"] = iso["year"].astype(int)
df["iso_week"] = iso["week"].astype(int)

weekly = (
    df.groupby(["iso_year", "iso_week"])
    .agg(
        article_count=("url", "count"),
        avg_tone=("tone", "mean"),
        std_tone=("tone", "std"),
        min_tone=("tone", "min"),
        max_tone=("tone", "max"),
        n_sources=("source", "nunique"),
        n_negative=("tone", lambda x: (x < -3).sum()),
        n_positive=("tone", lambda x: (x > 3).sum()),
    )
    .reset_index()
)
weekly["pct_negative"] = (weekly["n_negative"] / weekly["article_count"] * 100).round(1)
weekly["pct_positive"] = (weekly["n_positive"] / weekly["article_count"] * 100).round(1)

weekly_path = "data/external/gdelt_chicago_weekly.csv"
weekly.to_csv(weekly_path, index=False)
print(f"Saved: {weekly_path} ({len(weekly)} weeks)")

# Quick stats
print(f"\n  Weeks covered: {len(weekly)}")
print(f"  Avg articles/week: {weekly['article_count'].mean():.0f}")
print(f"  Avg tone: {weekly['avg_tone'].mean():.2f}")
print(f"  Avg % negative: {weekly['pct_negative'].mean():.1f}%")

print("\nDone.")
