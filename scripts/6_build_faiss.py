"""
Phase 7 — Build FAISS Index from GDELT Articles
=================================================
Embeds 232K Chicago news titles with sentence-transformers,
builds a FAISS index, saves to disk. Run once.

Prerequisites:
    pip install sentence-transformers faiss-cpu

Output:
    data/derived/gdelt_faiss.index    — FAISS flat L2 index
    data/derived/gdelt_articles_meta.pkl — article metadata with ISO weeks

Run from project root:
    python scripts/6_build_faiss.py
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path

Path("data/derived").mkdir(parents=True, exist_ok=True)

INDEX_PATH = "data/derived/gdelt_faiss.index"
META_PATH = "data/derived/gdelt_articles_meta.pkl"

# Check if already built
if Path(INDEX_PATH).exists() and Path(META_PATH).exists():
    meta = pd.read_pickle(META_PATH)
    print(f"FAISS index already exists ({len(meta):,} articles). Delete files to rebuild.")
    print(f"  {INDEX_PATH}")
    print(f"  {META_PATH}")
    exit(0)

# ── Load articles ────────────────────────────────────────────────────
print("Loading articles...")
articles = pd.read_csv("data/external/gdelt_chicago_articles.csv")
articles["article_date"] = pd.to_datetime(articles["article_date"])

# Add ISO year/week for date filtering during search
iso = articles["article_date"].dt.isocalendar()
articles["iso_year"] = iso["year"].astype(int)
articles["iso_week"] = iso["week"].astype(int)

# Drop rows with empty titles
articles = articles.dropna(subset=["title"])
articles = articles[articles["title"].str.strip() != ""].reset_index(drop=True)
print(f"  Articles with valid titles: {len(articles):,}")

# ── Embed titles ─────────────────────────────────────────────────────
print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Embedding {len(articles):,} titles (batch_size=256)...")
t0 = time.time()
embeddings = model.encode(
    articles["title"].tolist(),
    show_progress_bar=True,
    batch_size=256,
    normalize_embeddings=True,  # for cosine similarity via dot product
)
elapsed = time.time() - t0
print(f"  Embedded in {elapsed:.1f}s ({len(articles) / elapsed:.0f} titles/sec)")
print(f"  Embedding shape: {embeddings.shape}")

# ── Build FAISS index ────────────────────────────────────────────────
print("Building FAISS index...")
import faiss

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (= cosine with normalized vectors)
index.add(embeddings.astype(np.float32))
print(f"  Index size: {index.ntotal:,} vectors, {dimension}D")

# ── Save ─────────────────────────────────────────────────────────────
faiss.write_index(index, INDEX_PATH)
articles.to_pickle(META_PATH)
print(f"\nSaved: {INDEX_PATH}")
print(f"Saved: {META_PATH}")
print("Done.")
