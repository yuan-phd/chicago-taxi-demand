"""
Latency test — embedding + FAISS search
========================================
Measures per-query encode latency for all-MiniLM-L6-v2 and a top-100 FAISS
search, comparing the default accelerator (MPS on Apple Silicon) vs CPU.

Runs the full 6-query sequence twice per device so the second pass shows
steady-state (post-warmup, shapes cached).

Run from project root:
    python scripts/test_latency.py
"""

import time
import numpy as np
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = "data/derived/gdelt_faiss.index"
TOP_K = 100

# 6 queries — varying lengths, mirroring the demo pipeline queries.
# Last one repeats query #5 to test shape caching (identical shape → no recompile).
QUERIES = [
    "Chicago Christmas Eve Christmas Day After Christmas travel demand",
    "Chicago Thanksgiving Black Friday RSNA Annual Meeting travel demand",
    "Chicago RSNA Annual Meeting travel demand",
    "Chicago Columbus Day travel demand",
    "Chicago travel demand",
    "Chicago travel demand",  # repeat — tests shape caching
]


def ms(seconds: float) -> float:
    return seconds * 1000.0


def run_sequence(model, index, label: str):
    """Encode each query + FAISS search, return list of per-query timing rows."""
    rows = []
    for q in QUERIES:
        t0 = time.perf_counter()
        emb = model.encode([q], normalize_embeddings=True).astype(np.float32)
        t1 = time.perf_counter()

        if index is not None:
            index.search(emb, TOP_K)
        t2 = time.perf_counter()

        encode_ms = ms(t1 - t0)
        faiss_ms = ms(t2 - t1)
        rows.append({
            "query": q,
            "length": len(q),
            "encode_ms": encode_ms,
            "faiss_ms": faiss_ms,
            "total_ms": encode_ms + faiss_ms,
        })
    return rows


def print_table(rows, title: str):
    print(f"\n  {title}")
    print(f"  {'-' * 96}")
    print(f"  {'query':<62} {'len':>4} {'encode_ms':>10} {'faiss_ms':>9} {'total_ms':>9}")
    print(f"  {'-' * 96}")
    for r in rows:
        q = r["query"] if len(r["query"]) <= 60 else r["query"][:57] + "..."
        print(f"  {q:<62} {r['length']:>4} {r['encode_ms']:>10.2f} "
              f"{r['faiss_ms']:>9.2f} {r['total_ms']:>9.2f}")
    print(f"  {'-' * 96}")
    avg_enc = np.mean([r["encode_ms"] for r in rows])
    avg_total = np.mean([r["total_ms"] for r in rows])
    print(f"  {'AVG':<62} {'':>4} {avg_enc:>10.2f} "
          f"{np.mean([r['faiss_ms'] for r in rows]):>9.2f} {avg_total:>9.2f}")


def test_device(device: str, index):
    """Load model on `device`, warmup, run the sequence twice, print both runs."""
    print("\n" + "=" * 100)
    print(f"DEVICE: {device.upper()}")
    print("=" * 100)

    t0 = time.perf_counter()
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print(f"  Model loaded on '{device}' in {ms(time.perf_counter() - t0):.1f} ms")

    # Warmup — first encode pays graph-build / kernel-compile cost.
    t0 = time.perf_counter()
    model.encode(["warmup"], normalize_embeddings=True)
    print(f"  Warmup encode: {ms(time.perf_counter() - t0):.2f} ms")

    run1 = run_sequence(model, index, device)
    print_table(run1, f"{device.upper()} — RUN 1 (cold-ish)")

    run2 = run_sequence(model, index, device)
    print_table(run2, f"{device.upper()} — RUN 2 (steady-state)")

    return {"run1": run1, "run2": run2}


def main():
    if not Path(FAISS_INDEX_PATH).exists():
        print(f"ERROR: FAISS index not found at {FAISS_INDEX_PATH}")
        print("Run: python scripts/6_build_faiss.py")
        return

    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"FAISS index loaded: {index.ntotal:,} vectors, dim {index.d}, top_k={TOP_K}")

    # Default accelerator (MPS on Apple Silicon, else CUDA/CPU).
    if torch.backends.mps.is_available():
        primary = "mps"
    elif torch.cuda.is_available():
        primary = "cuda"
    else:
        primary = "cpu"
    print(f"Primary accelerator detected: {primary.upper()}")

    test_device(primary, index)

    # Always also test CPU for comparison (skip if primary already CPU).
    if primary != "cpu":
        test_device("cpu", index)
    else:
        print("\n(Primary device is already CPU — no separate CPU section needed.)")

    print("\nDone.")


if __name__ == "__main__":
    main()
