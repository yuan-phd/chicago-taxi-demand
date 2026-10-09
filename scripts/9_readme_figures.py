"""
README figures — static PNGs for the two headline forecasting results.

  slides/fig_single_series_benchmark.png  single dense series (NNS): classical beats ML/DL
  slides/fig_corridor_global_xgb.png      183 corridors, daily: global XGBoost vs baseline by tier

Inputs: outputs/phase4_dl_benchmark.csv, outputs/phase5_daily_corridor_detail.csv
Run:    python scripts/9_readme_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "slides"

CLASSICAL = "#2a78d6"
ML_DL = "#eb6834"
BASELINE = "#b4b2a9"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_single_series():
    bench = pd.read_csv(ROOT / "outputs/phase4_dl_benchmark.csv")
    labels = {
        "Trend × seasonal (weekly baseline)": ("Trend × seasonal (weekly)", "classical"),
        "M1 (DOW×month×holiday factors)": ("DOW × month × holiday factors", "classical"),
        "timesfm": ("TimesFM zero-shot", "ml"),
        "nbeats": ("N-BEATS trained", "ml"),
        "naive_lag7": ("Naive lag-7", "classical"),
    }
    rows = [(labels[m][0], labels[m][1], v) for m, v in zip(bench["model"], bench["weekly_mape"])]
    # Phase 3 XGBoost v1 on the same series (slides/phase1_3_conclusions.md)
    rows.append(("XGBoost (lag-7, holiday flags)", "ml", 6.10))
    df = pd.DataFrame(rows, columns=["label", "kind", "mape"]).sort_values("mape")

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    colors = [CLASSICAL if k == "classical" else ML_DL for k in df["kind"]]
    ax.barh(df["label"], df["mape"], color=colors, height=0.6)
    ax.invert_yaxis()
    for y, v in enumerate(df["mape"]):
        ax.text(v + 0.15, y, f"{v:.2f}%", va="center", color=TEXT, fontsize=10)
    ax.axvline(6.01, color=TEXT_2, linestyle="--", linewidth=1)
    ax.set_xlim(0, df["mape"].max() * 1.15)
    ax.set_xlabel("Weekly MAPE (%) — lower is better")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CLASSICAL),
                       plt.Rectangle((0, 0), 1, 1, color=ML_DL)],
              labels=["Classical / naive", "ML / deep learning"],
              loc="upper right", frameon=False)
    ax.set_title("Single dense series (NNS, 882 days): classical is unbeaten",
                 loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_single_series_benchmark.png", facecolor="white")
    plt.close(fig)


def fig_corridor():
    d = pd.read_csv(ROOT / "outputs/phase5_daily_corridor_detail.csv")
    tiers = ["dense", "medium", "sparse"]
    g = (d.groupby("tier")
          .agg(n=("corridor_id", "size"),
               base=("baseline_weekly_mape", "mean"),
               xgb=("xgb_weekly_mape", "mean"))
          .reindex(tiers)
          .round(1))  # reductions from 1-dp MAPEs, matching the published 18.2%→9.7% = 47%

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    x = range(len(tiers))
    w = 0.36
    ax.bar([i - w / 2 - 0.01 for i in x], g["base"], w, color=BASELINE,
           label="Trailing 4-wk same-DOW baseline")
    ax.bar([i + w / 2 + 0.01 for i in x], g["xgb"], w, color=CLASSICAL,
           label="Global XGBoost (one model, all corridors)")
    for i, (b, m) in enumerate(zip(g["base"], g["xgb"])):
        ax.text(i - w / 2 - 0.01, b + 0.4, f"{b:.1f}%", ha="center", color=TEXT, fontsize=10)
        ax.text(i + w / 2 + 0.01, m + 0.4, f"{m:.1f}%", ha="center", color=TEXT, fontsize=10)
        ax.text(i, max(b, m) + 2.4, f"−{(1 - m / b) * 100:.0f}%", ha="center",
                color=TEXT, fontsize=11, fontweight="bold")
    ax.set_xticks(list(x), [f"{t.capitalize()} ({n})" for t, n in zip(tiers, g["n"])])
    ax.set_ylim(0, g["base"].max() * 1.3)
    ax.set_ylabel("Weekly MAPE (%) — lower is better")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_title("183 corridors, daily model: cross-corridor learning cuts dense-route error 47%",
                 loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_corridor_global_xgb.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_single_series()
    fig_corridor()
    print(f"Saved figures to {OUT}")
