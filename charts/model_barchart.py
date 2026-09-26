"""Resource usage histograms, one image: rows = models (first on top), cols = CPU / memory,
bins of 1 %, all tasks pooled like the boxplot.

Usage: uv run python charts/model_barchart.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model_boxplot import DEFAULT_MODELS, model_samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("-o", "--out", type=Path, default=Path("charts/usage-barchart.png"))
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    per_model = [model_samples(args.runs / model) for model in args.models]
    metrics = [("CPU % (of 4 cores)", 0), ("memory % (of 4 GiB)", 1)]

    bins = np.arange(0, 102)  # 1 % wide
    colors = ["tab:blue", "tab:green"]
    fig, axes = plt.subplots(len(args.models), len(metrics),
                             figsize=(10 * len(metrics), 3.5 * len(args.models)),
                             sharex="col", sharey="row", squeeze=False)
    for row, (model, pm) in enumerate(zip(args.models, per_model)):
        for col, (title, idx) in enumerate(metrics):
            ax = axes[row][col]
            counts, _ = np.histogram(pm[idx], bins=bins)
            ax.bar(bins[:-1] + 0.5, counts, width=1.0, color=colors[col])
            ax.set_yscale("log")
            for p in (1, 10, 100, 1000, 10_000, 100_000):  # powers of 10 guides
                ax.axhline(p, color="0.6", ls="--", lw=0.8)
            if col == 0:
                ax.set_ylabel(f"{model}\nsamples")
    for col, (title, _) in enumerate(metrics):
        axes[-1][col].set_xlabel(title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
