"""Resource usage CCDF: y = % of samples needing at least x % of the allocation.

Grid layout: rows = models (first on top), cols = CPU / memory, all tasks pooled
like the boxplot. Long tail = curve staying high far right.

Usage: uv run python charts/model_ccdf.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model_boxplot import DEFAULT_MODELS, model_samples


def ccdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """(x = required % 0..100, y = % of samples at or above x)."""
    xs = np.arange(0, 101)
    arr = np.asarray(values)
    ys = 100.0 * (arr.size - np.searchsorted(np.sort(arr), xs, side="left")) / arr.size
    return xs, ys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("-o", "--out", type=Path, default=Path("images/usage-ccdf.png"))
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    per_model = [model_samples(args.runs / model) for model in args.models]
    metrics = [("CPU % (of 4 cores)", 0, "tab:blue"),
               ("memory % (of 4 GiB)", 1, "tab:green")]

    fig, axes = plt.subplots(len(args.models), len(metrics),
                             figsize=(6 * len(metrics), 4 * len(args.models)),
                             sharex=True, squeeze=False)
    for row, (model, pm) in enumerate(zip(args.models, per_model)):
        for col, (title, idx, color) in enumerate(metrics):
            ax = axes[row][col]
            xs, ys = ccdf(pm[idx])
            ax.step(xs, ys, where="post", color=color, lw=1.5)
            ax.set_xlim(0, 100)
            ax.set_yscale("function", functions=(np.sqrt, np.square))
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 1, 5, 10, 25, 50, 100])
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("% of samples ≥ x")
            ax.set_title(model)
            if row == len(args.models) - 1:
                ax.set_xlabel(title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
