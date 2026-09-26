"""Model comparison grid: one column per task, one row per model (first model on top).

Usage: uv run python charts/compare_grid.py
       uv run python charts/compare_grid.py --models qwen3.8-flash deepseek-v4-flash-0731
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from taskplot import plot_task

DEFAULT_MODELS = ["qwen3.8-flash", "deepseek-v4-flash-0731"]  # first = top row


def dataset_order(dataset: Path) -> list[str]:
    with dataset.open(encoding="utf-8") as fh:
        return [row["instance_id"] for row in csv.DictReader(fh)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="top row first")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--dataset", type=Path, default=Path("data/sample20.csv"))
    ap.add_argument("--tasks", nargs="+", default=None, help="default: dataset order")
    ap.add_argument("-o", "--out", type=Path, default=Path("charts/compare-grid.png"))
    ap.add_argument("--width", type=float, default=12.0, help="inches per task column")
    ap.add_argument("--height", type=float, default=8.0, help="inches per model row")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--fontsize", type=float, default=12.0)
    args = ap.parse_args()

    tasks = args.tasks or dataset_order(args.dataset)
    models = args.models
    ncols, nrows = len(tasks), len(models)

    fig, axes = plt.subplots(nrows, ncols, figsize=(args.width * ncols, args.height * nrows),
                             squeeze=False)
    handles = labels = None
    plotted = skipped = 0
    for col, task in enumerate(tasks):
        for row, model in enumerate(models):
            ax = axes[row][col]
            run_dir = args.runs / model / task
            if not (run_dir / "metrics.jsonl").exists():
                ax.text(0.5, 0.5, "no run", ha="center", va="center", transform=ax.transAxes,
                        fontsize=args.fontsize, color="0.5")
                ax.set_xticks([])
                ax.set_yticks([])
                skipped += 1
                continue
            # Colours carry the series identity (see the shared legend), so no per-panel y labels.
            h, l = plot_task(ax, run_dir, ylabels=False, legend=False,
                             title=task if row == 0 else None,
                             title_fontsize=args.fontsize)
            if row == nrows - 1:
                ax.set_xlabel("elapsed time [s]", fontsize=args.fontsize)
            if handles is None:
                handles, labels = h, l
            plotted += 1

    # Model name per row, on the left margin.
    for row, model in enumerate(models):
        fig.text(0.002, 1 - (row + 0.5) / nrows, model, rotation=90, va="center",
                 ha="center", fontsize=args.fontsize)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=args.fontsize)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.005, 0.05, 1, 1))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out} ({nrows}x{ncols}, {plotted} panels plotted, {skipped} missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
