"""Overall CPU + RAM usage distributions: one boxplot per model per metric, pooling every
1-second sample of every task. No per-task aggregation, no task names.

Usage: uv run python charts/model_boxplot.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from taskplot import _epoch, load_samples

DEFAULT_MODELS = ["qwen3.8-flash", "deepseek-v4-flash-0731"]  # first = left


def model_samples(model_dir: Path) -> tuple[list[float], list[float]]:
    """(cpu %, memory %) per-interval samples across every task of one model."""
    cpus, mems = [], []
    for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        if not (run_dir / "metrics.jsonl").exists():
            continue
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        phase = status["phases"]["agent"]
        t0, t1 = _epoch(phase["start"]), _epoch(phase["end"])
        epochs, usage, mem, alloc_cpu, alloc_mem = load_samples(run_dir)
        pts = [(e, u, m) for e, u, m in zip(epochs, usage, mem) if t0 <= e <= t1]
        cpus += [100.0 * (b[1] - a[1]) / 1e6 / (b[0] - a[0]) / alloc_cpu
                 for a, b in zip(pts, pts[1:])]
        mems += [100.0 * p[2] / (alloc_mem * 2**30) for p in pts]
    return cpus, mems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("-o", "--out", type=Path, default=Path("charts/usage-boxplot.png"))
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    metrics = [("CPU % (of 4 cores)", 0, "tab:blue"),
               ("memory % (of 4 GiB)", 1, "tab:green")]
    per_model = [model_samples(args.runs / model) for model in args.models]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.5),
                             sharey=True)
    for ax, (title, idx, color) in zip(axes, metrics):
        data = [pm[idx] for pm in per_model]
        ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True,
                   boxprops=dict(color=color, facecolor=color, alpha=0.4),
                   whiskerprops=dict(color=color), capprops=dict(color=color),
                   medianprops=dict(color="black"),
                   flierprops=dict(marker=".", markersize=3, markerfacecolor=color,
                                   markeredgecolor="none", alpha=0.4))
        ax.set_xlabel(title)
        ax.set_xlim(0, 105)
        ax.set_yticks(range(1, len(args.models) + 1),
                      [f"{m}\n(n={len(pm[idx])})" for m, pm in zip(args.models, per_model)])
    axes[0].invert_yaxis()  # first model on top; once only, sharey syncs all
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
