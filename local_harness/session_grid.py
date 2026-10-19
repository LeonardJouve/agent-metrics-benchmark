"""Session comparison grid, like compare_grid.py: one column per task, per model a CPU row then a
memory row (first model on top), in cores and MiB so both agents compare. Shared y per row.

--agent opencode|miniswe: experiment 2 (opencode_tools/) or 3 (miniswe_tools/), harness and tool
containers, see session_chart.py.

Usage: uv run python -m local_harness.session_grid --agent opencode
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from local_harness.session_chart import plot_session
from shared.plot import DEFAULT_DATASET, dataset_order
from shared.runner import MODELS

EXPERIMENTS = {"opencode": Path("opencode_tools"), "miniswe": Path("miniswe_tools")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=EXPERIMENTS, default="opencode")
    ap.add_argument("--models", nargs="+", default=MODELS, help="top rows first")
    ap.add_argument("--runs", type=Path, default=None, help="default: <exp>/runs")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="default: <exp>/images/compare-grid.png")
    ap.add_argument("--width", type=float, default=12.0, help="inches per task column")
    ap.add_argument("--height", type=float, default=4.0, help="inches per metric row")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--fontsize", type=float, default=12.0)
    args = ap.parse_args()
    exp_dir = EXPERIMENTS[args.agent]
    runs = args.runs or exp_dir / "runs"
    out = args.out or exp_dir / "images" / "compare-grid.png"

    tasks = dataset_order(args.dataset)
    fig, axes = plt.subplots(2 * len(args.models), len(tasks), squeeze=False,
                             figsize=(args.width * len(tasks), args.height * 2 * len(args.models)),
                             sharey="row")
    for m, model in enumerate(args.models):
        cpu_row, mem_row = axes[2 * m], axes[2 * m + 1]
        cpu_row[0].set_ylabel(f"{model}\nCPU (cores)", fontsize=args.fontsize)
        mem_row[0].set_ylabel(f"{model}\nmemory (MiB)", fontsize=args.fontsize)
        for col, task in enumerate(tasks):
            run_dir = runs / model / task
            if m == 0:
                cpu_row[col].set_title(task, fontsize=args.fontsize)
            status = run_dir / "status.json"
            # finished_at: a run still in progress has no agent end to scale against.
            if not (run_dir / "toolcalls.jsonl").exists() or '"finished_at"' not in status.read_text():
                cpu_row[col].text(0.5, 0.5, "no run", ha="center", va="center",
                                  transform=cpu_row[col].transAxes, color="0.5")
                continue
            plot_session(cpu_row[col], mem_row[col], run_dir)
    for row in axes:
        row[0].set_ylim(bottom=0)  # shared per row
    for ax in axes[-1]:
        ax.set_xlabel("s since agent start", fontsize=args.fontsize)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=args.fontsize)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
