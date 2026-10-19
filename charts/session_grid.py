"""Session comparison grid, like compare_grid.py: one column per task, per model a CPU row then a
memory row (first model on top), in cores and MiB so both agents compare. Shared y per row.

--agent opencode|miniswe  (runs/<agent>-<model>/, local_benchmark.py): harness and tool
                  containers, see session_chart.py.

Usage: uv run python charts/session_grid.py --agent opencode
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_grid import DEFAULT_MODELS, dataset_order
from session_chart import TOOL_COLORS, cgroup_series, plot_session
from taskplot import load_tool_windows


def plot_daytona(ax_cpu: plt.Axes, ax_mem: plt.Axes, run_dir: Path) -> None:
    t0, cpu, mem = cgroup_series(run_dir)
    x = np.arange(len(cpu))
    centres = x + 0.5
    windows = [(centres >= s - t0) & (centres <= e - t0) for s, e in load_tool_windows(run_dir)]
    for ax, y in ((ax_cpu, cpu), (ax_mem, mem)):
        ax.plot(x, y, color="tab:blue", lw=2.2, marker=".", markersize=4,
                label="sandbox (agent + tools)")
        for i, w in enumerate(windows):
            ax.plot(x[w], y[w], color=TOOL_COLORS[i % len(TOOL_COLORS)], lw=0.6, marker=".",
                    markersize=2, label="during a tool call (one colour per call)" if i == 0 else None)
        ax.set_xlim(0, len(x))
        ax.grid(alpha=0.3)


AGENTS = {
    "opencode": (lambda runs, model: runs / f"opencode-{model}", "toolcalls.jsonl", plot_session),
    "miniswe": (lambda runs, model: runs / f"miniswe-{model}", "toolcalls.jsonl", plot_session),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=AGENTS, default="opencode")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="top rows first")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--dataset", type=Path, default=Path("data/sample20.csv"))
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="default: images/<agent>-compare-grid.png")
    ap.add_argument("--width", type=float, default=12.0, help="inches per task column")
    ap.add_argument("--height", type=float, default=4.0, help="inches per metric row")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--fontsize", type=float, default=12.0)
    args = ap.parse_args()
    model_dir, marker_file, plot = AGENTS[args.agent]
    out = args.out or Path("images") / f"{args.agent}-compare-grid.png"

    tasks = dataset_order(args.dataset)
    fig, axes = plt.subplots(2 * len(args.models), len(tasks), squeeze=False,
                             figsize=(args.width * len(tasks), args.height * 2 * len(args.models)),
                             sharey="row")
    for m, model in enumerate(args.models):
        cpu_row, mem_row = axes[2 * m], axes[2 * m + 1]
        cpu_row[0].set_ylabel(f"{model}\nCPU (cores)", fontsize=args.fontsize)
        mem_row[0].set_ylabel(f"{model}\nmemory (MiB)", fontsize=args.fontsize)
        for col, task in enumerate(tasks):
            run_dir = model_dir(args.runs, model) / task
            if m == 0:
                cpu_row[col].set_title(task, fontsize=args.fontsize)
            status = run_dir / "status.json"
            # finished_at: a run still in progress has no agent end to scale against.
            if not (run_dir / marker_file).exists() or '"finished_at"' not in status.read_text():
                cpu_row[col].text(0.5, 0.5, "no run", ha="center", va="center",
                                  transform=cpu_row[col].transAxes, color="0.5")
                continue
            plot(cpu_row[col], mem_row[col], run_dir)
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
