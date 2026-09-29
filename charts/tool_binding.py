"""How much resource usage is bound to tool calls.

Every 1 s sample of every task is classified inside vs outside a tool-call window
(from trajectory.traj.json). Panels: violins of CPU % / memory % per model.

Usage: uv run python charts/tool_binding.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from taskplot import _epoch, load_samples, load_tool_windows

DEFAULT_MODELS = ["qwen3.8-flash", "deepseek-v4-flash-0731"]


def run_points(run_dir: Path):
    """(cpu %, mem %, in_tool, interval_s, cpu_core_s) per 1 s sample of the agent phase."""
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    phase = status["phases"]["agent"]
    t0, t1 = _epoch(phase["start"]), _epoch(phase["end"])
    epochs, usage, mem, alloc_cpu, alloc_mem = load_samples(run_dir)
    windows = load_tool_windows(run_dir)
    pts = [(e, u, m) for e, u, m in zip(epochs, usage, mem) if t0 <= e <= t1]
    out = []
    for a, b in zip(pts, pts[1:]):
        dt = b[0] - a[0]
        if dt <= 0:
            continue
        mid = (a[0] + b[0]) / 2
        inside = any(s <= mid <= e for s, e in windows)
        cpu = 100.0 * (b[1] - a[1]) / 1e6 / dt / alloc_cpu
        mem_pct = 100.0 * b[2] / (alloc_mem * 2**30)
        out.append((cpu, mem_pct, inside, dt, (b[1] - a[1]) / 1e6))
    return out


def collect(model_dir: Path):
    cpu_in, cpu_out, mem_in, mem_out = [], [], [], []
    wall_total = wall_tool = cpu_s_total = cpu_s_tool = 0.0
    for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        if not (run_dir / "metrics.jsonl").exists():
            continue
        for cpu, mem_pct, inside, dt, cpu_s in run_points(run_dir):
            (cpu_in if inside else cpu_out).append(cpu)
            (mem_in if inside else mem_out).append(mem_pct)
            wall_total += dt
            cpu_s_total += cpu_s
            if inside:
                wall_tool += dt
                cpu_s_tool += cpu_s
    shares = (100 * wall_tool / wall_total, 100 * cpu_s_tool / cpu_s_total)
    return (cpu_in, cpu_out, mem_in, mem_out), shares


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("-o", "--out", type=Path, default=Path("images/tool-binding.png"))
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    per_model = [collect(args.runs / m) for m in args.models]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["tab:red", "tab:blue"]  # in-tool, outside

    # col 1 + 2: violins
    for col, (title, i_in, i_out) in enumerate(
            [("CPU % (of 4 cores)", 0, 1), ("memory % (of 4 GiB)", 2, 3)]):
        ax = axes[col]
        data, positions, labels = [], [], []
        for row, (model, (groups, _)) in enumerate(zip(args.models, per_model)):
            base = row * 3
            data += [groups[i_in], groups[i_out]]
            positions += [base + 1, base + 2]
            labels += ["tool", "idle"]
        parts = ax.violinplot(data, positions=positions, widths=0.8,
                              showmeans=True, showextrema=False)
        for pc, k in zip(parts["bodies"], range(len(data))):
            pc.set_facecolor(colors[k % 2])
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")
        ax.set_xticks(positions, labels, fontsize=8)
        ax.set_title(title)
        # model separators
        ax.set_xlim(0.3, len(args.models) * 3 - 0.3)
        for row in range(1, len(args.models)):
            ax.axvline(row * 3, color="0.8", lw=1)
        for row, model in enumerate(args.models):
            ax.annotate(model, (row * 3 + 1.5, 0.98), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=10)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    for m, (_, s) in zip(args.models, per_model):
        print(f"{m}: wall {s[0]:.1f}%, cpu-seconds {s[1]:.1f}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
