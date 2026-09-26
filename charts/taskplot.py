"""Plot one task's resource timeline onto an Axes.

Used by task_chart.py (single figure) and compare_grid.py (models side by side).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def load_samples(run_dir: Path) -> tuple[list[float], list[int], list[int], int, float]:
    """(epoch, usage_usec, memory_current, alloc_cpu, alloc_mem_gib) from metrics.jsonl."""
    epochs, usage, mem = [], [], []
    alloc_cpu, alloc_mem = 4, 4.0
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            epochs.append(s["epoch"])
            usage.append(s["usage_usec"])
            mem.append(s["memory_current"])
            alloc_cpu = s.get("alloc_cpu", alloc_cpu)
            alloc_mem = s.get("alloc_mem_gib", alloc_mem)
    return epochs, usage, mem, alloc_cpu, alloc_mem


def load_tool_windows(run_dir: Path) -> list[tuple[float, float]]:
    """(tool call start, tool response end) pairs.

    mini-swe-agent stamps each message on append: the assistant message carrying the tool call
    is stamped when the API response lands, the following tool message when the call returns.
    """
    traj = json.loads((run_dir / "trajectory.traj.json").read_text(encoding="utf-8"))
    windows, pending = [], None
    for msg in traj["messages"]:
        ts = (msg.get("extra") or {}).get("timestamp")
        if ts is None:
            continue
        if msg["role"] == "assistant":
            pending = ts
        elif msg["role"] == "tool" and pending is not None:
            windows.append((pending, ts))
            pending = None
    return windows


def plot_task(ax: plt.Axes, run_dir: Path, *, ylabels: bool = True, legend: bool = True,
              title: str | None = None,
              title_fontsize: float = 9) -> tuple[list, list]:
    """Draw cpu + memory + tool-call spans for one run onto ax (x = s since agent start).

    Returns the ordered (handles, labels) for a cpu, memory, tool call legend.
    """
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    phase = status["phases"]["agent"]
    t0, t1 = _epoch(phase["start"]), _epoch(phase["end"])

    epochs, usage, mem, alloc_cpu, alloc_mem = load_samples(run_dir)
    pts = [(e, u, m) for e, u, m in zip(epochs, usage, mem) if t0 <= e <= t1]
    if len(pts) < 2:
        raise ValueError(f"{run_dir}: fewer than 2 metric samples inside the agent phase")
    # Raw per-interval ratios: no smoothing, no clipping. metrics.sh reads cpu.stat before
    # stamping the clock, so single intervals can overshoot 100% by a few percent; the fixed
    # 0-100 axis draws those past the top edge instead of hiding them.
    cpu = [100.0 * (b[1] - a[1]) / 1e6 / (b[0] - a[0]) / alloc_cpu for a, b in zip(pts, pts[1:])]
    xs = [(a[0] + b[0]) / 2 - t0 for a, b in zip(pts, pts[1:])]
    mem_pct = [100.0 * p[2] / (alloc_mem * 2**30) for p in pts]

    ax.plot(xs, cpu, color="tab:blue", lw=1, label=f"CPU (% of {alloc_cpu} cores)")
    ax.set_xlim(0, t1 - t0)
    ax.set_ylim(0, 100)
    if ylabels:
        ax.set_ylabel("CPU %", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
    else:
        ax.set_yticks([])

    ax2 = ax.twinx()
    ax2.plot([p[0] - t0 for p in pts], mem_pct, color="tab:green", lw=1,
             label=f"memory (% of {alloc_mem:g} GiB)")
    ax2.set_ylim(0, 100)
    if ylabels:
        ax2.set_ylabel("memory %", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
    else:
        ax2.set_yticks([])

    spans = [(s, e) for s, e in load_tool_windows(run_dir) if t0 <= s <= t1]
    for i, (s, e) in enumerate(spans):
        ax.axvspan(s - t0, max(e - t0, s - t0 + 0.5), color="red", alpha=0.15,
                   label="tool call" if i == 0 else None)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles = [h1[0], *h2, *h1[1:]]
    labels = [l1[0], *l2, *l1[1:]]
    if legend:
        ax.legend(handles, labels, loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    return handles, labels
