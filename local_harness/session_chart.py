"""One local_benchmark.py session (opencode or mini-swe-agent): CPU and memory, harness and tool containers
overlaid, on a 1 s grid. x = s since agent start.

Harness: metrics.jsonl (1 s cgroup samples). Tools: toolcalls.jsonl, one summary per container;
its CPU is spread evenly over the `docker run` window, its memory peak held over that window
(upper bound: most calls live ~1 s, too short for their own 1 s samples).

Usage: uv run python -m local_harness.session_chart opencode_tools/runs/qwen3.8-flash/sympy__sympy-15976
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared.plot import _epoch, load_samples


def cgroup_series(run_dir: Path):
    """(t0, cores, MiB) of metrics.jsonl per 1 s bin of the agent phase.

    Interpolated onto bin centres: samples land every ~1.0x s, so bucketing leaves empty bins.
    """
    phase = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["phases"]["agent"]
    t0, t1 = _epoch(phase["start"]), _epoch(phase["end"])
    epochs, usage, mem, _, _ = load_samples(run_dir)
    e, u = np.asarray(epochs) - t0, np.asarray(usage)
    centres = np.arange(int(np.ceil(t1 - t0))) + 0.5
    cpu = np.interp(centres, (e[1:] + e[:-1]) / 2, np.diff(u) / 1e6 / np.diff(e))
    return t0, cpu, np.interp(centres, e, np.asarray(mem) / 2**20)


# One colour per tool call, cycling; tab10 without its blue, which is the harness/sandbox.
TOOL_COLORS = plt.get_cmap("tab10").colors[1:]


def session_series(run_dir: Path):
    """(x, harness cores, harness MiB, calls) per 1 s bin of the agent phase.

    calls: per tool container, (bins, cores, MiB), padded with a 0 bin each side so it draws
    as a spike from the axis.
    """
    t0, h_cpu, h_mem = cgroup_series(run_dir)
    n, calls = len(h_cpu), []
    for line in (run_dir / "toolcalls.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if c.get("cpu_usec") is None:
            continue  # the wrapper never finished: no summary
        start, end = c["ts"] - c["wall_s"] - t0, c["ts"] - t0
        bins = np.arange(max(0, int(start)), min(n, int(np.ceil(end))))
        overlap = np.minimum(end, bins + 1) - np.maximum(start, bins)
        cpu = c["cpu_usec"] / 1e6 * overlap / (end - start)
        mem = np.full(len(bins), c["memory_peak"] / 2**20)
        pad = lambda a, lo, hi: np.concatenate(([lo], a, [hi]))
        calls.append((pad(bins, bins[0] - 1, bins[-1] + 1), pad(cpu, 0, 0), pad(mem, 0, 0)))
    return np.arange(n), h_cpu, h_mem, calls


def plot_session(ax_cpu: plt.Axes, ax_mem: plt.Axes, run_dir: Path) -> None:
    """Harness line and one line per tool container (own colour), overlaid from 0."""
    x, h_cpu, h_mem, calls = session_series(run_dir)
    for ax, harness, k in ((ax_cpu, h_cpu, 1), (ax_mem, h_mem, 2)):
        ax.plot(x, harness, color="tab:blue", lw=2.2, marker=".", markersize=4,
                label="harness (agent)")
        for i, call in enumerate(calls):
            ax.plot(call[0], call[k], color=TOOL_COLORS[i % len(TOOL_COLORS)], lw=0.6,
                    marker=".", markersize=2,
                    label="tool containers (one colour per call)" if i == 0 else None)
        ax.set_xlim(0, len(x))
        ax.grid(alpha=0.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="default: <exp>/images/<model>-<task>.png")
    args = ap.parse_args()
    run_dir = args.run_dir
    # <exp>/runs/<model>/<task>
    exp_dir = run_dir.parents[2]
    out = args.out or exp_dir / "images" / f"{run_dir.parent.name}-{run_dir.name}.png"

    fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    plot_session(ax_cpu, ax_mem, run_dir)
    ax_cpu.set_ylabel("CPU (cores)")
    ax_mem.set_ylabel("memory (MiB)")
    ax_cpu.set_ylim(bottom=0)
    ax_mem.set_ylim(bottom=0)
    ax_cpu.legend(loc="upper right")
    ax_cpu.set_title(f"{exp_dir.name} / {run_dir.parent.name} / {run_dir.name}")
    ax_mem.set_xlabel("s since agent start")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
