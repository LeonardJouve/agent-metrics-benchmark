"""Schema: CPU and memory usage vs a static allocation and vs dynamic vertical autoscaling.

One panel per resource, shared time axis, tool calls drawn once. Real data, smoothed; memory shown above
its baseline footprint, the window's highest burst = 100% (schematic). The autoscaler is idealised: stepped, rounded up to
a resize unit, sized on the usage peak of the surrounding seconds (it sees the future).

Usage: uv run python -m miniswe_daytona.charts.slack_schema miniswe_daytona/runs/qwen3.8-flash/pydata__xarray-6721
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from numpy.lib.stride_tricks import sliding_window_view

from miniswe_daytona.charts.taskplot import load_tool_windows
from shared.plot import _epoch, load_samples

CPU, MEM, LIMIT, TOOL, FREE = "#2563eb", "#f59e0b", "#111827", "#ef4444", "#10b981"
STEP, WIN = 0.4, 1  # resample step [s], smoothing window [points] (1 = raw)
MARGIN = 1.25  # headroom over the observed peak
MIN_STEP = 8.0  # shortest allowed allocation step [s]
LOOK, HOLD, UNIT = 18.0, 12.0, 12.5  #autoscaler look-around [s], resize unit [% of limit] (0.5 core / 0.5 GiB)


def smooth(x: np.ndarray, win: int = WIN) -> np.ndarray:
    k = np.hanning(win); k /= k.sum()
    return np.convolve(np.pad(x, win // 2, mode="edge"), k, mode="valid")


def autoscale(x: np.ndarray) -> np.ndarray:
    """Sharp allocation: running max over +-LOOK s, rounded up to UNIT steps."""
    win = lambda v, s: sliding_window_view(np.pad(v, int(s / STEP), mode="edge"), 2 * int(s / STEP) + 1)
    peak = np.minimum(np.ceil(win(x, LOOK).max(axis=1) * MARGIN / UNIT) * UNIT, 100)
    return merge_short(win(peak, HOLD).min(axis=1))  # closing: no short dips


def merge_short(y: np.ndarray) -> np.ndarray:
    """Merge steps shorter than MIN_STEP s into a neighbour, only ever raising the allocation (never below usage)."""
    n = int(MIN_STEP / STEP)
    y = y.copy()
    while True:
        edges = np.flatnonzero(np.diff(y)) + 1
        b = np.concatenate([[0], edges, [len(y)]])
        lens = np.diff(b)
        short = [i for i in np.argsort(lens) if lens[i] < n and len(lens) > 1]
        if not short:
            return y
        i = short[0]
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(lens)]
        v = y[b[i]]
        if v < max(y[b[j]] for j in nb):  # dip: take the higher neighbour
            y[b[i]:b[i + 1]] = max(y[b[j]] for j in nb)
        else:  # spike: raise the shorter neighbour to it
            j = min(nb, key=lambda j: lens[j])
            y[b[j]:b[j + 1]] = v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--window", type=float, nargs=2, default=(330, 560), metavar=("START", "END"),
                    help="seconds since agent start")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    run_dir = args.run_dir
    out = args.out or run_dir.parents[2] / "images" / "slack-schema.png"
    a, b = args.window

    ph = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["phases"]["agent"]
    t0, t1 = _epoch(ph["start"]), _epoch(ph["end"])
    ep, usage, mem, cores, gib = load_samples(run_dir)
    pts = [(e, u, m) for e, u, m in zip(ep, usage, mem) if t0 <= e <= t1]
    cpu_x = np.array([(p[0] + q[0]) / 2 - t0 for p, q in zip(pts, pts[1:])])
    cpu_y = np.array([100 * (q[1] - p[1]) / 1e6 / (q[0] - p[0]) / cores for p, q in zip(pts, pts[1:])])
    mem_x = np.array([p[0] - t0 for p in pts])
    mem_y = np.array([100 * p[2] / (gib * 2**30) for p in pts])

    xf = np.arange(0, t1 - t0, STEP)  # full run, then crop: no edge artefacts in the window
    cpu = np.clip(smooth(np.interp(xf, cpu_x, cpu_y)), 0, 100)
    memo = smooth(np.interp(xf, mem_x, mem_y))
    sel = (xf >= a) & (xf <= b)
    lo, hi = memo[sel].min(), memo[sel].max()  # window range: drop the footprint, highest burst = 100%
    memo = np.clip(100 * (memo - lo) / (hi - lo), 0, 100)
    x = xf[sel]
    spans = [(s - t0, max(e - t0, s - t0 + 0.5)) for s, e in load_tool_windows(run_dir) if a <= e - t0 and s - t0 <= b]

    fig, axs = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True,
                            gridspec_kw={"height_ratios": [5, 5, 0.7], "hspace": 0.12})
    for ax, name, color, full in zip(axs, ("CPU", "Memory"), (CPU, MEM), (cpu, memo)):
        y, dyn = full[sel], autoscale(full)[sel]
        ax.fill_between(x, y, dyn, where=dyn >= y, color=color, alpha=0.2, lw=0)  # slack left after autoscaling
        ax.fill_between(x, dyn, 100, color=FREE, alpha=0.25, lw=0)  # reclaimed
        ax.axhline(100, color=LIMIT, lw=1)
        ax.step(x, dyn, where="mid", color=LIMIT, lw=1)
        ax.plot(x, y, color=color, lw=1.8)
        ax.set_ylabel(name, color=color, fontsize=14, weight="bold")
        ax.set_ylim(0, 104)
        tr = ax.get_yaxis_transform()
        if name == "CPU":
            ax.text(1.02, 100, "static size", transform=tr, va="center", color=LIMIT, fontsize=11)
            ax.text(1.02, dyn[-1], "autoscaled", transform=tr, va="center", color=LIMIT, fontsize=11)
            sq = lambda c, al: Patch(facecolor=tuple(al * np.array(to_rgb(c)) + (1 - al)), lw=0)  # opaque, as on white
            ax.legend([sq(FREE, 0.25), (sq(CPU, 0.2), sq(MEM, 0.2))], ["reclaimed", "slack"], loc="upper left",
                      facecolor="white", edgecolor="none", framealpha=1, fontsize=11, handlelength=2, handleheight=1,
                      handler_map={tuple: HandlerTuple(ndivide=2, pad=0)})

    ax = axs[2]
    for s, e in spans:
        ax.fill_between([max(s, a), min(e, b)], 0, 1, color=TOOL, lw=0)
    ax.text(1.02, 0.5, "tool calls", transform=ax.get_yaxis_transform(), va="center", color=TOOL, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_xlim(a, b)
    ax.set_xlabel("time", fontsize=12)

    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
