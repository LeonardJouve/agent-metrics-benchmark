"""Mean gap between a tool response and the next tool call (agent thinking + LLM time).

Per run, per consecutive pair (tool response -> next assistant tool submission) the gap is
trajectory.traj.json extra.timestamp deltas. One stacked figure: one panel per model,
qwen on top, deepseek below. Bars = per-task mean gap, error bars = std, dashed = model mean.

Usage: uv run python charts/tool_gaps.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_MODELS = ["qwen3.8-flash", "deepseek-v4-flash-0731"]


def tool_gaps(run_dir: Path) -> list[float]:
    """Seconds from each tool response to the next assistant tool submission (empty if none)."""
    traj = json.loads((run_dir / "trajectory.traj.json").read_text(encoding="utf-8"))
    gaps, last_tool = [], None
    for msg in traj["messages"]:
        ts = (msg.get("extra") or {}).get("timestamp")
        if ts is None:
            continue
        if msg["role"] == "assistant":
            if last_tool is not None:
                gaps.append(ts - last_tool)
            last_tool = None
        elif msg["role"] == "tool":
            last_tool = ts
    return gaps


def collect(model_dir: Path) -> list[tuple[str, list[float]]]:
    out = []
    for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        if not (run_dir / "trajectory.traj.json").exists():
            continue
        gaps = tool_gaps(run_dir)
        if gaps:
            out.append((run_dir.name, gaps))
    return out


def _panel(ax: plt.Axes, model: str, tasks: list[tuple[str, list[float]]]) -> None:
    xs = range(len(tasks))
    means = [sum(g) / len(g) for _, g in tasks]
    stds = [(sum((s - means[i]) ** 2 for s in g) / len(g)) ** 0.5 for i, (_, g) in enumerate(tasks)]
    overall = sum(means) / len(means)

    ax.bar(xs, means, yerr=stds, capsize=3, color="tab:blue", alpha=0.75,
           error_kw=dict(ecolor="0.4"))
    ax.axhline(overall, color="tab:red", ls="--", lw=1, label=f"mean gap {overall:.0f}s")
    ax.set_xticks(xs, [name.split("__")[-1] for name, _ in tasks],
                  rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("s (tool response → next tool call)")
    ax.set_title(model, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("-o", "--out", type=Path, default=Path("images/tool-gaps.png"))
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    fig, axes = plt.subplots(len(args.models), 1, figsize=(11, 2.8 * len(args.models)),
                             sharex=True)
    for ax, model in zip(axes, args.models):
        _panel(ax, model, collect(args.runs / model))

    fig.suptitle("Mean gap between tool response and next tool call", y=0.99, fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())