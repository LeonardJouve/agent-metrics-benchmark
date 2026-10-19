"""Per-task resource timeline: CPU + memory vs agent session, tool calls shaded red.

Usage: uv run python -m miniswe_daytona.charts.task_chart miniswe_daytona/runs/qwen3.8-flash/astropy__astropy-7166
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from miniswe_daytona.charts.taskplot import plot_task


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    run_dir: Path = args.run_dir
    # <exp>/runs/<model>/<task> -> <exp>/images/<model>-<task>.png
    out = args.out or run_dir.parents[2] / "images" / f"{run_dir.parent.name}-{run_dir.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_task(ax, run_dir, title=f"{status['model_name']} / {status['instance_id']}")
    ax.set_xlabel("elapsed time [s]")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
