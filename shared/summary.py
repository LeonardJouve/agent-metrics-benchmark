"""Per experiment and model: the figures quoted in the experiment READMEs.

Agent phase only (status.json phases.agent). "Sandbox/harness" is metrics.jsonl: the whole
Daytona sandbox in experiment 1, the harness container (agent alone) in experiments 2 and 3.
"Tool" columns come from toolcalls.jsonl (one summary per tool container, experiments 2 and 3).

Usage: uv run python -m shared.summary
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from shared.plot import _epoch
from shared.runner import MODELS

EXPERIMENTS = ["miniswe_daytona", "opencode_tools", "miniswe_tools"]
MIB = 2**20


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run_stats(run_dir: Path) -> dict:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    phase = status["phases"]["agent"]
    t0, t1 = _epoch(phase["start"]), _epoch(phase["end"])
    samples = [json.loads(line) for line in (run_dir / "metrics.jsonl").open(encoding="utf-8")]
    samples = [s for s in samples if t0 <= s["epoch"] <= t1]
    cores = [(b["usage_usec"] - a["usage_usec"]) / 1e6 / (b["epoch"] - a["epoch"])
             for a, b in zip(samples, samples[1:]) if b["epoch"] > a["epoch"]]
    out = {
        "ok": status["outcome"] == "success",
        "censored": status.get("censored", False),
        "patch": (run_dir / "patch.diff").exists() and (run_dir / "patch.diff").stat().st_size > 0,
        "duration_s": t1 - t0,
        "cores": cores,
        "mem_peak_mib": max(s["memory_current"] for s in samples) / MIB,
        "oom_kill": max(s.get("oom_kill", 0) for s in samples),
    }
    calls_path = run_dir / "toolcalls.jsonl"
    if calls_path.exists():
        calls = [json.loads(line) for line in calls_path.open(encoding="utf-8")]
        out["calls"] = calls
        out["tool_calls"] = len(calls)
    else:  # experiment 1: count tool messages in the trajectory
        traj = json.loads((run_dir / "trajectory.traj.json").read_text(encoding="utf-8"))
        out["tool_calls"] = sum(m["role"] == "tool" for m in traj["messages"])
    return out


def main() -> int:
    for exp in EXPERIMENTS:
        print(f"\n## {exp}\n")
        print("| model | runs ok | censored | non-empty patch | agent time median (max) | tool calls median "
              "| CPU cores mean / p95 / max | memory peak median (max) | OOM kills |")
        print("|---|---|---|---|---|---|---|---|---|")
        tool_rows = []
        for model in MODELS:
            runs = [run_stats(d) for d in sorted((Path(exp) / "runs" / model).iterdir()) if d.is_dir()]
            cores = [c for r in runs for c in r["cores"]]
            durations = [r["duration_s"] for r in runs]
            peaks = [r["mem_peak_mib"] for r in runs]
            print(f"| {model} | {sum(r['ok'] for r in runs)}/{len(runs)} | {sum(r['censored'] for r in runs)} "
                  f"| {sum(r['patch'] for r in runs)} "
                  f"| {statistics.median(durations) / 60:.1f} min ({max(durations) / 60:.1f}) "
                  f"| {statistics.median(r['tool_calls'] for r in runs):.0f} "
                  f"| {statistics.fmean(cores):.2f} / {pct(cores, 0.95):.2f} / {max(cores):.2f} "
                  f"| {statistics.median(peaks):.0f} MiB ({max(peaks):.0f}) "
                  f"| {sum(r['oom_kill'] for r in runs)} |")
            calls = [c for r in runs for c in r.get("calls", [])]
            if calls:
                done = [c for c in calls if c.get("memory_peak")]
                mem = [c["memory_peak"] / MIB for c in done]
                exec_s = [c["exec_end"] - c["exec_start"] for c in done]
                cpu_s = [c["cpu_usec"] / 1e6 for c in done]
                wall = [c["wall_s"] for c in calls]
                in_tools = sum(exec_s) / sum(durations)
                tool_rows.append(
                    f"| {model} | {len(calls)} | {statistics.median(exec_s):.2f} s / {pct(exec_s, 0.95):.1f} s "
                    f"| {statistics.median(wall):.2f} s | {statistics.median(cpu_s):.2f} / {pct(cpu_s, 0.95):.1f} core-s "
                    f"| {statistics.median(mem):.0f} / {pct(mem, 0.95):.0f} / {max(mem):.0f} MiB | {in_tools:.0%} |")
        if tool_rows:
            print("\n| model | tool calls | command time median / p95 | docker run wall median "
                  "| CPU per call median / p95 | memory peak per call median / p95 / max | agent time inside commands |")
            print("|---|---|---|---|---|---|---|")
            print("\n".join(tool_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
