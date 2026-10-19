# Experiment 1: mini-swe-agent, whole-sandbox system metrics (Daytona)

**Question:** what does one coding-agent session cost a sandbox in CPU and RAM over time, and when do the peaks happen?

**Answer:** usage is low on average and bursty. The sandbox idles while the model thinks and spikes during tool calls, sometimes up to the full 4 vCPU. The distribution is long-tailed: rare peaks set the size.

## How it works

```
runner (benchmark.py) ──Daytona SDK──▶ sandbox from the task image (4 vCPU / 4 GiB / 10 GiB)
                                        ├─ metrics.sh: whole-sandbox cgroup, 1 s samples
                                        └─ mini-swe-agent (local mode) ──▶ bash in the same sandbox
```

One Daytona sandbox per task, created from the SWE-bench task image. mini-swe-agent 2.4.6 is installed in `/opt/minisweagent` and runs its commands in `local` mode, with no nested Docker. `shared/metrics.sh` samples the whole sandbox cgroup, so agent and tools are measured together. Tool-call windows come from the trajectory's timestamps.

Phases in `status.json`: create → install (mini-swe-agent) → agent → export (patch, trajectory, metrics) → cleanup (delete the sandbox). The metrics cover every phase; the charts and figures keep only the agent phase.

- Concurrency: 2 sandboxes. The Daytona org caps total concurrent use at 10 vCPU / 10 GiB, which is why memory is 4 GiB: RAM demand above 4 GiB would be censored, visible only as OOM counters.
- `QWEN_TOKEN_PLAN_API_KEY` is provisioned as the Daytona secret `qwen-token-plan` → sandbox env `OPENAI_API_KEY`, host allowlist `token-plan.ap-southeast-1.maas.aliyuncs.com`. The key never reaches config files, logs or `status.json`.

Uses from `shared/`: `runner.py` (dataset, pool, status, mini-swe-agent config), `metrics.sh`, `plot.py`, `dataset/sample20.csv`.

## Files

| File | Role |
|---|---|
| `benchmark.py` | Daytona runner: secret, sandbox lifecycle, metrics collector, export |
| `charts/taskplot.py` | one task's timeline onto an Axes; tool windows from the trajectory |
| `charts/task_chart.py`, `charts/compare_grid.py` | one task; task × model grid |
| `charts/model_boxplot.py`, `charts/model_ccdf.py` | usage distributions per model, all samples pooled |
| `charts/tool_binding.py`, `charts/tool_gaps.py` | usage inside vs outside tool calls; time between calls |
| `runs/<model>/<instance_id>/` | `install.log`, `agent.log`, `trajectory.traj.json`, `patch.diff`, `metrics.jsonl`, `status.json` |

## Run

```bash
uv run python -m miniswe_daytona.benchmark     # from the repo root; needs DAYTONA_API_KEY and QWEN_TOKEN_PLAN_API_KEY
```

Charts:

```bash
uv run python -m miniswe_daytona.charts.task_chart miniswe_daytona/runs/qwen3.8-flash/astropy__astropy-7166
uv run python -m miniswe_daytona.charts.compare_grid   # task × model grid
uv run python -m miniswe_daytona.charts.model_boxplot  # CPU/RAM distribution per model
uv run python -m miniswe_daytona.charts.model_ccdf     # % of samples needing ≥ x % of the allocation
uv run python -m miniswe_daytona.charts.tool_binding   # usage inside vs outside tool calls
uv run python -m miniswe_daytona.charts.tool_gaps      # time between a tool response and the next call
```

## Findings

From `uv run python -m shared.summary`, agent phase only. CPU in cores of the 4 allocated; memory is the sandbox's `memory_current` peak, page cache included.

| model | runs ok | agent time median (max) | tool calls median | CPU cores mean / p95 / max | memory peak median (max) | OOM kills |
|---|---|---|---|---|---|---|
| qwen3.8-flash | 20/20 | 7.8 min (23.5) | 58 | 0.51 / 3.99 / 4.34 | 1289 MiB (2311) | 0 |
| deepseek-v4-flash-0731 | 20/20 | 2.7 min (8.8) | 34 | 0.15 / 1.01 / 4.03 | 1252 MiB (1956) | 0 |

(A max above 4 cores is a sampling artefact: two 1 s samples a little less than 1 s apart.)

- **Bursts follow tool calls.** Above its idle level, usage happens inside tool-call windows ([tool binding](images/tool-binding.png)), e.g. test runs. Between calls the sandbox waits for the model ([tool gaps](images/tool-gaps.png)).
- **Long tail.** A 4 vCPU / 4 GiB sandbox is mostly idle; the few seconds near the top decide the allocation ([CCDF](images/usage-ccdf.png), [boxplot](images/usage-boxplot.png)).
- **The model changes the profile.** Qwen works about 3× longer and makes more calls on the same tasks, with a much heavier CPU tail (p95 3.99 vs 1.01 cores).

Example task profile, tool calls shaded ([all tasks](images/compare-grid.png)):

![qwen3.8-flash scikit-learn__scikit-learn-14629](images/qwen3.8-flash-scikit-learn__scikit-learn-14629.png)

## Limits

- Agent and tools share one cgroup: what the agent costs alone is not separable here (experiment 3 does that).
- Tool-call windows come from trajectory timestamps, stamped when messages are appended: approximate to about the model's response time.
- Memory is capped at 4 GiB, and `memory_current` counts page cache: peaks are upper bounds and could not exceed 4 GiB.
