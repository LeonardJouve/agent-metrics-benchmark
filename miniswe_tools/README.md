# Experiment 3: mini-swe-agent, per-tool system metrics (local Docker)

**Question:** with the same agent as [experiment 1](../miniswe_daytona/README.md), how much does the agent cost alone, and how much does each tool call cost?

**Answer:** the agent alone is cheap, about 210 MiB and a few hundredths of a core on average. Nearly all the cost is in tool calls, and most calls are tiny: median 0.04 s and 17 MiB. A few calls (test suites, `pip install`) hit the 1 GiB per-call cap.

## How it works

mini-swe-agent 2.4.6 runs in the [local harness](../local_harness/README.md). Each bash call runs in its own container from the task image, on a shared `/testbed` volume, so agent and tools are measured separately. The agent config is experiment 1's (`shared/runner.py` `make_agent_config`), with one change.

Specific to this experiment: `mini_env.py`. It sets `environment_class: mini_env.ToolContainerEnvironment`, mini-swe-agent's local environment with every command sent to the tool client. The 60 s timeout, output handling and submission handling stay mini-swe-agent's own.

Shared: `local_harness/` (runner, harness image, tool client, charts) and `shared/` (dataset, pool, status, mini-swe-agent config, `metrics.sh`).

## Run

```bash
uv run python -m local_harness.local_benchmark miniswe   # from the repo root; Docker running
uv run python -m local_harness.session_grid --agent miniswe
uv run python -m local_harness.session_chart miniswe_tools/runs/qwen3.8-flash/pytest-dev__pytest-5809
```

Artifacts: `runs/<model>/<instance_id>/` (see the [local harness](../local_harness/README.md#artifacts)), plus `trajectory.traj.json`. `agent.log` is mini-swe-agent's output. `runs/full.log` holds the matrix summary.

## Findings

From `uv run python -m shared.summary`, agent phase only.

Harness (the agent alone, 1 vCPU / 2 GiB):

| model | runs ok | agent time median (max) | tool calls median | CPU cores mean / p95 / max | memory peak median (max) |
|---|---|---|---|---|---|
| qwen3.8-flash | 20/20 | 8.6 min (24.7) | 50 | 0.02 / 0.05 / 1.00 | 212 MiB (215) |
| deepseek-v4-flash-0731 | 20/20 | 4.1 min (11.9) | 42 | 0.04 / 0.07 / 1.00 | 211 MiB (218) |

Tool containers (one per call, 1 vCPU / 1 GiB):

| model | tool calls | command time median / p95 | `docker run` wall median | CPU per call median / p95 | memory peak per call median / p95 / max | agent time inside commands |
|---|---|---|---|---|---|---|
| qwen3.8-flash | 1094 | 0.04 s / 20.9 s | 0.97 s | 0.27 / 10.8 core-s | 17 / 213 / 1024 MiB | 24 % |
| deepseek-v4-flash-0731 | 849 | 0.02 s / 2.9 s | 0.97 s | 0.26 / 2.9 core-s | 17 / 85 / 1024 MiB | 12 % |

- **The agent is negligible next to its tools.** The harness holds steady at about 210 MiB. Compared with experiment 1, where the whole sandbox peaked at about 1.3 GiB, almost all of that peak came from tools and page cache.
- **Tool calls are heavy-tailed.** Half the calls finish in hundredths of a second and peak at 17 MiB. The p95 is 85–213 MiB. Test suites (django, xarray) and a `pip install` hit the 1 GiB cap without being killed (page cache was reclaimed).
- **Container start dominates small calls.** `docker run` takes 0.97 s median end to end, against a median command of 0.02–0.04 s.
- Results are close to experiment 1's: same agent and tasks, similar call counts (50 vs 58 median with Qwen), so the per-tool split did not change the agent's behaviour much.

Example session, harness and one colour per tool container ([all tasks](images/compare-grid.png)):

![qwen3.8-flash pytest-dev__pytest-5809](images/qwen3.8-flash-pytest-dev__pytest-5809.png)

## Limits

- Local Docker on one machine (Docker Desktop on Windows), 8 runs in parallel: absolute times include host contention.
- State outside `/testbed` (pip installs, `/tmp`) is lost between calls. The agent sees a slightly different environment than in experiment 1.
- Most calls end before their first 1 s sample: per-call CPU and memory come from the container's cgroup summary (`cpu_usec`, `memory_peak`), not from the time series.
