# Agent System Metrics Benchmark

**Question:** how much CPU and RAM does a coding agent use over a task, and how much of that comes from its tool calls rather than the agent itself? The answer sizes the sandbox a coding agent needs, and shows whether sandboxes could be sized per tool call instead of per session.

Three experiments, same 20 tasks and two models:

| # | Folder | Agent | Where | What is measured |
|---|---|---|---|---|
| 1 | [`miniswe_daytona/`](miniswe_daytona/README.md) | mini-swe-agent 2.4.6 | Daytona sandbox, 4 vCPU / 4 GiB | whole sandbox (agent + tools together), 1 s samples |
| 2 | [`opencode_tools/`](opencode_tools/README.md) | opencode 1.19.2 | local Docker, one container per bash call | harness (agent) and every tool call separately |
| 3 | [`miniswe_tools/`](miniswe_tools/README.md) | mini-swe-agent 2.4.6 | local Docker, one container per bash call | harness (agent) and every tool call separately |

Experiments 1 and 3 run the same agent with the same config, so comparing them shows what the per-tool split changes. Experiments 2 and 3 share one harness, so comparing them shows what the agent changes.

## Findings

Figures from `uv run python -m shared.summary` (agent phase only, 40 runs per experiment, all successful, none censored). Each experiment README has the full tables.

1. **Usage is bursty and tied to tool calls.** In experiment 1 the whole sandbox averages 0.15–0.51 of 4 vCPU but reaches all 4 in bursts (p95 3.99 cores with Qwen). The bursts line up with tool calls ([tool binding](miniswe_daytona/images/tool-binding.png)). The distribution is long-tailed ([CCDF](miniswe_daytona/images/usage-ccdf.png)): high levels are rare, but they set the size.
2. **Most tool calls are tiny; a few dominate.** Per call (experiments 2 and 3), the median command runs in 0.01–0.04 s and peaks at 17 MiB. The p95 is 3–21 s and 85–213 MiB. Test suites and `pip install` reach the 1 GiB tool-container cap.
3. **The agent's own footprint depends heavily on the harness.** Measured alone, mini-swe-agent uses about 210 MiB and 0.02–0.04 cores on average. opencode uses about 1 GiB, 0.4–0.5 cores on average, and regularly saturates its 1 vCPU. Sizing per tool call only pays off if the harness is light.
4. **Commands occupy a small share of the session.** Commands run during 8–24 % of the agent phase. The rest is mostly model time (see [tool gaps](miniswe_daytona/images/tool-gaps.png)).
5. **A fresh container per call costs about 1 s.** `docker run` takes a median 0.97–1.42 s end to end, against a median command time of 0.01–0.04 s. Per-call isolation multiplies latency for the common tiny call, so warm pools or reuse are needed.
6. **The model matters as much as the task.** On the same tasks, Qwen3.8 Flash runs 2–3× longer than DeepSeek V4 Flash (median agent time) and makes more tool calls (median 58 vs 34 in experiment 1).

Limits: 20 short Python tasks, one run per task and model, no rerun. Patches are not evaluated, so there is no resolve rate; every run produced a non-empty patch. `memory_current` includes page cache, so memory peaks are upper bounds on what the agent needs. Experiment 1's RAM is capped at 4 GiB (Daytona org limit), so higher demand would be censored; no OOM kill was observed.

## Layout

```
shared/              used by all three experiments
  runner.py          dataset loading, task × model pool, status.json/phases, MODELS, mini-swe-agent config,
                     metrics.sh paths and summary, 45 min cap
  metrics.sh         cgroup v2 collector, 1 s samples → metrics.jsonl
  plot.py            chart helpers: load metrics.jsonl, dataset order
  summary.py         the figures quoted in the READMEs
  dataset/           build_dataset.py → sample20.csv (the 20 tasks); one.csv (1-task smoke test)
local_harness/       shared by experiments 2 and 3 only
  local_benchmark.py runner: harness container + tool server, `opencode` or `miniswe`
  harness.Dockerfile both agents in one image
  tool_client.py     the agents' bash: forwards each command to a fresh tool container
  session_chart.py   one session: harness + tool containers
  session_grid.py    task × model grid
miniswe_daytona/     experiment 1: benchmark.py, charts/, runs/, images/
opencode_tools/      experiment 2: opencode.json, runs/, images/
miniswe_tools/       experiment 3: mini_env.py, runs/, images/
tests/               test_runner.py (shared), test_metrics_script.py (shared), test_build_dataset.py (shared),
                     test_benchmark.py (exp 1), test_tool_server.py (exps 2+3)
```

Each experiment stores runs as `<exp>/runs/<model>/<instance_id>/` and charts in `<exp>/images/`.

## Common conditions

- Tasks: 20 SWE-bench Verified Python tasks (config `default`, split `test`; seed 42, round-robin by repo; only <1h-rated, Docker Hub image ≤2 GB).
- Models: Qwen3.8 Flash + DeepSeek V4 Flash via the Qwen token plan → 20 × 2 = 40 runs per experiment, 1 run per task and model, no reruns.
- Run cap: 45 min, censored if hit. Sampling: 1 s, cgroup v2 (`shared/metrics.sh`: CPU usage, memory current/peak, throttling, OOM events).

## Config

Env vars only (copy `.env.example` to `.env`, export it):

| Variable | Meaning |
|---|---|
| `DATASET_PATH` | Dataset CSV; default `shared/dataset/sample20.csv`. All rows × all models run. |
| `BATCH_SIZE` | Concurrent runs; default `2`. |
| `QWEN_TOKEN_PLAN_API_KEY` | Required by all experiments. |
| `DAYTONA_API_KEY` | Required by experiment 1. |
| `MODELS` | Experiments 2 and 3 only: `a,b` limits the models. |

## Run

Every command runs from this folder (the repo root):

```bash
set -a; . ./.env; set +a                                  # export .env
uv run python -m miniswe_daytona.benchmark                # experiment 1
uv run python -m local_harness.local_benchmark opencode   # experiment 2 (Docker running)
uv run python -m local_harness.local_benchmark miniswe    # experiment 3 (Docker running)
uv run python -m shared.summary                           # figures for the READMEs
uv run python -m shared.dataset.build_dataset             # rebuild the dataset
uv run pytest
```

Exit code: `0` all runs OK; `1` any fail/censor/cleanup fail. No resume; a rerun overwrites.

Every run writes `status.json` (phases, exit codes, outcome, censoring), `install.log`, `agent.log`, `patch.diff` and `metrics.jsonl`. Each experiment README lists its extra files.
