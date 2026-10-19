# Agent System Metrics Benchmark

Benchmark of coding-agent runtime resource use (CPU/RAM), to size CPU/RAM allocation. Three experiments, same tasks and models:

| # | Folder | Agent | Where | What is measured |
|---|---|---|---|---|
| 1 | [`miniswe_daytona/`](miniswe_daytona/README.md) | mini-swe-agent 2.4.6 | Daytona sandbox, 4 vCPU / 4 GiB | whole sandbox (agent + tools together), 1 s samples |
| 2 | [`opencode_tools/`](opencode_tools/README.md) | opencode 1.19.2 | local Docker, one container per bash call | harness (agent) and every tool call separately |
| 3 | [`miniswe_tools/`](miniswe_tools/README.md) | mini-swe-agent 2.4.6 | local Docker, one container per bash call | harness (agent) and every tool call separately |

Experiments 1 and 3 run the same agent with the same config, so comparing them shows what the per-tool split changes. Experiments 2 and 3 share one harness, so comparing them shows what the agent changes.

## Layout

```
shared/              used by all three experiments
  runner.py          dataset loading, task × model pool, status.json/phases, MODELS, mini-swe-agent config,
                     metrics.sh paths and summary, 45 min cap
  metrics.sh         cgroup v2 collector, 1 s samples → metrics.jsonl
  plot.py            chart helpers: load metrics.jsonl, dataset order
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
- Run cap: 45 min, censored if hit. Sampling: 1 s.

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
set -a; . ./.env; set +a                              # export .env
uv run python -m miniswe_daytona.benchmark            # experiment 1
uv run python -m local_harness.local_benchmark opencode   # experiment 2 (Docker running)
uv run python -m local_harness.local_benchmark miniswe    # experiment 3 (Docker running)
uv run python -m shared.dataset.build_dataset         # rebuild the dataset
uv run pytest
```

Exit code: `0` all runs OK; `1` any fail/censor/cleanup fail. No resume; a rerun overwrites.

Every run writes `status.json` (phases, exit codes, outcome, censoring), `install.log`, `agent.log`, `patch.diff` and `metrics.jsonl`. Each experiment README lists its extra files.
