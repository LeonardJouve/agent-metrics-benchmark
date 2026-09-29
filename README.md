# Agent System Metrics Benchmark

Benchmark of agent runtime resource use (CPU/RAM). Measures Python coding agents on Daytona sandboxes. Output: data for CPU/RAM allocation.

## Conditions

- Tasks: 20 SWE-bench Verified Python tasks (config `default`, split `test`; seed 42, round-robin by repo; only <1h-rated, Docker Hub image ≤2 GB).
- Models: Qwen3.8 Flash + DeepSeek V4 Flash → matrix 20 × 2 = 40 runs, 1 run/task/model, no reruns.
- Sandbox: 4 vCPU / 4 GiB / 10 GiB disk.
- Concurrency: 2 sandboxes (`BATCH_SIZE`). Sampling: 1s. Run cap: 45 min, censored if hit.
- Cleanup: export artifacts, then delete sandbox.

## Tools

- mini-SWE-agent 2.4.6 — in-sandbox `/opt/minisweagent`, `local` mode, no nested Docker
- Daytona Cloud — sandbox runner/profiler
- Python + uv — `benchmark.py` (matrix runner), `build_dataset.py` (dataset builder)
- cgroup v2 collector — `metrics.sh` (1s sampling)

## Config

Env vars only (copy `.env.example` to `.env`, export it):

| Variable | Meaning |
|---|---|
| `DATASET_PATH` | Dataset CSV; default `./data/sample20.csv`. All rows × all models run. |
| `BATCH_SIZE` | Concurrent sandboxes; default `2`. |
| `DAYTONA_API_KEY` | Required; runner fails before creating sandboxes if missing. |
| `QWEN_TOKEN_PLAN_API_KEY` | Required; provisions Daytona secret `qwen-token-plan` → sandbox env `OPENAI_API_KEY`, host allowlist `token-plan.ap-southeast-1.maas.aliyuncs.com`. |

## Run

```bash
set -a; . ./.env; set +a   # export .env
uv run python benchmark.py
```

Rebuild dataset: `uv run python build_dataset.py`.

Artifacts: `runs/<model>/<instance_id>/` — `install.log`, `agent.log`, `trajectory.traj.json`, `patch.diff`, `metrics.jsonl`, `status.json`.
Exit code: `0` all runs OK; `1` any fail/censor/cleanup fail. No resume; rerun overwrites.