# Experiment 2: opencode, per-tool system metrics (local Docker)

**Question:** does a different, richer coding agent change the picture from [experiment 3](../miniswe_tools/README.md)? The tools are the same and only the agent changes.

**Answer:** the tool calls look alike: median 0.01–0.02 s and 17 MiB per call, with a heavy tail. The agent does not. The opencode harness alone uses about 1 GiB and 0.4–0.5 cores on average, against about 210 MiB and 0.03 cores for mini-swe-agent. It regularly saturates its 1 vCPU.

## How it works

opencode 1.19.2 runs in the [local harness](../local_harness/README.md). Each bash call runs in its own container from the task image, on a shared `/testbed` volume, so agent and tools are measured separately.

Specific to this experiment: `opencode.json`, which does three things:
- defines the provider `qwen-token-plan`, the same token plan as the other experiments;
- denies every tool but bash, so the model never sees the others and every action goes through a measured container;
- sets opencode's `shell` to the tool client.

The prompt asks for a fix in `/testbed` without touching tests (`PROMPT` in `local_benchmark.py`).

Shared: `local_harness/` (runner, harness image, tool client, charts) and `shared/` (dataset, pool, status, `metrics.sh`).

## Run

```bash
uv run python -m local_harness.local_benchmark opencode   # from the repo root; Docker running
uv run python -m local_harness.session_grid --agent opencode
uv run python -m local_harness.session_chart opencode_tools/runs/qwen3.8-flash/pytest-dev__pytest-5809
```

Artifacts: `runs/<model>/<instance_id>/` (see the [local harness](../local_harness/README.md#artifacts)). `agent.log` is opencode's JSON event stream (`--format json`), which also serves as the trajectory. `runs/full.log` holds the matrix summary.

## Findings

From `uv run python -m shared.summary`, agent phase only.

Harness (the agent alone, 1 vCPU / 2 GiB):

| model | runs ok | agent time median (max) | tool calls median | CPU cores mean / p95 / max | memory peak median (max) |
|---|---|---|---|---|---|
| qwen3.8-flash | 20/20 | 4.9 min (14.0) | 26 | 0.39 / 1.00 / 1.07 | 1035 MiB (1278) |
| deepseek-v4-flash-0731 | 20/20 | 2.5 min (8.8) | 20 | 0.52 / 1.01 / 1.07 | 1038 MiB (2048) |

Tool containers (one per call, 1 vCPU / 1 GiB):

| model | tool calls | command time median / p95 | `docker run` wall median | CPU per call median / p95 | memory peak per call median / p95 / max | agent time inside commands |
|---|---|---|---|---|---|---|
| qwen3.8-flash | 525 | 0.02 s / 9.4 s | 1.42 s | 0.41 / 9.3 core-s | 17 / 176 / 389 MiB | 24 % |
| deepseek-v4-flash-0731 | 563 | 0.01 s / 2.6 s | 1.25 s | 0.36 / 2.7 core-s | 17 / 99 / 303 MiB | 8 % |

- **The harness is the heavy part.** opencode holds about 1 GiB throughout, and its 1 vCPU is often saturated (p95 1.00 core). One run (deepseek, sphinx-9258) sat at the 2 GiB cap with 1,358 memory-limit events and no OOM kill. With this agent, a per-tool-call design saves little unless the harness itself stays outside the sandbox and only commands go in, as this harness does.
- **Fewer, smaller tool calls than mini-swe-agent.** The median is 20–26 calls per task against 42–50 for mini-swe-agent, and per-call memory stays below 400 MiB. One test run was killed by opencode's own command timeout after about 300 s (exit 137).
- **Container start dominates small calls.** `docker run` takes 1.25–1.42 s median end to end, against a median command of 0.01–0.02 s.

Example session, harness and one colour per tool container ([all tasks](images/compare-grid.png)):

![qwen3.8-flash pytest-dev__pytest-5809](images/qwen3.8-flash-pytest-dev__pytest-5809.png)

## Limits

- Same as experiment 3 (local Docker with host contention; state outside `/testbed` lost between calls).
- The harness cap (1 vCPU) throttles opencode, so its agent time is partly CPU-bound and not comparable one to one with mini-swe-agent's.
- `memory_current` counts page cache: the harness's 1 GiB includes cached files, so its own working set may be smaller.
