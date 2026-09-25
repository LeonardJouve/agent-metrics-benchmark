# AI Agent System Metrics Benchmark

Research plan for measuring **agent runtime resource use**, excluding remote LLM inference, from a sandbox-provider perspective.

## Status

- The full task x model matrix runner is implemented: `benchmark.py` runs every dataset row against both pinned models through a slot pool (`BATCH_SIZE` sandboxes at a time, default 2; a queued job starts the moment a slot frees) and exports `install.log`, `agent.log`, `trajectory.traj.json`, `patch.diff`, `metrics.jsonl`, and `status.json` under `runs/<model>/<instance_id>/`. Each run drives mini-SWE-agent in its `local` environment inside one Daytona sandbox (no nested Docker) and is capped at 45 minutes by an in-sandbox `timeout`, which records `censored: true` and still exports artifacts.
- Execution plan for the 40 measured runs: [`plan/run_plan.md`](plan/run_plan.md). No measured run has been executed yet.
- Resource metrics collection is implemented: `metrics.sh` is uploaded to the sandbox and detached at 1-second intervals, writing cgroup v2 JSONL samples until sandbox deletion; the file is downloaded during export. Patch evaluation/resolved detection remains unimplemented. One development canary completed successfully on `scikit-learn__scikit-learn-14629`, with all expected artifacts exported and the sandbox deleted. Because evaluation is still unimplemented, that canary does not claim patch correctness or resolved status.
- Daytona organization limits are **10 vCPU and 10 GiB total across live sandboxes**, not per sandbox. A first matrix attempt at 4 vCPU / 8 GiB with `BATCH_SIZE=2` got exactly one sandbox; the other 39 jobs failed at `create` with `Total memory limit exceeded. Maximum allowed: 10GiB`. The measured spec is therefore 4 vCPU / 4 GiB per sandbox (8 vCPU / 8 GiB for two slots). That attempt also left one sandbox running after the process was interrupted, which was deleted by hand: an interrupted run does not clean up its live sandboxes.
- Target workload: Python coding agents.
- Agent: [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6, installed in-sandbox at `/opt/minisweagent`.
- Sandbox runner: Daytona Cloud. Daytona is only the measurement platform, not the future hotplug target.
- Dataset: [`data/sample20.csv`](data/sample20.csv) (20 tasks; default runner input).
- Research sources were checked on 2026-09-24. Raw research cache exists under `.firecrawl/` but is not part of the durable plan.

## Running the Matrix

```bash
set -a; . ./.env; set +a   # export the gitignored .env; the runner reads process env only
uv run python benchmark.py
```

Environment (process environment only; copy `.env.example` to `.env`, export it, and fill in both keys):

| Variable | Meaning |
|---|---|
| `DATASET_PATH` | Dataset CSV to read; defaults to `./data/sample20.csv`. Every row runs, against every model in `MODELS`. |
| `BATCH_SIZE` | Concurrent sandboxes in the pool; defaults to `2`. Must be a positive integer; invalid values fail before any client or sandbox is created. |
| `DAYTONA_API_KEY` | Daytona API key. Required; the runner fails before creating any sandbox when it is missing. |
| `QWEN_TOKEN_PLAN_API_KEY` | Plaintext model-provider token. Required; the runner fails before creating any client when it is missing or blank, and provisions the Daytona secret from it before any sandbox starts. |

Exit codes: `0` only when every run in the matrix succeeds, `1` when any run fails, is censored, or cleanup fails. A single raising or hung run is isolated: it cannot stop its siblings, and its sandbox is still deleted. Run artifacts land in `runs/<model>/<instance_id>/`, which is gitignored. There is no resume: re-running re-executes every job and overwrites those directories. If any dataset row fails validation (missing required field, duplicate instance id), no Daytona client or sandbox is created at all: the failure is recorded in `runs/validation-error/status.json` with `phase: validate` and `cleanup: not_started`, and that record carries the offending row's `instance_id` as data only (the directory name stays fixed, so a hostile id can never create a path).

Required Daytona organization secret (provisioned by the code from `QWEN_TOKEN_PLAN_API_KEY`, whose plaintext source is kept only in the gitignored `.env`/process environment):

- Secret name: `qwen-token-plan`
- Exposed inside the sandbox as: `OPENAI_API_KEY`
- Host allowlist: `token-plan.ap-southeast-1.maas.aliyuncs.com` (must be the only allowed host)

Daytona substitutes the opaque `OPENAI_API_KEY` placeholder through its outbound proxy. The plaintext model credential is not tracked by git or written to run logs, status, or trajectory; it exists locally only in the gitignored `.env`/exported process environment and is passed to Daytona's secrets API. Before each run, the code provisions this secret (create when absent, update when present), and no artifact records the plaintext.

Both samples are checked in as the rebuilt 17-field SWE-bench Verified rows the runner reads. Re-running `uv run python build_dataset.py` is only for refreshing or reproducing the selection from the upstream source and needs public network access, so it is not part of automated verification.

## Goal

Collect session-aligned CPU and RAM time series that can inform CPU/RAM allocation and hotplug policy for a future agent sandbox platform.

Primary outputs:

1. CPU/RAM demand distributions by task, repository, phase, outcome, and model.
2. Peak, percentile, sustained-pressure, idle, throttling, and OOM signals.
3. Candidate initial allocations and scale-up thresholds.
4. Estimated sandbox cost per task and per successful task.

This is a resource benchmark, not primarily an agent-accuracy leaderboard.

## Fixed Decisions

| Item | Decision |
|---|---|
| Tasks | 20 Python coding tasks |
| Dataset mix | 20 SWE-bench Verified tasks from `SWE-bench/SWE-bench_Verified` (config `default`, split `test`) |
| Image cap | Docker Hub reported image size ≤2 GB; no substitution above cap without approval |
| Models | Qwen3.8 Flash + DeepSeek V4 Flash; verify exact provider API IDs before execution |
| Matrix | 20 tasks × 2 models = 40 live runs |
| Repetition | One live run per task/model; no rerun of the same pair |
| Agent retries | Internal failed-tool retries allowed and recorded |
| Sampling | 1 second |
| Artifacts | Full prompts, LLM responses, tool calls/results, timestamps, outcome, metrics |
| Expected duration | 15–30 minutes reserved per run; enforced cap 45 minutes per agent rollout, killed in-sandbox and recorded as censored |
| Concurrency | 2 sandboxes at a time (`BATCH_SIZE=2`); deviation from the single-slot baseline is documented in `plan/run_plan.md` |
| Cleanup | Export artifacts, then immediately delete sandbox |
| Analysis | Report models separately; combine only with explicit weighting |

“Model-agnostic” means one harness/schema supports both models. Results are not model-independent: model behavior changes tool count, idle time, runtime, and resource demand.

## Dataset Choice

Use SWE-bench Verified as the single source of truth for the measured sample. Do not use SWE-bench Lite as the primary set; it has the same narrow repository family as Verified and gives less coverage.

The tracked 20-task sample is built by `build_dataset.py` from `SWE-bench/SWE-bench_Verified`:

- All 20 tasks are SWE-bench Verified; the sample has no other source.
- Only tasks rated `<15 min fix` or `15 min - 1 hour` are eligible, so every task is intended to take under one hour.
- Eligible rows are grouped by repository and taken round-robin, so repositories are represented as evenly as eligible capacity permits.
- Only images present on Docker Hub with a reported size no larger than 2 GB (`full_size <= 2_000_000_000` bytes) are eligible.
- Selection is deterministic: seed `42`, so the same source data always yields the same 20 rows.

The builder preserves the source dataset's 17 columns and column order, with list/object cells JSON-encoded; it adds no metadata columns.

### Disk-headroom constraint

The ≤2 GB reported-image cap is mandatory. It leaves room within Daytona’s default 10 GiB disk for extracted layers, the writable layer, repository changes, tests, logs, and metrics. Docker Hub size does not prove final on-disk use, so preflight must still verify free space after image startup.

Do not substitute an image above 2 GB without user approval. Do not silently drop failed tasks during execution. Record image/setup/infra failures separately from agent failures.

SWE-rebench is a later expansion candidate: 21,336 tasks across 3,400+ Python repositories, but its automated validation is noisier.

## Benchmark Flow

1. **Preflight with separate canary tasks**
   - Do not consume any of the 20 measured task/model pairs.
   - Validate image pull, repository state, model access, session capture, 1-second sampling, upload, and deletion.
2. **Validate sample feasibility**
   - Recheck every image digest and enforce reported size ≤2 GB.
   - Confirm Daytona disk quota, post-start free space, and network access.
   - Run gold-patch/environment checks without invoking benchmark models.
3. **Run measured matrix**
   - Run 2 sandboxes at a time; keep setup-phase samples separate from agent-phase samples, because concurrent sandboxes share host disk and network.
   - Use fixed 4 vCPU / 4 GiB RAM. 8 GiB was the intent, but the Daytona organization caps total concurrent memory at 10 GiB, which allows only one 8 GiB sandbox; 4 GiB is what two slots fit under.
   - Start metrics before agent setup; stop only after final outcome is recorded.
4. **Export atomically**
   - Session log, metric series, run metadata, and outcome must share one run ID.
   - Confirm artifacts are durable before deleting the sandbox.
5. **Analyze**
   - Segment by model, dataset source, repository, resource tier, task horizon, infrastructure, outcome, and phase.
   - Mark censored runs that hit the wall-clock cap.

Do not test the future hotplug policy by changing Daytona allocations during this baseline. First collect demand on 4 vCPU / 4 GiB, and treat RAM above the 4 GiB ceiling as right-censored: the sampled `memory.events` `oom`/`oom_kill` counters reveal that the cap bound, but not by how much. Daytona is the profiler; the future sandbox platform is the optimization target.

## Metrics Design

Implemented as `metrics.sh`, a bash cgroup v2 collector started right after sandbox creation and left running until deletion. It appends one JSON object per line to `/tmp/metrics.jsonl`, which export downloads to `runs/<model>/<instance_id>/metrics.jsonl`. Sampling uses `EPOCHREALTIME` and bash `read` loops, so a tick forks only `sleep`; the collector stays inside the measured cgroup. Knob environment variables (`CG_ROOT`, `OUT`, `INTERVAL`, `DURATION`, `ALLOC_CPU`, `ALLOC_MEM_GIB`) let tests run it against a fixture cgroup directory. A missing `cpu.stat` writes one `{"error": ...}` line and exits 1, which fails the run. `status.metrics` records `samples`, `first_ts`, `last_ts`, and `interval_s` (median gap; torn trailing lines are skipped).

Required raw fields:

- `cpu.stat`: `usage_usec`, `user_usec`, `system_usec`, `nr_periods`, `nr_throttled`, `throttled_usec`
- `cpu.max`
- `memory.current`
- `memory.max`
- `memory.events`: `high`, `max`, `oom`, `oom_kill`
- `memory.peak`, when available
- Optional: `memory.stat` cache/anonymous split and cgroup pressure files
- Allocated CPU/RAM, process exit status, wall time, and sample timestamp

Derived fields:

- CPU cores used and utilization relative to allocation
- CPU throttling ratio
- Current/peak memory and allocation headroom
- Time above pressure thresholds
- Idle duration, burst duration, and phase-level demand
- Integral CPU-seconds and GiB-seconds

Do not use cAdvisor or node_exporter inside a Daytona sandbox for primary attribution. They require host/runtime visibility or expose host-wide values. Daytona historical telemetry is approximately 60-second resolution. `get_metrics_latest()` bypasses historical aggregation, but its internal refresh window is undocumented; it may be used only after measuring its effective cadence.

Collector overhead belongs to the measured cgroup. Record collector CPU/RAM and keep it minimal.

## Session Record

Each run needs immutable metadata:

- run ID, task ID, dataset source, repository, commit, image digest
- model provider/API ID and model parameters
- mini-SWE-agent version/config
- sandbox resources, region, timestamps, and wall-clock cap
- prompts, responses, tool calls/results, retries, command exits
- 1-second metrics keyed by run ID
- final patch, resolved status, timeout/censor flag, and failure class

Store remote LLM latency and token usage for interpretation, but exclude remote inference compute from sandbox resource totals. Continue polling during LLM waits so idle RAM and wall time remain visible.

## Daytona Cost Estimate

Official published rates used:

- vCPU: $0.0504/vCPU-hour
- RAM: $0.0162/GiB-hour
- Storage: $0.000108/GiB-hour after the first 5 GiB
- Billing: per second

For 40 runs:

| Sandbox | Hourly estimate | 15 min/run | 30 min/run |
|---|---:|---:|---:|
| 2 vCPU / 4 GiB / 8 GiB disk | $0.165924 | $1.66 | $3.32 |
| 4 vCPU / 4 GiB / 10 GiB disk (measured spec) | $0.266940 | $2.67 | $5.34 |
| 4 vCPU / 8 GiB / 10 GiB disk | $0.331740 | $3.32 | $6.63 |

These totals exclude LLM API charges and extra image-pull/setup time. Each extra minute across 40 runs adds about $0.11 at 2/4 or $0.22 at 4/8. Existing Daytona free compute credit may reduce cash cost, but never use credit availability as the reported resource cost.

## Known Constraints

- No same-model task reruns means infrastructure-noise variance cannot be estimated directly.
- Tag every infra failure; do not merge it with agent failure.
- Large image pulls may dominate network, disk, and startup metrics. Analyze setup separately from agent work.
- Long SWE-bench regression suites can dominate evaluation time. Keep evaluation phase separate from agent rollout in analysis.
- mini-SWE-agent has no upstream Daytona environment, so the runner drives it in the sandbox’s `local` mode: it installs the pinned version into `/opt/minisweagent` and uses Daytona process/filesystem APIs for install, config upload, execution, and artifact download. No nested Docker is started. Remaining work is patch evaluation, not the agent environment or the cgroup collector.
- Daytona allows live CPU/RAM increases, but decreases require stopping. This does not constrain the future provider’s hotplug design because Daytona is only the profiler.

## Success Criteria

Research run is usable when:

- 40 intended task/model runs have a terminal record, including explicit infra failures.
- Every non-setup run has aligned session and 1-second metric artifacts.
- Missing-sample rate and collector overhead are reported.
- CPU/RAM recommendations include uncertainty and censored-run handling.
- Cost is reported per attempted task, completed task, and resolved task.
- Proposed hotplug thresholds can be traced to observed metrics, not averages alone.

## Next Agent Handoff

Next agent should:

1. Read this README and `data/sample20.csv`.
2. Verify current Daytona SDK/API, pricing, quotas, and observability behavior against official docs.
3. Verify exact Qwen/DeepSeek model API identifiers; do not rename models silently.
4. Preserve the ≤2 GB reported-image cap; verify post-start disk headroom before measured runs.
5. Write a separate implementation design/spec before coding.
6. Add patch evaluation using TDD; the matrix runner, 2-slot pool, 45-minute cap, mini-SWE-agent local-mode integration, and 1-second cgroup collector already exist.
7. Use canary tasks outside the 20-task sample before any measured run.
8. Never commit without fresh explicit user approval.

One-task agent execution, artifact export, and 1-second cgroup metrics are implemented; patch evaluation is not. One development canary completed successfully on `scikit-learn__scikit-learn-14629`, with artifacts exported and the sandbox deleted; it predates metrics collection, claims no patch correctness/resolved status, and no commit has been approved.

## Sources

- [SWE-bench](https://www.swebench.com/)
- [SWE-bench Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)
- [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench)
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)
- [Daytona pricing](https://www.daytona.io/pricing)
- [Daytona sandboxes and resizing](https://www.daytona.io/docs/en/sandboxes)
- [Daytona OpenTelemetry collection](https://www.daytona.io/docs/en/observability/otel-collection)
- [Daytona Toolbox OpenAPI](https://www.daytona.io/docs/toolbox-openapi.json)
