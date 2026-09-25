# AI Agent System Metrics Benchmark

Research plan for measuring **agent runtime resource use**, excluding remote LLM inference, from a sandbox-provider perspective.

## Status

- Initial concurrent Daytona sandbox creation runner implemented; agent execution, sessions, and metrics remain unimplemented.
- Target workload: Python coding agents.
- Agent: [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent).
- Sandbox runner: Daytona Cloud. Daytona is only the measurement platform, not the future hotplug target.
- Main dataset sample: [`data/sample50.csv`](data/sample50.csv).
- Research sources were checked on 2026-09-24. Raw research cache exists under `.firecrawl/` but is not part of the durable plan.

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
| Tasks | 50 Python coding tasks |
| Dataset mix | 35 SWE-bench-Live Verified + 15 SWE-bench Verified |
| Image cap | Docker Hub reported image size ≤2 GB; no substitution above cap without approval |
| Models | Qwen3.8 Flash + DeepSeek V4 Flash; verify exact provider API IDs before execution |
| Matrix | 50 tasks × 2 models = 100 live runs |
| Repetition | One live run per task/model; no rerun of the same pair |
| Agent retries | Internal failed-tool retries allowed and recorded |
| Sampling | 1 second |
| Artifacts | Full prompts, LLM responses, tool calls/results, timestamps, outcome, metrics |
| Expected duration | 15–30 minutes reserved per run; enforce an explicit wall-clock cap |
| Cleanup | Export artifacts, then immediately delete sandbox |
| Analysis | Report models separately; combine only with explicit weighting |

“Model-agnostic” means one harness/schema supports both models. Results are not model-independent: model behavior changes tool count, idle time, runtime, and resource demand.

## Dataset Choice

Do not use SWE-bench Lite as the primary set. It has the same narrow 12-repository family as Verified and gives less coverage.

Use the tracked 50-task sample:

- 35 tasks from SWE-bench-Live Verified for current, diverse repositories and wider infrastructure/resource profiles.
- 15 tasks from SWE-bench Verified for a stable, human-filtered, canonical anchor.
- 41 unique repositories total.
- All 50 referenced images existed when checked.
- Every Docker Hub reported image size is ≤2 GB.
- Reported image sizes total 47.58 GB; individual images span 0.44–1.92 GB.
- Live task horizons are balanced: 12 `h0`, 12 `h1`, and 11 `h2`.

`data/sample50.csv` records source, instance ID, repository, resource tier, horizon band, test infrastructure, patch/test proxies, image namespace, image availability, and reported image size.

### Disk-headroom constraint

The ≤2 GB reported-image cap is mandatory. It leaves room within Daytona’s default 10 GiB disk for extracted layers, the writable layer, repository changes, tests, logs, and metrics. Docker Hub size does not prove final on-disk use, so preflight must still verify free space after image startup.

Do not substitute an image above 2 GB without user approval. Do not silently drop failed tasks during execution. Record image/setup/infra failures separately from agent failures.

SWE-rebench is a later expansion candidate: 21,336 tasks across 3,400+ Python repositories, but its automated validation is noisier.

## Benchmark Flow

1. **Preflight with separate canary tasks**
   - Do not consume any of the 50 measured task/model pairs.
   - Validate image pull, repository state, model access, session capture, 1-second sampling, upload, and deletion.
2. **Validate sample feasibility**
   - Recheck every image digest and enforce reported size ≤2 GB.
   - Confirm Daytona disk quota, post-start free space, and network access.
   - Run gold-patch/environment checks without invoking benchmark models.
3. **Run measured matrix**
   - Prefer one active measured sandbox at a time for the baseline.
   - Use fixed 4 vCPU / 8 GiB RAM to reduce cap-induced distortion.
   - Start metrics before agent setup; stop only after final outcome is recorded.
4. **Export atomically**
   - Session log, metric series, run metadata, and outcome must share one run ID.
   - Confirm artifacts are durable before deleting the sandbox.
5. **Analyze**
   - Segment by model, dataset source, repository, resource tier, task horizon, infrastructure, outcome, and phase.
   - Mark censored runs that hit the wall-clock cap.

Do not test the future hotplug policy by changing Daytona allocations during this baseline. First collect uncensored demand on 4 vCPU / 8 GiB. Daytona is the profiler; the future sandbox platform is the optimization target.

## Metrics Design

Use a small in-sandbox **cgroup v2 collector** at 1-second intervals. Expose Prometheus text or write timestamped JSONL for later import. Run Prometheus/remote storage outside the measured sandbox.

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

For 100 runs:

| Sandbox | Hourly estimate | 15 min/run | 30 min/run |
|---|---:|---:|---:|
| 2 vCPU / 4 GiB / 8 GiB disk | $0.165924 | $4.15 | $8.30 |
| 4 vCPU / 8 GiB / 10 GiB disk | $0.331740 | $8.29 | $16.59 |

These totals exclude LLM API charges and extra image-pull/setup time. Each extra minute across 100 runs adds about $0.28 at 2/4 or $0.55 at 4/8. Existing Daytona free compute credit may reduce cash cost, but never use credit availability as the reported resource cost.

## Known Constraints

- No same-model task reruns means infrastructure-noise variance cannot be estimated directly.
- Tag every infra failure; do not merge it with agent failure.
- Large image pulls may dominate network, disk, and startup metrics. Analyze setup separately from agent work.
- SWE-bench-Live regression suites can be much larger than Verified. Keep evaluation phase separate from agent rollout in analysis.
- mini-SWE-agent currently has no upstream Daytona environment. Implementation will need a small environment adapter using Daytona process/session APIs.
- Daytona allows live CPU/RAM increases, but decreases require stopping. This does not constrain the future provider’s hotplug design because Daytona is only the profiler.

## Success Criteria

Research run is usable when:

- 100 intended task/model runs have a terminal record, including explicit infra failures.
- Every non-setup run has aligned session and 1-second metric artifacts.
- Missing-sample rate and collector overhead are reported.
- CPU/RAM recommendations include uncertainty and censored-run handling.
- Cost is reported per attempted task, completed task, and resolved task.
- Proposed hotplug thresholds can be traced to observed metrics, not averages alone.

## Next Agent Handoff

Next agent should:

1. Read this README and `data/sample50.csv`.
2. Verify current Daytona SDK/API, pricing, quotas, and observability behavior against official docs.
3. Verify exact Qwen/DeepSeek model API identifiers; do not rename models silently.
4. Preserve the ≤2 GB reported-image cap; verify post-start disk headroom before measured runs.
5. Write a separate implementation design/spec before coding.
6. Build the smallest Daytona environment adapter and cgroup collector using TDD.
7. Use canary tasks outside the 50-task sample before any measured run.
8. Never commit without fresh explicit user approval.

Only initial sandbox creation is implemented. No commit has been approved.

## Sources

- [SWE-bench](https://www.swebench.com/)
- [SWE-bench Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)
- [SWE-bench-Live](https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live)
- [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench)
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)
- [Daytona pricing](https://www.daytona.io/pricing)
- [Daytona sandboxes and resizing](https://www.daytona.io/docs/en/sandboxes)
- [Daytona OpenTelemetry collection](https://www.daytona.io/docs/en/observability/otel-collection)
- [Daytona Toolbox OpenAPI](https://www.daytona.io/docs/toolbox-openapi.json)
