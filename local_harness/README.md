# Local harness: one container per bash call (shared by experiments 2 and 3)

`local_benchmark.py opencode|miniswe` runs one agent over the task × model matrix on local Docker. Only the agent differs between [experiment 2](../opencode_tools/README.md) and [experiment 3](../miniswe_tools/README.md).

- Harness: one container per task from `harness.Dockerfile` (both agents, built once from the repo root, versions passed from `local_benchmark.py`), 1 vCPU / 2 GiB, no swap. `shared/metrics.sh` samples it → `metrics.jsonl` (the agent alone).
- Tools: `tool_client.py`, installed as `/opt/hook/bin/bash`, receives `bash -c <command>` from either agent untouched, so the model's history shows exactly what it typed. (An earlier version rewrote commands in an opencode plugin, and the model started imitating the rewritten form.) It forwards each command to a tool server in the runner (via `host.docker.internal`), which runs it with `docker run --rm` from the task image, 1 vCPU / 1 GiB, no swap. If the agent's own command timeout kills the client, the server kills the container too.
- State: one named volume per task holds `/testbed`. Docker seeds it from the task image, and the harness and every tool container mount it, so nothing needs syncing. Only state outside `/testbed` (pip installs, `/tmp`) is lost between calls.
- Concurrency: `BATCH_SIZE`, default 2, bounded by the machine (the published runs used 8).

Uses from `shared/`: `runner.py` (dataset, pool, status, mini-swe-agent config), `metrics.sh`, `plot.py`.

## Artifacts

`<exp>/runs/<model>/<instance_id>/`, as described in the root README, plus:

- `toolcalls.jsonl`: one line per call, with `docker run` wall time, exit code, `exec_start`/`exec_end`, whole-container CPU µs and memory peak.
- `toolmetrics.jsonl`: `metrics.sh` 1 s samples of every tool container, tagged with `call`.
- `metrics.jsonl` covers the harness only. `install.log` covers the image pull and volume seed.

## Charts

```bash
uv run python -m local_harness.session_chart opencode_tools/runs/qwen3.8-flash/sympy__sympy-15976   # one session
uv run python -m local_harness.session_grid --agent opencode|miniswe                               # task × model grid
```
