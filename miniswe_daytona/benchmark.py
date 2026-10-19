"""Experiment 1: mini-swe-agent on Daytona, whole-sandbox system metrics.

One Daytona sandbox per task from the task image; mini-swe-agent is installed in it and runs its
commands locally (no nested Docker). metrics.sh samples the whole sandbox cgroup every second, so
agent and tools are measured together.

Usage (from the repo root): uv run python -m miniswe_daytona.benchmark
"""
import asyncio
import os
import shlex
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSecretParams,
    DaytonaConfig,
    Resources,
    UpdateSecretParams,
)

from shared.runner import (
    CENSORED_EXIT_CODES,
    DEFAULT_DATASET_PATH,
    MINI_VERSION,
    METRICS_ARTIFACT,
    METRICS_REMOTE_OUT,
    METRICS_SCRIPT_PATH,
    OVERRIDE_PATH,
    RUN_TIMEOUT_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
    TRAJECTORY_PATH,
    _batch_size,
    _phase,
    _set_error,
    _utc_now,
    _write_status,
    build_jobs,
    load_rows,
    make_agent_config,
    metrics_summary,
    run_pool,
)

DAYTONA_SECRET = "qwen-token-plan"
DAYTONA_SECRET_HOSTS = ["token-plan.ap-southeast-1.maas.aliyuncs.com"]
DEFAULT_OUTPUT_ROOT = Path("miniswe_daytona/runs")  # relative: run from the repo root
VALIDATION_RUN_DIR = "validation-error"

# Daytona organization concurrency ceilings (total across live sandboxes, not per sandbox).
ORG_CPU_LIMIT = 10
ORG_MEMORY_LIMIT_GIB = 10

# Sandbox allocation, echoed into every metrics sample so demand can be compared
# against capacity without joining another file.
CPU_ALLOCATION = 4
# Halved from 8 GiB: the Daytona organization caps *total* concurrent memory at 10 GiB, so two
# 8 GiB sandboxes cannot coexist and every second create fails with "Total memory limit exceeded".
# Consequence for analysis: RAM demand above 4 GiB is censored from above. It is still detectable
# via the sampled memory.events oom/oom_kill counters, but the size of the overshoot is not.
MEMORY_ALLOCATION_GIB = 4
DISK_ALLOCATION_GIB = 10
METRICS_REMOTE_SCRIPT = "/tmp/metrics.sh"


def sandbox_params(row: dict[str, str]) -> CreateSandboxFromImageParams:
    return CreateSandboxFromImageParams(
        image=row["image"],
        resources=Resources(
            cpu=CPU_ALLOCATION, memory=MEMORY_ALLOCATION_GIB, disk=DISK_ALLOCATION_GIB
        ),
        secrets={"OPENAI_API_KEY": DAYTONA_SECRET},
    )


async def _start_metrics(sandbox) -> None:
    """Upload the collector and detach it. Nothing stops it: sandbox deletion reaps it."""
    await sandbox.fs.upload_file(METRICS_SCRIPT_PATH.read_bytes(), METRICS_REMOTE_SCRIPT)
    started = await sandbox.process.exec(
        f"nohup setsid bash {METRICS_REMOTE_SCRIPT} >/tmp/metrics.err 2>&1 & echo started",
        env={
            "OUT": METRICS_REMOTE_OUT,
            "INTERVAL": str(SAMPLE_INTERVAL_SECONDS),
            "ALLOC_CPU": str(CPU_ALLOCATION),
            "ALLOC_MEM_GIB": str(MEMORY_ALLOCATION_GIB),
        },
    )
    if started.exit_code != 0:
        raise RuntimeError(f"metrics collector exited with code {started.exit_code}")


INSTALL_COMMAND = (
    "python3 -m venv /opt/minisweagent && "
    f"/opt/minisweagent/bin/pip install mini-swe-agent=={MINI_VERSION}"
)
AGENT_BODY = (
    "source /root/.bashrc >/dev/null 2>&1 || true; "
    "export MSWEA_CONFIGURED=true; export MSWEA_COST_TRACKING=ignore_errors; "
    "exec /opt/minisweagent/bin/mini "
    "-c swebench.yaml -c /tmp/minisweagent/override.yaml "
    "-y --exit-immediately -o /tmp/minisweagent/trajectory.traj.json"
)


def agent_command(timeout_seconds: int = RUN_TIMEOUT_SECONDS) -> str:
    """The agent invocation, capped in-sandbox so the cap survives a transport stall.

    Killing remotely matters: cancelling the local await leaves mini-swe-agent running, which would
    keep burning CPU and mutating /testbed after the run was declared over.
    """
    return (
        f"mkdir -p /tmp/minisweagent && timeout -k 30 {int(timeout_seconds)} "
        f"bash -lc '{AGENT_BODY}'"
    )


async def _export(
    sandbox, row: dict[str, str], run_dir: Path, status: dict, agent_ok: bool
) -> None:
    patch_ok = False
    try:
        patch = await sandbox.process.exec(
            f"git -C /testbed diff --binary {shlex.quote(row['base_commit'])}", cwd="/testbed"
        )
    except Exception as error:
        # Transport failure: still try to recover the trajectory below.
        _set_error(status, error)
    else:
        (run_dir / "patch.diff").write_text(patch.result or "", encoding="utf-8")
        status["exit_codes"]["patch"] = patch.exit_code
        status["artifacts"]["patch_diff"] = patch.exit_code == 0
        patch_ok = patch.exit_code == 0
        if not patch_ok:
            _set_error(status, RuntimeError(f"patch exited with code {patch.exit_code}"))

    try:
        metrics = await sandbox.fs.download_file(METRICS_REMOTE_OUT)
    except Exception as error:
        _set_error(status, error)
    else:
        (run_dir / METRICS_ARTIFACT).write_bytes(metrics)
        status["artifacts"]["metrics"] = True
        status["metrics"] = metrics_summary(metrics)

    try:
        trajectory = await sandbox.fs.download_file(TRAJECTORY_PATH)
        (run_dir / "trajectory.traj.json").write_bytes(trajectory)
        status["artifacts"]["trajectory"] = True
    except Exception as error:
        if agent_ok and patch_ok:
            # A missing trajectory after a successful agent run is an export failure.
            _set_error(status, error)


async def run_task(
    row: dict[str, str],
    daytona: AsyncDaytona,
    output_root: Path,
    model_name: str,
    timeout: int = RUN_TIMEOUT_SECONDS,
) -> bool:
    run_dir = output_root / model_name / row["instance_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    status: dict = {
        "instance_id": row["instance_id"],
        "model_name": model_name,
        "sandbox_id": None,
        "phase": "create",
        "outcome": "failure",
        "error": None,
        "exit_codes": {},
        "phases": {},
        "artifacts": {
            "install_log": False,
            "agent_log": False,
            "trajectory": False,
            "patch_diff": False,
            "metrics": False,
        },
        "metrics": None,
        "cleanup": None,
        "timeout_s": timeout,
        "censored": False,
        "started_at": _utc_now(),
    }
    sandbox = None
    try:
        with _phase(status, run_dir, "create"):
            # timeout=0: disable Daytona's default 60s sandbox-start wait.
            sandbox = await daytona.create(sandbox_params(row), timeout=0)
            status["sandbox_id"] = sandbox.id
            # Sampling starts here so install and agent work are both covered.
            await _start_metrics(sandbox)

        with _phase(status, run_dir, "install"):
            install = await sandbox.process.exec(INSTALL_COMMAND, cwd="/testbed")
            (run_dir / "install.log").write_text(install.result or "", encoding="utf-8")
            status["exit_codes"]["install"] = install.exit_code
            status["artifacts"]["install_log"] = True
            install_ok = install.exit_code == 0
            if not install_ok:
                _set_error(status, RuntimeError(f"install exited with code {install.exit_code}"))

        if install_ok:
            agent_ok = False
            try:
                with _phase(status, run_dir, "agent"):
                    await sandbox.process.exec("mkdir -p /tmp/minisweagent")
                    await sandbox.fs.upload_file(make_agent_config(row, model_name), OVERRIDE_PATH)
                    # timeout=None: the in-sandbox `timeout` cap owns the wall clock, so a hung
                    # agent is still killed and exported instead of leaving the local await open.
                    agent = await sandbox.process.exec(
                        agent_command(timeout), cwd="/testbed", timeout=None
                    )
                    (run_dir / "agent.log").write_text(agent.result or "", encoding="utf-8")
                    status["exit_codes"]["agent"] = agent.exit_code
                    status["artifacts"]["agent_log"] = True
                    agent_ok = agent.exit_code == 0
                    if not agent_ok:
                        if agent.exit_code in CENSORED_EXIT_CODES:
                            status["censored"] = True
                            _set_error(
                                status,
                                f"censored: agent exceeded the {timeout}s wall-clock cap",
                            )
                        else:
                            _set_error(
                                status,
                                RuntimeError(f"agent exited with code {agent.exit_code}"),
                            )
            except Exception as error:
                _set_error(status, error)

            with _phase(status, run_dir, "export"):
                await _export(sandbox, row, run_dir, status, agent_ok)

        if status["error"] is None:
            status["outcome"] = "success"
    except Exception as error:
        _set_error(status, error)
    finally:
        if sandbox is not None:
            failing_phase = status["phase"]
            with _phase(status, run_dir, "cleanup"):
                try:
                    await daytona.delete(sandbox, wait=True)
                    status["cleanup"] = "deleted"
                except Exception as error:
                    # Cleanup failure dominates: the sandbox leaked.
                    status["cleanup"] = "failed"
                    _set_error(status, error)
                    status["outcome"] = "failure"
                    failing_phase = "cleanup"
            if status["cleanup"] == "deleted":
                # Successful cleanup must not mask the phase that decided the outcome.
                status["phase"] = failing_phase
        else:
            status["cleanup"] = "skipped"
        status["finished_at"] = _utc_now()
        _write_status(run_dir, status)
    return status["outcome"] == "success"


def _validation_status(output_root: Path, error: BaseException) -> None:
    """Record a host-side validation failure without touching the provider.

    The run directory is a fixed safe name: no value derived from the bad row is ever used as a path.
    """
    run_dir = output_root / VALIDATION_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    instance_id = getattr(error, "instance_id", None)
    status: dict = {
        "instance_id": str(instance_id) if instance_id else None,
        "model_name": "unknown",
        "sandbox_id": None,
        "phase": "validate",
        "outcome": "failure",
        "error": None,
        "exit_codes": {},
        "phases": {},
        "artifacts": {
            "install_log": False,
            "agent_log": False,
            "trajectory": False,
            "patch_diff": False,
            "metrics": False,
        },
        "metrics": None,
        "cleanup": "not_started",
        "started_at": _utc_now(),
    }
    with _phase(status, run_dir, "validate"):
        _set_error(status, error)
    status["finished_at"] = _utc_now()
    _write_status(run_dir, status)


async def _provision_secret(client: AsyncDaytona, plaintext: str) -> None:
    """Create or update the organization secret ``qwen-token-plan`` from a plaintext value.

    The plaintext is only ever passed to the provider SDK call; it is never logged, written to
    status, config, or run artifacts. The provider's name filter is partial, so exact-name
    matches are filtered here; more than one exact match is a loud defensive failure.
    """
    page = await client.secret.list(name=DAYTONA_SECRET)
    items = list(getattr(page, "items", None) or [])
    exact = [secret for secret in items if getattr(secret, "name", None) == DAYTONA_SECRET]
    if len(exact) > 1:
        # Do not interpolate secret fields; only safe identifiers, and never the plaintext.
        raise RuntimeError(
            f"multiple Daytona secrets named {DAYTONA_SECRET!r}: "
            f"ids={[getattr(secret, 'id', None) for secret in exact]}"
        )
    if exact:
        await client.secret.update(
            exact[0].id,
            UpdateSecretParams(value=plaintext, hosts=list(DAYTONA_SECRET_HOSTS)),
        )
    else:
        await client.secret.create(
            CreateSecretParams(
                name=DAYTONA_SECRET, value=plaintext, hosts=list(DAYTONA_SECRET_HOSTS)
            )
        )


async def run_benchmark(
    env: Mapping[str, str] = os.environ,
    client_factory: Callable[[DaytonaConfig], AsyncDaytona] = AsyncDaytona,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> bool:
    api_key = (env.get("DAYTONA_API_KEY") or "").strip()
    if not api_key:
        # Checked first so a missing credential can never read data or create a client.
        raise ValueError("DAYTONA_API_KEY is required")
    token = (env.get("QWEN_TOKEN_PLAN_API_KEY") or "").strip()
    if not token:
        # Checked before dataset access and client creation, like the API key.
        raise ValueError("QWEN_TOKEN_PLAN_API_KEY is required")
    # Validated before any dataset read or client creation, so a bad pool width is loud and cheap.
    batch_size = _batch_size(env)
    dataset_path = Path(env.get("DATASET_PATH") or DEFAULT_DATASET_PATH)
    try:
        rows = load_rows(dataset_path)
    except Exception as error:
        try:
            _validation_status(output_root, error)
        except OSError:
            pass  # never mask the dataset failure with a status-write failure
        raise
    daytona = client_factory(DaytonaConfig(api_key=api_key))
    try:
        # Provision before any sandbox: a failure must propagate without running a single task.
        await _provision_secret(daytona, token)
        outcomes = await run_pool(
            build_jobs(rows),
            lambda row, model_name: run_task(row, daytona, output_root, model_name),
            batch_size=batch_size,
        )
        succeeded = sum(1 for outcome in outcomes if outcome)
        print(f"runs: {succeeded}/{len(outcomes)} succeeded", flush=True)
        return succeeded == len(outcomes)
    finally:
        await daytona.close()


def main() -> int:
    try:
        return 0 if asyncio.run(run_benchmark()) else 1
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
