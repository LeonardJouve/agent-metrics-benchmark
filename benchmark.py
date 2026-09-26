import asyncio
import csv
import json
import os
import shlex
import statistics
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSecretParams,
    DaytonaConfig,
    Resources,
    UpdateSecretParams,
)

DEFAULT_DATASET_PATH = Path("data/sample20.csv")
MODELS = [
    "qwen3.8-flash",
    "deepseek-v4-flash-0731",
]
MODEL_API_BASE = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
MINI_VERSION = "2.4.6"
DAYTONA_SECRET = "qwen-token-plan"
DAYTONA_SECRET_HOSTS = ["token-plan.ap-southeast-1.maas.aliyuncs.com"]
REQUIRED_FIELDS = ("instance_id", "image", "base_commit", "problem_statement")
DEFAULT_OUTPUT_ROOT = Path("runs")
VALIDATION_RUN_DIR = "validation-error"

# Pool sizing and the wall-clock cap. The cap is enforced twice: remotely, by wrapping the agent
# command in coreutils `timeout` so the artifact export still runs, and locally, by cancelling a
# job whose transport has hung. The local belt gets grace on top so the remote kill is what
# normally fires.
DEFAULT_BATCH_SIZE = 2
# Daytona organization concurrency ceilings (total across live sandboxes, not per sandbox).
ORG_CPU_LIMIT = 10
ORG_MEMORY_LIMIT_GIB = 10
RUN_TIMEOUT_SECONDS = 45 * 60
POOL_GRACE_SECONDS = 300

# Sandbox allocation, echoed into every metrics sample so demand can be compared
# against capacity without joining another file.
CPU_ALLOCATION = 4
# Halved from 8 GiB: the Daytona organization caps *total* concurrent memory at 10 GiB, so two
# 8 GiB sandboxes cannot coexist and every second create fails with "Total memory limit exceeded".
# Consequence for analysis: RAM demand above 4 GiB is censored from above. It is still detectable
# via the sampled memory.events oom/oom_kill counters, but the size of the overshoot is not.
MEMORY_ALLOCATION_GIB = 4
DISK_ALLOCATION_GIB = 10
SAMPLE_INTERVAL_SECONDS = 1
METRICS_SCRIPT_PATH = Path(__file__).resolve().parent / "metrics.sh"
METRICS_REMOTE_SCRIPT = "/tmp/metrics.sh"
METRICS_REMOTE_OUT = "/tmp/metrics.jsonl"
METRICS_ARTIFACT = "metrics.jsonl"

# SWE-bench rows carry multi-hundred-KiB patch/test_patch cells, which exceed the
# stdlib csv reader's 128 KiB default limit.
CSV_FIELD_SIZE_LIMIT = 2**31 - 1


@contextmanager
def _large_csv_fields() -> Iterator[None]:
    """Temporarily lift the process-global csv reader limit, then restore it exactly."""
    previous = csv.field_size_limit()
    try:
        try:
            csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)
        except (OverflowError, ValueError):
            pass  # platform rejects the raise; keep the current limit
        yield
    finally:
        csv.field_size_limit(previous)


def missing_fields(row: dict[str, str]) -> list[str]:
    return [field for field in REQUIRED_FIELDS if not row.get(field)]


class DatasetValidationError(ValueError):
    """A row that parsed but failed required-field validation.

    Carries the offending ``instance_id`` as status data only. Callers must never use it as a path:
    dataset rows are untrusted input.
    """

    def __init__(self, message: str, instance_id: str | None = None) -> None:
        super().__init__(message)
        self.instance_id = instance_id


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Every dataset row, each validated for required fields.

    The whole file is read inside one field-size-limit window so the process-global
    csv limit is restored before this returns, whatever the outcome.
    """
    with _large_csv_fields(), path.open(encoding="utf-8-sig", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            missing = missing_fields(row)
            if missing:
                raise DatasetValidationError(
                    f"{path}: row missing required fields: {', '.join(missing)}",
                    instance_id=row.get("instance_id") or None,
                )
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"{path}: dataset has no rows")
    return rows


def load_rows(path: Path) -> list[dict[str, str]]:
    """All rows. Instance ids must be unique: they name run directories."""
    rows = _read_rows(path)
    seen: set[str] = set()
    for row in rows:
        if row["instance_id"] in seen:
            raise ValueError(f"{path}: duplicate instance_id: {row['instance_id']}")
        seen.add(row["instance_id"])
    return rows


def make_agent_config(row: dict[str, str], model_name: str) -> bytes:
    missing = missing_fields(row)
    if missing:
        raise ValueError(f"row missing required fields: {', '.join(missing)}")
    config = {
        "run": {"task": row["problem_statement"]},
        "agent": {"step_limit": 250, "cost_limit": 0, "confirm_exit": False},
        "environment": {"environment_class": "local", "cwd": "/testbed", "timeout": 60},
        "model": {
            "model_name": model_name,
            "model_class": "litellm",
            "model_kwargs": {
                "custom_llm_provider": "openai",
                "api_base": MODEL_API_BASE,
                "drop_params": True,
            },
        },
    }
    return json.dumps(config, ensure_ascii=False).encode("utf-8")


def sandbox_params(row: dict[str, str]) -> CreateSandboxFromImageParams:
    return CreateSandboxFromImageParams(
        image=row["image"],
        resources=Resources(
            cpu=CPU_ALLOCATION, memory=MEMORY_ALLOCATION_GIB, disk=DISK_ALLOCATION_GIB
        ),
        secrets={"OPENAI_API_KEY": DAYTONA_SECRET},
    )


def metrics_summary(data: bytes) -> dict:
    """Sample count, first/last timestamp, and median sampling gap of a metrics JSONL payload.

    Lines without a numeric ``epoch`` are skipped: the collector is still appending when the
    file is downloaded, so the last line may be torn, and an error line carries no samples.
    """
    samples = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("epoch"), (int, float)):
            samples.append(parsed)
    if not samples:
        return {"samples": 0, "first_ts": None, "last_ts": None, "interval_s": None}
    epochs = [sample["epoch"] for sample in samples]
    gaps = [later - earlier for earlier, later in zip(epochs, epochs[1:])]
    return {
        "samples": len(samples),
        "first_ts": samples[0].get("ts"),
        "last_ts": samples[-1].get("ts"),
        "interval_s": statistics.median(gaps) if gaps else None,
    }


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


TRAJECTORY_PATH = "/tmp/minisweagent/trajectory.traj.json"
OVERRIDE_PATH = "/tmp/minisweagent/override.yaml"
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
# `timeout` exit codes: 124 when TERM fires at the cap, 137 when the -k grace escalates to KILL.
CENSORED_EXIT_CODES = frozenset({124, 137})


def agent_command(timeout_seconds: int = RUN_TIMEOUT_SECONDS) -> str:
    """The agent invocation, capped in-sandbox so the cap survives a transport stall.

    Killing remotely matters: cancelling the local await leaves mini-swe-agent running, which would
    keep burning CPU and mutating /testbed after the run was declared over.
    """
    return (
        f"mkdir -p /tmp/minisweagent && timeout -k 30 {int(timeout_seconds)} "
        f"bash -lc '{AGENT_BODY}'"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(run_dir: Path, status: dict) -> None:
    status["updated_at"] = _utc_now()
    (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def _set_error(status: dict, error: BaseException | str) -> None:
    # Never replace the original failure with a later one (e.g. a missing artifact).
    if status["error"] is None:
        text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        status["error"] = text
        status["outcome"] = "failure"


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


@contextmanager
def _phase(status: dict, run_dir: Path, name: str) -> Iterator[dict]:
    entry = {"start": _utc_now()}
    status.setdefault("phases", {})[name] = entry
    status["phase"] = name
    _write_status(run_dir, status)
    try:
        yield entry
    finally:
        entry["end"] = _utc_now()


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


def build_jobs(
    rows: list[dict[str, str]], models: list[str] = MODELS
) -> list[tuple[dict[str, str], str]]:
    """The full task x model matrix, model-major: every task for model 1, then model 2."""
    return [(row, model) for model in models for row in rows]


async def run_pool(
    jobs: list[tuple[dict[str, str], str]],
    daytona: AsyncDaytona,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = RUN_TIMEOUT_SECONDS + POOL_GRACE_SECONDS,
    runner: Callable[..., any] | None = None,
) -> list[bool]:
    """Run jobs through at most ``batch_size`` sandboxes at a time.

    The semaphore is released the moment a run finishes, so the next queued job starts then; no
    batch barrier. A job that raises or hangs is recorded False and cannot stop its siblings: its
    own cleanup still runs, because cancelling ``run_task`` executes its delete-before-exit.
    Outcomes come back in job order.
    """
    run_one = runner or (
        lambda row, client, root, model_name: run_task(
            row, client, root, model_name
        )
    )
    slots = asyncio.Semaphore(max(1, batch_size))

    async def slot(row: dict[str, str], model_name: str) -> bool:
        async with slots:
            try:
                return await asyncio.wait_for(
                    run_one(row, daytona, output_root, model_name), timeout
                )
            except Exception:
                return False

    return list(
        await asyncio.gather(*(slot(row, model_name) for row, model_name in jobs))
    )


def _batch_size(env: Mapping[str, str]) -> int:
    """Pool width from BATCH_SIZE. Absent/blank means the default; anything else must be >=1."""
    raw = (env.get("BATCH_SIZE") or "").strip()
    if not raw:
        return DEFAULT_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"BATCH_SIZE must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"BATCH_SIZE must be a positive integer, got {value}")
    return value


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
            build_jobs(rows), daytona, output_root, batch_size=batch_size
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
