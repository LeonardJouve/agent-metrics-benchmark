"""Shared by every experiment: dataset loading, the task x model pool, run status bookkeeping,
the mini-swe-agent config and the metrics collector (metrics.sh) paths and summary.

Runners: miniswe_daytona/benchmark.py (Daytona) and local_harness/local_benchmark.py (local Docker).
"""
import asyncio
import csv
import json
import statistics
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SHARED_DIR / "dataset" / "sample20.csv"
MODELS = [
    "qwen3.8-flash",
    "deepseek-v4-flash-0731",
]
MODEL_API_BASE = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
MINI_VERSION = "2.4.6"
REQUIRED_FIELDS = ("instance_id", "image", "base_commit", "problem_statement")

# Pool sizing and the wall-clock cap. The cap is enforced twice: remotely, by wrapping the agent
# command in coreutils `timeout` so the artifact export still runs, and locally, by cancelling a
# job whose transport has hung. The local belt gets grace on top so the remote kill is what
# normally fires.
DEFAULT_BATCH_SIZE = 2
RUN_TIMEOUT_SECONDS = 45 * 60
POOL_GRACE_SECONDS = 300
# `timeout` exit codes: 124 when TERM fires at the cap, 137 when the -k grace escalates to KILL.
CENSORED_EXIT_CODES = frozenset({124, 137})

SAMPLE_INTERVAL_SECONDS = 1
METRICS_SCRIPT_PATH = SHARED_DIR / "metrics.sh"
METRICS_REMOTE_OUT = "/tmp/metrics.jsonl"
METRICS_ARTIFACT = "metrics.jsonl"

TRAJECTORY_PATH = "/tmp/minisweagent/trajectory.traj.json"
OVERRIDE_PATH = "/tmp/minisweagent/override.yaml"

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
    """mini-swe-agent config (exps 1 and 3); exp 3 swaps the environment class."""
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


def build_jobs(
    rows: list[dict[str, str]], models: list[str] = MODELS
) -> list[tuple[dict[str, str], str]]:
    """The full task x model matrix, model-major: every task for model 1, then model 2."""
    return [(row, model) for model in models for row in rows]


async def run_pool(
    jobs: list[tuple[dict[str, str], str]],
    runner: Callable[[dict[str, str], str], Awaitable[bool]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = RUN_TIMEOUT_SECONDS + POOL_GRACE_SECONDS,
) -> list[bool]:
    """Run ``runner(row, model)`` for every job, at most ``batch_size`` at a time.

    The semaphore is released the moment a run finishes, so the next queued job starts then; no
    batch barrier. A job that raises or hangs is recorded False and cannot stop its siblings: its
    own cleanup still runs, because cancelling the runner executes its delete-before-exit.
    Outcomes come back in job order.
    """
    slots = asyncio.Semaphore(max(1, batch_size))

    async def slot(row: dict[str, str], model_name: str) -> bool:
        async with slots:
            try:
                return await asyncio.wait_for(runner(row, model_name), timeout)
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
