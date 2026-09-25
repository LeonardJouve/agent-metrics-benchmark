import asyncio
import csv
import json
import shlex
from datetime import datetime
from pathlib import Path

import pytest

import benchmark


def task_row() -> dict[str, str]:
    return {
        "instance_id": "owner__repo-1",
        "image": "swebench/task:latest",
        "base_commit": "abc123",
        "problem_statement": "Fix the issue",
    }


def test_load_rows_requires_data(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("instance_id,image,base_commit,problem_statement\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset has no rows"):
        benchmark.load_rows(path)


def test_load_rows_reads_oversized_optional_cell(tmp_path: Path):
    """A >128 KiB cell must not break loading; required fields stay intact."""
    huge = "p" * (200 * 1024)
    path = tmp_path / "big.csv"
    fields = [*benchmark.REQUIRED_FIELDS, "test_patch"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**task_row(), "test_patch": huge})

    row = benchmark.load_rows(path)[0]

    assert len(row["test_patch"]) > 128 * 1024
    assert {key: row[key] for key in benchmark.REQUIRED_FIELDS} == task_row()


@pytest.fixture
def stdlib_csv_limit():
    """Pin a known baseline so restoration is asserted, not order-dependent."""
    previous = csv.field_size_limit()
    csv.field_size_limit(131072)
    yield 131072
    csv.field_size_limit(previous)


def test_load_rows_restores_field_size_limit_after_oversized_load(
    tmp_path: Path, stdlib_csv_limit: int
):
    before = csv.field_size_limit()
    assert before == stdlib_csv_limit
    path = tmp_path / "big.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*benchmark.REQUIRED_FIELDS, "test_patch"])
        writer.writeheader()
        writer.writerow({**task_row(), "test_patch": "q" * (200 * 1024)})

    assert len(benchmark.load_rows(path)[0]["test_patch"]) > 128 * 1024
    assert csv.field_size_limit() == before


def test_load_rows_restores_field_size_limit_after_validation_failure(
    tmp_path: Path, stdlib_csv_limit: int
):
    before = csv.field_size_limit()
    path = tmp_path / "big_bad.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(benchmark.REQUIRED_FIELDS))
        writer.writeheader()
        writer.writerow({**task_row(), "image": ""})

    with pytest.raises(ValueError, match="image"):
        benchmark.load_rows(path)
    assert csv.field_size_limit() == before


def test_load_rows_restores_field_size_limit_after_open_failure(
    tmp_path: Path, stdlib_csv_limit: int
):
    before = csv.field_size_limit()
    with pytest.raises(OSError):
        benchmark.load_rows(tmp_path / "missing.csv")
    assert csv.field_size_limit() == before


def test_load_rows_requires_fields(tmp_path: Path):
    path = tmp_path / "missing.csv"
    path.write_text("instance_id,image,base_commit\nabc,t,123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="problem_statement"):
        benchmark.load_rows(path)


@pytest.mark.parametrize("field", ["instance_id", "image", "base_commit", "problem_statement"])
def test_make_agent_config_requires_fields(field: str):
    row = task_row()
    row[field] = ""
    with pytest.raises(ValueError, match=field):
        benchmark.make_agent_config(row)


def test_make_agent_config_is_local_noninteractive_and_secret_free():
    config = json.loads(benchmark.make_agent_config(task_row()))
    assert config["run"]["task"] == "Fix the issue"
    assert config["agent"]["step_limit"] == 250
    assert config["agent"]["cost_limit"] == 0
    assert config["environment"] == {"environment_class": "local", "cwd": "/testbed", "timeout": 60}
    assert config["model"]["model_name"] == "qwen3.8-flash"
    assert config["model"]["model_kwargs"]["custom_llm_provider"] == "openai"
    assert "OPENAI_API_KEY" not in benchmark.make_agent_config(task_row()).decode()


def test_sandbox_params_use_task_image_resources_and_daytona_secret():
    params = benchmark.sandbox_params(task_row())
    assert params.image == "swebench/task:latest"
    assert (params.resources.cpu, params.resources.memory, params.resources.disk) == (4, 4, 10)
    assert params.secrets == {"OPENAI_API_KEY": "qwen-token-plan"}


from types import SimpleNamespace

# Two complete samples plus a torn trailing line: the collector is still appending
# when the file is downloaded, so a partial last line must be skipped, not crash.
METRICS_JSONL = (
    b'{"ts":"2026-09-25T16:50:40Z","epoch":1000.0,"usage_usec":10}\n'
    b'{"ts":"2026-09-25T16:50:41Z","epoch":1001.0,"usage_usec":20}\n'
    b'{"ts":"2026-09-25T16:50:4'
)


class FakeFS:
    def __init__(
        self,
        download_error: Exception | None = None,
        journal: list | None = None,
        upload_error: Exception | None = None,
        download_errors: dict[str, Exception] | None = None,
        upload_errors: dict[str, Exception] | None = None,
    ):
        self.uploads = {}
        self.downloads = {
            "/tmp/minisweagent/trajectory.traj.json": b'{"exit_status":"Submitted"}',
            "/tmp/metrics.jsonl": METRICS_JSONL,
        }
        self.download_error = download_error
        self.download_errors = download_errors or {}
        self.upload_error = upload_error
        self.upload_errors = upload_errors or {}
        self.journal = journal

    async def upload_file(self, source, destination):
        if destination in self.upload_errors:
            raise self.upload_errors[destination]
        if self.upload_error is not None:
            raise self.upload_error
        self.uploads[destination] = source
        if self.journal is not None:
            self.journal.append(("upload", destination))

    async def download_file(self, path):
        if path in self.download_errors:
            raise self.download_errors[path]
        if self.download_error is not None:
            raise self.download_error
        return self.downloads[path]


class FakeProcess:
    def __init__(
        self,
        failures: dict[str, int] | None = None,
        journal: list | None = None,
        errors: dict[str, Exception] | None = None,
    ):
        self.commands = []
        self.failures = failures or {}
        self.errors = errors or {}
        self.journal = journal

    async def exec(self, command, cwd=None, env=None, timeout=None):
        self.commands.append((command, cwd, env, timeout))
        if self.journal is not None:
            self.journal.append(("exec", command))
        for prefix, error in self.errors.items():
            if command.startswith(prefix):
                raise error
        for prefix, exit_code in self.failures.items():
            if command.startswith(prefix):
                return SimpleNamespace(exit_code=exit_code, result="boom\n")
        if command.startswith("git -C /testbed diff"):
            return SimpleNamespace(exit_code=0, result="diff --git a/a.py b/a.py\n")
        return SimpleNamespace(exit_code=0, result="ok\n")


class FakeSandbox:
    id = "sandbox-1"

    def __init__(self, journal: list | None = None):
        self.fs = FakeFS(journal=journal)
        self.process = FakeProcess(journal=journal)


class FakeSecret:
    def __init__(self, name: str, id: str = "secret-id", **extra):
        self.name = name
        self.id = id
        for key, value in extra.items():
            setattr(self, key, value)


class FakeSecretService:
    def __init__(
        self,
        secrets: list | None = None,
        list_error: Exception | None = None,
        create_error: Exception | None = None,
        update_error: Exception | None = None,
    ):
        self.pages = [secrets or []]
        self.list_error = list_error
        self.create_error = create_error
        self.update_error = update_error
        self.list_calls: list = []
        self.created: list = []
        self.updated: list = []

    async def list(self, name=None, **kwargs):
        self.list_calls.append(name)
        if self.list_error is not None:
            raise self.list_error
        return SimpleNamespace(items=self.pages[0])

    async def create(self, params):
        if self.create_error is not None:
            raise self.create_error
        self.created.append(params)
        return FakeSecret(params.name, id="new-id")

    async def update(self, secret_id, params):
        if self.update_error is not None:
            raise self.update_error
        self.updated.append((secret_id, params))
        return FakeSecret("qwen-token-plan", id=secret_id)


class FakeDaytona:
    def __init__(
        self,
        process_failures: dict[str, int] | None = None,
        download_error: Exception | None = None,
        delete_error: Exception | None = None,
        create_error: Exception | None = None,
        process_errors: dict[str, Exception] | None = None,
        upload_error: Exception | None = None,
        secret_service: FakeSecretService | None = None,
        download_errors: dict[str, Exception] | None = None,
        upload_errors: dict[str, Exception] | None = None,
    ):
        self.secret = secret_service or FakeSecretService()
        self.journal: list = []
        self.sandbox = FakeSandbox(journal=self.journal)
        self.sandbox.fs = FakeFS(
            download_error=download_error,
            journal=self.journal,
            upload_error=upload_error,
            download_errors=download_errors,
            upload_errors=upload_errors,
        )
        self.sandbox.process = FakeProcess(
            failures=process_failures, journal=self.journal, errors=process_errors
        )
        self.created_with = None
        self.created_timeout = None
        self.deleted = False
        self.closed = False
        self.delete_error = delete_error
        self.create_error = create_error

    async def create(self, params, timeout=60):
        if self.create_error is not None:
            raise self.create_error
        self.created_with = params
        self.created_timeout = timeout
        return self.sandbox

    async def delete(self, sandbox, timeout=60, wait=False):
        assert sandbox is self.sandbox
        assert wait is True
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted = True

    async def close(self):
        self.closed = True


def test_run_task_exports_artifacts_and_deletes_sandbox(tmp_path: Path):
    daytona = FakeDaytona()

    succeeded = asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path))

    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    assert succeeded is True
    assert (run_dir / "install.log").read_text() == "ok\n"
    assert (run_dir / "agent.log").read_text() == "ok\n"
    assert (run_dir / "trajectory.traj.json").exists()
    assert (run_dir / "patch.diff").read_text().startswith("diff --git")
    status = json.loads((run_dir / "status.json").read_text())
    assert status["outcome"] == "success"
    assert status["sandbox_id"] == "sandbox-1"
    assert status["cleanup"] == "deleted"
    assert daytona.deleted is True


def read_status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


def assert_terminal_failure(daytona: FakeDaytona, status: dict, run_dir: Path):
    assert daytona.deleted is True  # install/agent/export failure
    assert status["phase"] in {"install", "agent", "export", "cleanup"}
    assert status["outcome"] == "failure"
    assert status["error"] is not None
    assert "OPENAI_API_KEY" not in json.dumps(status)
    assert "qwen-token-plan" not in json.dumps(status)
    for log in ("install.log", "agent.log"):
        path = run_dir / log
        if path.exists():
            assert "qwen-token-plan" not in path.read_text(encoding="utf-8")


INSTALL_PREFIX = "python3 -m venv /opt/minisweagent"
AGENT_PREFIX = "mkdir -p /tmp/minisweagent && timeout -k 30"


def test_run_task_success_records_timed_status_and_quotes_commit(tmp_path: Path):
    daytona = FakeDaytona()
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is True
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    for key in ("started_at", "finished_at", "updated_at"):
        datetime.fromisoformat(status[key])
    assert status["model_name"] == benchmark.MODEL_NAME
    assert set(status["phases"]) == {"create", "install", "agent", "export", "cleanup"}
    for entry in status["phases"].values():
        datetime.fromisoformat(entry["start"])
        datetime.fromisoformat(entry["end"])
    commands = [c[0] for c in daytona.sandbox.process.commands]
    agent_call = next(c for c in daytona.sandbox.process.commands if c[0].startswith(AGENT_PREFIX))
    assert agent_call[3] is None  # no wall-clock timeout for the agent
    assert f"git -C /testbed diff --binary {shlex.quote('abc123')}" in commands
    assert daytona.sandbox.fs.uploads["/tmp/minisweagent/override.yaml"] == (
        benchmark.make_agent_config(task_row())
    )
    assert status["artifacts"] == {
        "install_log": True,
        "agent_log": True,
        "trajectory": True,
        "patch_diff": True,
        "metrics": True,
    }


def test_run_task_agent_command_exports_mini_env_and_keeps_run_flags(tmp_path: Path):
    daytona = FakeDaytona()

    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is True

    agent_command = next(command for command, *_ in daytona.sandbox.process.commands if command.startswith(AGENT_PREFIX))
    assert (
        "export MSWEA_CONFIGURED=true; export MSWEA_COST_TRACKING=ignore_errors; "
        "exec /opt/minisweagent/bin/mini "
    ) in agent_command
    assert agent_command.index("export MSWEA_CONFIGURED=true;") < agent_command.index(
        "export MSWEA_COST_TRACKING=ignore_errors;"
    ) < agent_command.index("exec /opt/minisweagent/bin/mini")
    assert "-c swebench.yaml -c /tmp/minisweagent/override.yaml " in agent_command
    assert "-y --exit-immediately " in agent_command
    assert "-o /tmp/minisweagent/trajectory.traj.json" in agent_command


def test_run_task_install_failure_skips_config_agent_and_patch(tmp_path: Path):
    daytona = FakeDaytona(process_failures={INSTALL_PREFIX: 1})
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert_terminal_failure(daytona, status, run_dir)
    assert status["phase"] == "install"
    assert set(status["phases"]) == {"create", "install", "cleanup"}
    assert status["model_name"] == benchmark.MODEL_NAME
    assert status["exit_codes"]["install"] == 1
    commands = [c[0] for c in daytona.sandbox.process.commands]
    assert not any(c.startswith(AGENT_PREFIX) for c in commands)
    assert not any(c.startswith("git -C") for c in commands)
    assert set(daytona.sandbox.fs.uploads) == {"/tmp/metrics.sh"}
    assert not (run_dir / "agent.log").exists()


def test_run_task_agent_failure_still_generates_patch(tmp_path: Path):
    daytona = FakeDaytona(process_failures={AGENT_PREFIX: 3})
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert_terminal_failure(daytona, status, run_dir)
    assert status["exit_codes"]["agent"] == 3
    assert set(status["phases"]) == {"create", "install", "agent", "export", "cleanup"}
    assert "agent exited with code 3" in status["error"]
    assert (run_dir / "patch.diff").read_text().startswith("diff --git")
    assert (run_dir / "agent.log").read_text() == "boom\n"


def test_run_task_export_failure_when_trajectory_missing_after_agent_success(tmp_path: Path):
    daytona = FakeDaytona(download_error=FileNotFoundError("no trajectory"))
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert_terminal_failure(daytona, status, run_dir)
    assert status["phase"] == "export"
    assert status["error"] == "FileNotFoundError: no trajectory"
    assert status["artifacts"]["patch_diff"] is True
    assert status["artifacts"]["trajectory"] is False
    assert not (run_dir / "trajectory.traj.json").exists()


def test_run_task_agent_failure_keeps_original_error_when_trajectory_missing(tmp_path: Path):
    daytona = FakeDaytona(
        process_failures={AGENT_PREFIX: 4}, download_error=FileNotFoundError("no trajectory")
    )
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    status = read_status(tmp_path / "qwen3.8-flash" / "owner__repo-1")
    assert "agent exited with code 4" in status["error"]


def test_run_task_creates_trajectory_dir_before_upload(tmp_path: Path):
    daytona = FakeDaytona()
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is True
    journal = daytona.journal
    mkdir_call = ("exec", "mkdir -p /tmp/minisweagent")
    upload_call = ("upload", "/tmp/minisweagent/override.yaml")
    assert mkdir_call in journal
    assert journal.index(mkdir_call) < journal.index(upload_call)


def test_run_task_patch_failure_is_export_failure(tmp_path: Path):
    daytona = FakeDaytona(process_failures={"git -C /testbed diff": 2})
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert_terminal_failure(daytona, status, run_dir)
    assert status["phase"] == "export"
    assert "patch exited with code 2" in status["error"]
    assert status["exit_codes"]["patch"] == 2
    assert status["artifacts"]["patch_diff"] is False
    assert (run_dir / "patch.diff").read_text() == "boom\n"
    assert set(status["phases"]) == {"create", "install", "agent", "export", "cleanup"}


def test_run_task_cleanup_failure_marks_failure(tmp_path: Path):
    daytona = FakeDaytona(delete_error=RuntimeError("delete refused"))
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert status["cleanup"] == "failed"
    assert status["phase"] == "cleanup"
    assert status["outcome"] == "failure"
    assert status["error"] == "RuntimeError: delete refused"
    assert (run_dir / "trajectory.traj.json").exists()  # artifacts survived


def test_run_task_create_failure_skips_delete_and_still_writes_status(tmp_path: Path):
    daytona = FakeDaytona(create_error=RuntimeError("quota"))
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert status["phase"] == "create"
    assert set(status["phases"]) == {"create"}
    assert status["outcome"] == "failure"
    assert status["cleanup"] == "skipped"
    assert status["sandbox_id"] is None
    assert daytona.deleted is False


def test_run_task_create_disables_daytona_timeout(tmp_path: Path):
    # daytona 0.217.0: create() defaults to timeout=60; timeout=0 disables the
    # sandbox-start wait, so the create phase must pass it explicitly.
    daytona = FakeDaytona()

    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is True

    assert daytona.created_with == benchmark.sandbox_params(task_row())
    assert daytona.created_timeout == 0


def test_make_agent_config_pins_exact_model_and_agent_settings():
    assert json.loads(benchmark.make_agent_config(task_row())) == {
        "run": {"task": "Fix the issue"},
        "agent": {"step_limit": 250, "cost_limit": 0, "confirm_exit": False},
        "environment": {"environment_class": "local", "cwd": "/testbed", "timeout": 60},
        "model": {
            "model_name": "qwen3.8-flash",
            "model_class": "litellm",
            "model_kwargs": {
                "custom_llm_provider": "openai",
                "api_base": "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
                "drop_params": True,
            },
        },
    }


def test_run_task_patch_transport_error_still_downloads_trajectory(tmp_path: Path):
    daytona = FakeDaytona(process_errors={"git -C /testbed diff": RuntimeError("transport")})
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert_terminal_failure(daytona, status, run_dir)
    assert status["error"] == "RuntimeError: transport"
    assert status["artifacts"]["patch_diff"] is False
    assert status["artifacts"]["trajectory"] is True
    assert (run_dir / "trajectory.traj.json").exists()
    assert "patch" not in status["exit_codes"]


def test_run_task_install_transport_error_stops_before_agent(tmp_path: Path):
    daytona = FakeDaytona(process_errors={INSTALL_PREFIX: RuntimeError("connection reset")})
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    status = read_status(tmp_path / "qwen3.8-flash" / "owner__repo-1")
    assert_terminal_failure(daytona, status, tmp_path / "qwen3.8-flash" / "owner__repo-1")
    assert status["phase"] == "install"
    assert status["error"] == "RuntimeError: connection reset"
    commands = [c[0] for c in daytona.sandbox.process.commands]
    assert not any(c.startswith(AGENT_PREFIX) for c in commands)
    assert not any(c.startswith("git -C") for c in commands)


def test_run_task_agent_transport_error_still_exports_patch(tmp_path: Path):
    daytona = FakeDaytona(process_errors={AGENT_PREFIX: RuntimeError("dropped")})
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert_terminal_failure(daytona, status, run_dir)
    assert status["error"] == "RuntimeError: dropped"
    assert (run_dir / "patch.diff").read_text().startswith("diff --git")
    assert not (run_dir / "agent.log").exists()


def test_run_task_first_error_survives_patch_and_trajectory_failures(tmp_path: Path):
    daytona = FakeDaytona(
        process_failures={AGENT_PREFIX: 3},
        process_errors={"git -C /testbed diff": RuntimeError("transport")},
        download_error=FileNotFoundError("no trajectory"),
    )
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert_terminal_failure(daytona, status, run_dir)
    assert status["error"] == "RuntimeError: agent exited with code 3"
    assert status["exit_codes"]["agent"] == 3
    assert "patch" not in status["exit_codes"]
    assert status["artifacts"] == {
        "install_log": True,
        "agent_log": True,
        "trajectory": False,
        "patch_diff": False,
        "metrics": False,
    }
    assert not (run_dir / "patch.diff").exists()


def test_run_task_upload_error_skips_agent_and_still_exports(tmp_path: Path):
    daytona = FakeDaytona(
        upload_errors={"/tmp/minisweagent/override.yaml": RuntimeError("upload refused")}
    )
    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False
    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert_terminal_failure(daytona, status, run_dir)
    assert status["error"] == "RuntimeError: upload refused"
    commands = [c[0] for c in daytona.sandbox.process.commands]
    assert not any(c.startswith(AGENT_PREFIX) for c in commands)
    assert (run_dir / "patch.diff").exists()


def write_dataset(tmp_path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    path = tmp_path / "tasks.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(task_row()))
        writer.writeheader()
        for row in rows if rows is not None else [task_row()]:
            writer.writerow(row)
    return path


def test_run_benchmark_requires_daytona_key_before_client_creation(tmp_path):
    dataset = write_dataset(tmp_path, rows=[])
    root = tmp_path / "runs"
    called = []

    def factory(config):
        called.append(config)
        return FakeDaytona()

    with pytest.raises(ValueError, match="DAYTONA_API_KEY"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(dataset)}, client_factory=factory, output_root=root
            )
        )
    assert called == []
    assert not root.exists()


@pytest.mark.parametrize("key", [None, ""])
def test_run_benchmark_rejects_missing_or_blank_key(tmp_path, key):
    env = {"DATASET_PATH": str(write_dataset(tmp_path))}
    if key is not None:
        env["DAYTONA_API_KEY"] = key
    with pytest.raises(ValueError, match="DAYTONA_API_KEY"):
        asyncio.run(
            benchmark.run_benchmark(
                env, client_factory=lambda config: FakeDaytona(), output_root=tmp_path / "runs"
            )
        )
    assert not (tmp_path / "runs").exists()


def test_run_benchmark_builds_the_matrix_and_closes_the_client(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path, rows=[task_row(), {**task_row(), "instance_id": "other-2"}])
    seen: dict = {}

    async def fake_run_task(row, daytona, output_root, model_name):
        seen["row"] = row
        seen["daytona"] = daytona
        seen.setdefault("jobs", []).append((row["instance_id"], model_name))
        seen["calls"] = seen.get("calls", 0) + 1
        return True

    configs: list = []
    client = FakeDaytona()
    monkeypatch.setattr(benchmark, "run_task", fake_run_task)

    def factory(config):
        configs.append(config)
        return client

    result = asyncio.run(
        benchmark.run_benchmark(
            {"DATASET_PATH": str(dataset), "DAYTONA_API_KEY": "daytona-key", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"},
            client_factory=factory,
        )
    )

    assert result is True
    assert seen["daytona"] is client
    assert seen["calls"] == 4
    assert seen["jobs"] == [
        ("owner__repo-1", "qwen3.8-flash"),
        ("other-2", "qwen3.8-flash"),
        ("owner__repo-1", "deepseek-v4-flash-0731"),
        ("other-2", "deepseek-v4-flash-0731"),
    ]
    assert [c.api_key for c in configs] == ["daytona-key"]
    assert client.closed is True


def test_run_benchmark_uses_default_dataset_path(monkeypatch):
    captured: list = []

    def fake_load(path):
        captured.append(path)
        return [task_row()]

    async def fake_run_task(row, daytona, output_root, model_name):
        return True

    monkeypatch.setattr(benchmark, "load_rows", fake_load)
    monkeypatch.setattr(benchmark, "run_task", fake_run_task)

    assert asyncio.run(
        benchmark.run_benchmark({"DAYTONA_API_KEY": "k", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"}, client_factory=lambda c: FakeDaytona())
    ) is True
    assert captured == [Path("data/sample20.csv")]


def test_run_benchmark_closes_client_when_a_run_raises(tmp_path, monkeypatch):
    async def explode(row, daytona, output_root, model_name):
        raise RuntimeError("agent boom")

    client = FakeDaytona()
    monkeypatch.setattr(benchmark, "run_task", explode)

    # A raising run is isolated by the pool: the benchmark finishes and reports failure.
    assert asyncio.run(
        benchmark.run_benchmark(
                {
                    "DATASET_PATH": str(write_dataset(tmp_path)),
                    "DAYTONA_API_KEY": "k",
                    "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token",
                },
                client_factory=lambda config: client,
            )
    ) is False
    assert client.closed is True


def test_main_returns_nonzero_when_run_fails(monkeypatch):
    async def fail(*args, **kwargs):
        return False

    monkeypatch.setattr(benchmark, "run_benchmark", fail)
    assert benchmark.main() == 1


def test_main_returns_zero_on_success(monkeypatch):
    async def ok(*args, **kwargs):
        return True

    monkeypatch.setattr(benchmark, "run_benchmark", ok)
    assert benchmark.main() == 0


def test_main_reports_exception_and_returns_nonzero(monkeypatch, capsys):
    async def explode(*args, **kwargs):
        raise ValueError("DAYTONA_API_KEY is required")

    monkeypatch.setattr(benchmark, "run_benchmark", explode)
    assert benchmark.main() == 1
    assert "DAYTONA_API_KEY" in capsys.readouterr().err


def test_default_run_output_dir_is_gitignored():
    from inspect import signature

    default = signature(benchmark.run_task).parameters["output_root"].default
    ignore_path = Path(__file__).resolve().parents[1] / ".gitignore"
    entries = ignore_path.read_text(encoding="utf-8").split()
    assert f"{default}/" in entries


def validation_status(root: Path) -> dict:
    return read_status(root / "validation-error")


def test_run_benchmark_writes_validation_status_without_client(tmp_path, monkeypatch):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "instance_id,image,base_commit,problem_statement\n"
        "owner__repo-1,,abc123,Fix the issue\n",
        encoding="utf-8",
    )
    created: list = []
    root = tmp_path / "runs"

    def factory(config):
        created.append(config)
        return FakeDaytona()

    with pytest.raises(ValueError, match="image"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(bad), "DAYTONA_API_KEY": "secret-key", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"},
                client_factory=factory,
                output_root=root,
            )
        )
    assert created == []
    status = validation_status(root)
    assert status["phase"] == "validate"
    assert status["outcome"] == "failure"
    assert status["model_name"] == "qwen3.8-flash"
    assert status["sandbox_id"] is None
    assert status["cleanup"] == "not_started"
    assert status["error"].startswith("DatasetValidationError:")
    assert "row missing required fields: image" in status["error"]
    assert "secret-key" not in json.dumps(status)
    assert status["artifacts"] == {
        "install_log": False,
        "agent_log": False,
        "trajectory": False,
        "patch_diff": False,
        "metrics": False,
    }
    datetime.fromisoformat(status["started_at"])
    datetime.fromisoformat(status["finished_at"])
    datetime.fromisoformat(status["phases"]["validate"]["start"])
    datetime.fromisoformat(status["phases"]["validate"]["end"])
    assert status["exit_codes"] == {}
    assert list((root / "validation-error").iterdir()) == [root / "validation-error" / "status.json"]


def test_load_rows_error_is_a_value_error_carrying_instance_id(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "\n".join(
            [
                "instance_id,image,base_commit,problem_statement",
                "owner__repo-1,,abc123,Fix the issue",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="image") as caught:
        benchmark.load_rows(path)
    assert isinstance(caught.value, benchmark.DatasetValidationError)
    assert caught.value.instance_id == "owner__repo-1"


def test_load_rows_empty_dataset_error_carries_no_instance_id(tmp_path):
    path = write_dataset(tmp_path, rows=[])
    with pytest.raises(ValueError, match="no rows") as caught:
        benchmark.load_rows(path)
    assert not isinstance(caught.value, benchmark.DatasetValidationError)
    assert getattr(caught.value, "instance_id", None) is None


def test_validation_status_records_invalid_row_instance_id(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "\n".join(
            [
                "instance_id,image,base_commit,problem_statement",
                "owner__repo-1,,abc123,Fix the issue",
                "",
            ]
        ),
        encoding="utf-8",
    )
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="image"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(bad), "DAYTONA_API_KEY": "k", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"},
                client_factory=lambda config: FakeDaytona(),
                output_root=root,
            )
        )
    assert validation_status(root)["instance_id"] == "owner__repo-1"


def test_validation_directory_ignores_hostile_instance_id(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "\n".join(
            [
                "instance_id,image,base_commit,problem_statement",
                '"../../escape",,abc,task',
                "",
            ]
        ),
        encoding="utf-8",
    )
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="image"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(bad), "DAYTONA_API_KEY": "k", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"},
                client_factory=lambda config: FakeDaytona(),
                output_root=root,
            )
        )
    status = validation_status(root)
    assert status["instance_id"] == "../../escape"
    assert [p.name for p in root.iterdir()] == ["validation-error"]
    assert list(root.iterdir()) == [root / "validation-error"]
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path.parent / "escape").exists()
    assert [d.name for d in tmp_path.rglob("escape")] == []
    assert list((root / "validation-error").iterdir()) == [root / "validation-error" / "status.json"]


def test_validation_status_has_no_instance_id_for_empty_or_unreadable_dataset(tmp_path):
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="no rows"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(write_dataset(tmp_path, rows=[])), "DAYTONA_API_KEY": "k", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"},
                client_factory=lambda config: FakeDaytona(),
                output_root=root,
            )
        )
    assert validation_status(root)["instance_id"] is None
    with pytest.raises(OSError):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(tmp_path / "gone.csv"), "DAYTONA_API_KEY": "k", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"},
                client_factory=lambda config: FakeDaytona(),
                output_root=root,
            )
        )
    assert validation_status(root)["instance_id"] is None


def test_run_benchmark_validation_status_for_empty_dataset(tmp_path):
    empty = write_dataset(tmp_path, rows=[])
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="no rows"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(empty), "DAYTONA_API_KEY": "k", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"},
                client_factory=lambda config: FakeDaytona(),
                output_root=root,
            )
        )
    assert validation_status(root)["error"].startswith("ValueError:")


def test_run_benchmark_writes_status_for_unreadable_dataset(tmp_path):
    root = tmp_path / "runs"
    with pytest.raises(OSError):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(tmp_path / "gone.csv"), "DAYTONA_API_KEY": "k", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"},
                client_factory=lambda config: FakeDaytona(),
                output_root=root,
            )
        )
    status = validation_status(root)
    assert status["phase"] == "validate"
    assert status["error"].startswith(("FileNotFoundError:", "OSError:"))


def env_with_tokens(dataset=None, **overrides) -> dict:
    env = {"DAYTONA_API_KEY": "k", "QWEN_TOKEN_PLAN_API_KEY": "plaintext-token"}
    if dataset is not None:
        env["DATASET_PATH"] = str(dataset)
    env.update(overrides)
    return env


def test_run_benchmark_requires_qwen_token_before_client_creation(tmp_path):
    dataset = write_dataset(tmp_path)
    called: list = []

    def factory(config):
        called.append(config)
        return FakeDaytona()

    with pytest.raises(ValueError, match="QWEN_TOKEN_PLAN_API_KEY"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(dataset), "DAYTONA_API_KEY": "k"},
                client_factory=factory,
                output_root=tmp_path / "runs",
            )
        )
    assert called == []
    assert not (tmp_path / "runs").exists()


def test_run_benchmark_qwen_token_check_precedes_dataset_access(tmp_path):
    # The dataset does not exist: if it were read first the error would be OSError.
    with pytest.raises(ValueError, match="QWEN_TOKEN_PLAN_API_KEY"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(tmp_path / "missing.csv"), "DAYTONA_API_KEY": "k"},
                client_factory=lambda config: FakeDaytona(),
                output_root=tmp_path / "runs",
            )
        )
    assert not (tmp_path / "runs").exists()


def test_run_benchmark_daytona_key_check_precedes_qwen_token_check(tmp_path):
    called: list = []

    def factory(config):
        called.append(config)
        return FakeDaytona()

    with pytest.raises(ValueError, match="DAYTONA_API_KEY"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(tmp_path / "missing.csv")},
                client_factory=factory,
                output_root=tmp_path / "runs",
            )
        )
    assert called == []


@pytest.mark.parametrize("token", ["", "   "])
def test_run_benchmark_rejects_blank_qwen_token(tmp_path, token):
    dataset = write_dataset(tmp_path)
    with pytest.raises(ValueError, match="QWEN_TOKEN_PLAN_API_KEY"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(dataset), "DAYTONA_API_KEY": "k", "QWEN_TOKEN_PLAN_API_KEY": token},
                client_factory=lambda config: FakeDaytona(),
                output_root=tmp_path / "runs",
            )
        )
    assert not (tmp_path / "runs").exists()


def test_provision_secret_creates_when_absent_with_exact_args(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    service = FakeSecretService(secrets=[])
    client = FakeDaytona(secret_service=service)

    async def fake_run_task(row, daytona, output_root):
        return True

    monkeypatch.setattr(benchmark, "run_task", fake_run_task)
    asyncio.run(
        benchmark.run_benchmark(
            env_with_tokens(dataset),
            client_factory=lambda config: client,
            output_root=tmp_path / "runs",
        )
    )
    assert service.list_calls == ["qwen-token-plan"]
    assert len(service.created) == 1
    params = service.created[0]
    assert params.name == "qwen-token-plan"
    assert params.value == "plaintext-token"
    assert params.hosts == ["token-plan.ap-southeast-1.maas.aliyuncs.com"]
    assert service.updated == []


def test_provision_secret_updates_existing_exact_match(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    service = FakeSecretService(secrets=[FakeSecret("qwen-token-plan", id="sid-7")])
    client = FakeDaytona(secret_service=service)

    async def fake_run_task(row, daytona, output_root):
        return True

    monkeypatch.setattr(benchmark, "run_task", fake_run_task)
    asyncio.run(
        benchmark.run_benchmark(
            env_with_tokens(dataset),
            client_factory=lambda config: client,
            output_root=tmp_path / "runs",
        )
    )
    assert service.created == []
    assert len(service.updated) == 1
    secret_id, params = service.updated[0]
    assert secret_id == "sid-7"
    assert params.value == "plaintext-token"
    assert params.hosts == ["token-plan.ap-southeast-1.maas.aliyuncs.com"]


def test_provision_secret_filters_partial_name_matches(tmp_path, monkeypatch):
    # The API name filter is partial: rows that merely contain the name must not match.
    dataset = write_dataset(tmp_path)
    service = FakeSecretService(
        secrets=[FakeSecret("qwen-token-plan-extra", id="p1"), FakeSecret("xqwen-token-plan", id="p2")]
    )
    client = FakeDaytona(secret_service=service)

    async def fake_run_task(row, daytona, output_root):
        return True

    monkeypatch.setattr(benchmark, "run_task", fake_run_task)
    asyncio.run(
        benchmark.run_benchmark(
            env_with_tokens(dataset),
            client_factory=lambda config: client,
            output_root=tmp_path / "runs",
        )
    )
    assert service.updated == []
    assert len(service.created) == 1


def test_provision_secret_fails_loudly_on_multiple_exact_matches(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    service = FakeSecretService(
        secrets=[FakeSecret("qwen-token-plan", id="a"), FakeSecret("qwen-token-plan", id="b")]
    )
    client = FakeDaytona(secret_service=service)

    async def fake_run_task(row, daytona, output_root):
        raise AssertionError("run_task must not be reached")

    monkeypatch.setattr(benchmark, "run_task", fake_run_task)
    with pytest.raises(Exception) as caught:
        asyncio.run(
            benchmark.run_benchmark(
                env_with_tokens(dataset),
                client_factory=lambda config: client,
                output_root=tmp_path / "runs",
            )
        )
    assert not isinstance(caught.value, AssertionError)
    assert service.created == []
    assert service.updated == []
    assert "plaintext-token" not in str(caught.value)
    assert client.closed is True


def test_provision_failure_propagates_closes_client_and_skips_sandbox(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    service = FakeSecretService(list_error=RuntimeError("provider unavailable"))
    client = FakeDaytona(secret_service=service)

    async def never_run(row, daytona, output_root):
        raise AssertionError("run_task must not be reached")

    monkeypatch.setattr(benchmark, "run_task", never_run)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            benchmark.run_benchmark(
                env_with_tokens(dataset),
                client_factory=lambda config: client,
                output_root=tmp_path / "runs",
            )
        )
    assert client.created_with is None
    assert client.closed is True
    assert list(client.journal) == []
    assert not (tmp_path / "runs" / "owner__repo-1").exists()


def test_provision_plaintext_never_reaches_artifacts_logs_or_status(tmp_path):
    dataset = write_dataset(tmp_path)
    service = FakeSecretService(secrets=[])
    client = FakeDaytona(secret_service=service)

    succeeded = asyncio.run(
        benchmark.run_benchmark(
            env_with_tokens(dataset),
            client_factory=lambda config: client,
            output_root=tmp_path / "runs",
        )
    )
    assert succeeded is True
    assert service.created[0].value == "plaintext-token"  # the only place it may exist
    for path in (tmp_path / "runs").rglob("*"):
        if path.is_file():
            assert "plaintext-token" not in path.read_text(encoding="utf-8", errors="replace"), path
    for command, *_ in client.sandbox.process.commands:
        assert "plaintext-token" not in command


def test_run_benchmark_provisions_secret_before_running_task(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    service = FakeSecretService(secrets=[])
    client = FakeDaytona(secret_service=service)
    order: list = []

    async def fake_run_task(row, daytona, output_root, model_name):
        order.append("run_task")
        assert service.created, "secret must already be provisioned"
        return True

    monkeypatch.setattr(benchmark, "run_task", fake_run_task)
    asyncio.run(
        benchmark.run_benchmark(
            env_with_tokens(dataset),
            client_factory=lambda config: client,
            output_root=tmp_path / "runs",
        )
    )
    assert order == ["run_task"] * len(benchmark.MODELS)


def test_run_benchmark_key_check_precedes_dataset_access_and_writes_no_status(tmp_path):
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="DAYTONA_API_KEY"):
        asyncio.run(
            benchmark.run_benchmark(
                {"DATASET_PATH": str(tmp_path / "does-not-exist.csv")},
                client_factory=lambda config: FakeDaytona(),
                output_root=root,
            )
        )
    assert not root.exists()


def test_main_reports_validation_failure_concisely(monkeypatch, capsys, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "instance_id,image,base_commit,problem_statement\nowner__repo-1,,abc,task\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATASET_PATH", str(bad))
    monkeypatch.setenv("DAYTONA_API_KEY", "k")
    monkeypatch.setenv("QWEN_TOKEN_PLAN_API_KEY", "plaintext-token")
    monkeypatch.chdir(tmp_path)
    assert benchmark.main() == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("ERROR: DatasetValidationError:")
    assert "\nTraceback" not in captured.err
    assert (tmp_path / "runs" / "validation-error" / "status.json").exists()


METRICS_PREFIX = "nohup setsid bash /tmp/metrics.sh"
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "metrics.sh"


def start_call(daytona: FakeDaytona):
    return next(c for c in daytona.sandbox.process.commands if c[0].startswith(METRICS_PREFIX))


def test_run_task_uploads_and_starts_collector_before_install(tmp_path: Path):
    daytona = FakeDaytona()

    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is True

    journal = daytona.journal
    upload = ("upload", "/tmp/metrics.sh")
    start = ("exec", start_call(daytona)[0])
    install = ("exec", next(c for c in journal if c[0] == "exec" and c[1].startswith(INSTALL_PREFIX))[1])
    assert journal.index(upload) < journal.index(start) < journal.index(install)
    assert daytona.sandbox.fs.uploads["/tmp/metrics.sh"] == SCRIPT_PATH.read_bytes()
    # Detached: exec returns at once, the collector keeps sampling until sandbox delete.
    assert start[1].endswith("& echo started")


def test_run_task_configures_collector_interval_output_and_allocation(tmp_path: Path):
    daytona = FakeDaytona()

    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is True

    env = start_call(daytona)[2]
    assert env["OUT"] == "/tmp/metrics.jsonl"
    assert env["INTERVAL"] == str(benchmark.SAMPLE_INTERVAL_SECONDS) == "1"
    assert (env["ALLOC_CPU"], env["ALLOC_MEM_GIB"]) == ("4", "4")


def test_run_task_downloads_metrics_and_records_summary(tmp_path: Path):
    daytona = FakeDaytona()

    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is True

    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    assert (run_dir / "metrics.jsonl").read_bytes() == METRICS_JSONL
    status = read_status(run_dir)
    assert status["artifacts"]["metrics"] is True
    assert status["metrics"] == {
        "samples": 2,
        "first_ts": "2026-09-25T16:50:40Z",
        "last_ts": "2026-09-25T16:50:41Z",
        "interval_s": 1.0,
    }


def test_run_task_metrics_start_exception_fails_run(tmp_path: Path):
    daytona = FakeDaytona(process_errors={METRICS_PREFIX: RuntimeError("start boom")})

    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False

    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert status["phase"] == "create"
    assert status["outcome"] == "failure"
    assert status["error"] == "RuntimeError: start boom"
    assert status["artifacts"]["metrics"] is False
    assert status["cleanup"] == "deleted"
    assert not (run_dir / "install.log").exists()


def test_run_task_metrics_start_nonzero_exit_fails_run(tmp_path: Path):
    daytona = FakeDaytona(process_failures={METRICS_PREFIX: 3})

    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False

    status = read_status(tmp_path / "qwen3.8-flash" / "owner__repo-1")
    assert status["outcome"] == "failure"
    assert "metrics collector exited with code 3" in status["error"]
    assert status["artifacts"]["metrics"] is False


def test_run_task_metrics_download_failure_fails_run(tmp_path: Path):
    daytona = FakeDaytona(
        download_errors={"/tmp/metrics.jsonl": FileNotFoundError("no metrics")}
    )

    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False

    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    status = read_status(run_dir)
    assert status["phase"] == "export"
    assert status["outcome"] == "failure"
    assert status["error"] == "FileNotFoundError: no metrics"
    assert status["artifacts"]["metrics"] is False
    assert not (run_dir / "metrics.jsonl").exists()
    assert daytona.deleted is True


def test_metrics_summary_uses_median_interval():
    payload = b'{"ts":"a","epoch":10.0}\n{"ts":"b","epoch":10.5}\n{"ts":"c","epoch":12.0}\n'

    assert benchmark.metrics_summary(payload) == {
        "samples": 3,
        "first_ts": "a",
        "last_ts": "c",
        "interval_s": 1.0,
    }


def test_metrics_summary_skips_torn_last_line():
    assert benchmark.metrics_summary(METRICS_JSONL)["samples"] == 2


def test_metrics_summary_handles_empty_unparsable_and_single_sample():
    empty = {"samples": 0, "first_ts": None, "last_ts": None, "interval_s": None}
    assert benchmark.metrics_summary(b"") == empty
    assert benchmark.metrics_summary(b"not json\n") == empty
    assert benchmark.metrics_summary(b'{"ts":"a","epoch":1.0}\n') == {
        "samples": 1,
        "first_ts": "a",
        "last_ts": "a",
        "interval_s": None,
    }


# --- load_rows: full-dataset reads for the matrix runner -------------------


def write_rows(path: Path, rows: list[dict[str, str]], extra_field: str | None = None):
    fields = [*benchmark.REQUIRED_FIELDS] + ([extra_field] if extra_field else [])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_load_rows_returns_every_row_in_order(tmp_path: Path):
    path = tmp_path / "tasks.csv"
    rows = [task_row(), {**task_row(), "instance_id": "other__repo-2"}]
    write_rows(path, rows)

    assert benchmark.load_rows(path) == rows


def test_load_rows_requires_data(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("instance_id,image,base_commit,problem_statement\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset has no rows"):
        benchmark.load_rows(path)


def test_load_rows_rejects_duplicate_instance_id(tmp_path: Path):
    path = tmp_path / "dupes.csv"
    write_rows(path, [task_row(), task_row()])

    with pytest.raises(ValueError, match="duplicate instance_id"):
        benchmark.load_rows(path)


def test_load_rows_reports_the_offending_row_id(tmp_path: Path):
    path = tmp_path / "bad.csv"
    write_rows(path, [task_row(), {**task_row(), "instance_id": "bad__repo-3", "image": ""}])

    with pytest.raises(ValueError, match="image") as excinfo:
        benchmark.load_rows(path)
    assert excinfo.value.instance_id == "bad__repo-3"


def test_load_rows_restores_field_size_limit(tmp_path: Path, stdlib_csv_limit: int):
    before = csv.field_size_limit()
    path = tmp_path / "big.csv"
    write_rows(path, [task_row()], extra_field="test_patch")
    path.write_text(
        path.read_text(encoding="utf-8") + "x,y,z,w," + "q" * (200 * 1024) + "\n",
        encoding="utf-8",
    )

    assert len(benchmark.load_rows(path)[1]["test_patch"]) > 128 * 1024
    assert csv.field_size_limit() == before


# --- model matrix: one model per job, model-scoped artifact paths -----------


def test_models_are_pinned_and_ordered():
    assert benchmark.MODELS == ["qwen3.8-flash", "deepseek-v4-flash-0731"]


def test_make_agent_config_uses_the_requested_model():
    config = json.loads(
        benchmark.make_agent_config(task_row(), model_name="deepseek-v4-flash-0731")
    )
    assert config["model"]["model_name"] == "deepseek-v4-flash-0731"


def test_make_agent_config_defaults_to_the_first_model():
    config = json.loads(benchmark.make_agent_config(task_row()))
    assert config["model"]["model_name"] == benchmark.MODELS[0]


def test_run_task_writes_artifacts_under_the_model_directory(tmp_path: Path):
    daytona = FakeDaytona()

    asyncio.run(
        benchmark.run_task(
            task_row(), daytona, tmp_path, model_name="deepseek-v4-flash-0731"
        )
    )

    run_dir = tmp_path / "deepseek-v4-flash-0731" / "owner__repo-1"
    assert (run_dir / "agent.log").exists()
    assert read_status(run_dir)["model_name"] == "deepseek-v4-flash-0731"
    assert list(tmp_path.iterdir()) == [tmp_path / "deepseek-v4-flash-0731"]


# --- concurrency pool: N slots, next job starts when one frees --------------


def make_tracker():
    return {"live": 0, "peak": 0, "started": [], "finished": []}


async def recording_runner(row, daytona, output_root, model_name, tracker):
    tracker["live"] += 1
    tracker["peak"] = max(tracker["peak"], tracker["live"])
    tracker["started"].append((row["instance_id"], model_name))
    await asyncio.sleep(0.01)
    tracker["live"] -= 1
    tracker["finished"].append((row["instance_id"], model_name))
    return True


def pool_jobs(count: int = 5) -> list[tuple[dict[str, str], str]]:
    return [
        ({**task_row(), "instance_id": f"owner__repo-{index}"}, benchmark.MODELS[0])
        for index in range(count)
    ]


def run_pool_with(tracker, jobs, batch_size: int, runner=None, **kwargs):
    async def main():
        return await benchmark.run_pool(
            jobs,
            FakeDaytona(),
            Path("/unused"),
            batch_size=batch_size,
            runner=runner
            or (
                lambda row, daytona, output_root, model_name: recording_runner(
                    row, daytona, output_root, model_name, tracker
                )
            ),
            **kwargs,
        )

    return asyncio.run(main())


def test_run_pool_never_exceeds_batch_size():
    tracker = make_tracker()

    run_pool_with(tracker, pool_jobs(5), batch_size=2)

    assert tracker["peak"] == 2


def test_run_pool_starts_the_next_job_as_soon_as_a_slot_frees():
    tracker = make_tracker()

    run_pool_with(tracker, pool_jobs(5), batch_size=2)

    assert tracker["started"] == [(job[0]["instance_id"], job[1]) for job in pool_jobs(5)]
    assert tracker["finished"] == tracker["started"]


def test_run_pool_runs_every_job_when_batch_is_one():
    tracker = make_tracker()

    run_pool_with(tracker, pool_jobs(3), batch_size=1)

    assert tracker["peak"] == 1
    assert len(tracker["finished"]) == 3


def test_run_pool_keeps_running_after_a_job_raises():
    tracker = make_tracker()

    async def flaky(row, daytona, output_root, model_name):
        if row["instance_id"] == "owner__repo-1":
            raise RuntimeError("sandbox leaked")
        return await recording_runner(row, daytona, output_root, model_name, tracker)

    outcomes = run_pool_with(tracker, pool_jobs(3), batch_size=1, runner=flaky)

    assert outcomes == [True, False, True]


def test_run_pool_returns_one_outcome_per_job_in_order():
    tracker = make_tracker()

    assert run_pool_with(tracker, pool_jobs(4), batch_size=2) == [True] * 4


def test_build_jobs_pairs_every_row_with_every_model_model_major():
    rows = [task_row(), {**task_row(), "instance_id": "other__repo-2"}]

    assert [(row["instance_id"], model) for row, model in benchmark.build_jobs(rows)] == [
        ("owner__repo-1", "qwen3.8-flash"),
        ("other__repo-2", "qwen3.8-flash"),
        ("owner__repo-1", "deepseek-v4-flash-0731"),
        ("other__repo-2", "deepseek-v4-flash-0731"),
    ]


# --- wall-clock cap: remote kill, censored flag, artifacts still exported ----


def test_run_timeout_is_forty_five_minutes():
    assert benchmark.RUN_TIMEOUT_SECONDS == 45 * 60


def test_agent_command_is_wrapped_in_a_remote_timeout():
    command = benchmark.agent_command(2700)

    assert command.startswith("mkdir -p /tmp/minisweagent && timeout -k 30 2700 bash -lc '")
    assert "/opt/minisweagent/bin/mini" in command


def test_run_task_records_the_cap_in_status(tmp_path: Path):
    asyncio.run(benchmark.run_task(task_row(), FakeDaytona(), tmp_path))

    status = read_status(tmp_path / "qwen3.8-flash" / "owner__repo-1")
    assert status["timeout_s"] == benchmark.RUN_TIMEOUT_SECONDS
    assert status["censored"] is False


def test_run_task_marks_censored_when_the_agent_hits_the_cap(tmp_path: Path):
    daytona = FakeDaytona(process_failures={AGENT_PREFIX: 124})

    assert asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path)) is False

    status = read_status(tmp_path / "qwen3.8-flash" / "owner__repo-1")
    assert status["censored"] is True
    assert status["outcome"] == "failure"
    assert "censored" in status["error"]


def test_run_task_still_exports_artifacts_after_a_censored_agent(tmp_path: Path):
    daytona = FakeDaytona(process_failures={AGENT_PREFIX: 124})

    asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path))

    run_dir = tmp_path / "qwen3.8-flash" / "owner__repo-1"
    assert (run_dir / "patch.diff").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert daytona.deleted is True


def test_run_task_accepts_a_custom_cap(tmp_path: Path):
    daytona = FakeDaytona()

    asyncio.run(benchmark.run_task(task_row(), daytona, tmp_path, timeout=600))

    agent = next(c for c, *_ in daytona.sandbox.process.commands if c.startswith(AGENT_PREFIX))
    assert "timeout -k 30 600" in agent


def test_run_pool_returns_false_for_a_hung_job_and_completes_the_rest(tmp_path: Path):
    tracker = make_tracker()

    async def hung(row, daytona, output_root, model_name):
        if row["instance_id"] == "owner__repo-1":
            await asyncio.sleep(5)
        return await recording_runner(row, daytona, output_root, model_name, tracker)

    outcomes = run_pool_with(
        tracker, pool_jobs(3), batch_size=1, runner=hung, timeout=0.05
    )

    assert outcomes == [True, False, True]


# --- matrix wiring: env-driven pool over every task x model -----------------


@pytest.fixture
def two_row_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "two.csv"
    write_rows(path, [task_row(), {**task_row(), "instance_id": "other__repo-2"}])
    return path


FAKE_CLIENT = lambda config: FakeDaytona()


def matrix_env(dataset: Path, **extra: str) -> dict[str, str]:
    env = {
        "DAYTONA_API_KEY": "key",
        "QWEN_TOKEN_PLAN_API_KEY": "token",
        "DATASET_PATH": str(dataset),
        "BATCH_SIZE": "2",
        **extra,
    }
    return env


def capture_jobs(monkeypatch, pool_kwargs: dict | None = None):
    captured: dict = {"jobs": [], **(pool_kwargs or {})}

    async def fake_pool(jobs, daytona, output_root, **kwargs):
        captured["jobs"] = list(jobs)
        captured["kwargs"] = kwargs
        captured["output_root"] = output_root
        return [True] * len(jobs)

    monkeypatch.setattr(benchmark, "run_pool", fake_pool)
    return captured


def test_run_benchmark_runs_every_task_model_pair_model_major(
    tmp_path, monkeypatch, two_row_dataset
):
    captured = capture_jobs(monkeypatch)

    assert asyncio.run(
        benchmark.run_benchmark(matrix_env(two_row_dataset), client_factory=FAKE_CLIENT, output_root=tmp_path)
    ) is True

    assert [(row["instance_id"], model) for row, model in captured["jobs"]] == [
        ("owner__repo-1", "qwen3.8-flash"),
        ("other__repo-2", "qwen3.8-flash"),
        ("owner__repo-1", "deepseek-v4-flash-0731"),
        ("other__repo-2", "deepseek-v4-flash-0731"),
    ]


def test_run_benchmark_passes_batch_size_to_the_pool(tmp_path, monkeypatch, two_row_dataset):
    captured = capture_jobs(monkeypatch)

    asyncio.run(
        benchmark.run_benchmark(matrix_env(two_row_dataset, BATCH_SIZE="3"), client_factory=FAKE_CLIENT, output_root=tmp_path)
    )

    assert captured["kwargs"]["batch_size"] == 3


def test_run_benchmark_defaults_to_two_slots(tmp_path, monkeypatch, two_row_dataset):
    captured = capture_jobs(monkeypatch)
    env = matrix_env(two_row_dataset)
    del env["BATCH_SIZE"]

    asyncio.run(benchmark.run_benchmark(env, client_factory=FAKE_CLIENT, output_root=tmp_path))

    assert captured["kwargs"]["batch_size"] == 2


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
def test_run_benchmark_rejects_a_bad_batch_size(tmp_path, monkeypatch, two_row_dataset, value):
    called = capture_jobs(monkeypatch)

    with pytest.raises(ValueError, match="BATCH_SIZE"):
        asyncio.run(
            benchmark.run_benchmark(
                matrix_env(two_row_dataset, BATCH_SIZE=value),
                client_factory=FAKE_CLIENT,
                output_root=tmp_path,
            )
        )
    assert called["jobs"] == []


def test_run_benchmark_returns_false_when_any_run_fails(tmp_path, monkeypatch, two_row_dataset):
    async def fake_pool(jobs, daytona, output_root, **kwargs):
        return [True, False, True, True]

    monkeypatch.setattr(benchmark, "run_pool", fake_pool)

    assert asyncio.run(
        benchmark.run_benchmark(matrix_env(two_row_dataset), client_factory=FAKE_CLIENT, output_root=tmp_path)
    ) is False


def test_run_benchmark_provisions_the_secret_once_for_the_whole_matrix(
    tmp_path, monkeypatch, two_row_dataset
):
    capture_jobs(monkeypatch)
    secrets = FakeSecretService()
    daytona = FakeDaytona(secret_service=secrets)

    asyncio.run(
        benchmark.run_benchmark(
            matrix_env(two_row_dataset),
            client_factory=lambda config: daytona,
            output_root=tmp_path,
        )
    )

    assert len(secrets.list_calls) == 1
    assert len(secrets.created) == 1
    assert daytona.closed is True


def test_run_benchmark_rejects_duplicate_instance_ids_before_creating_a_client(
    tmp_path, monkeypatch
):
    path = tmp_path / "dupes.csv"
    write_rows(path, [task_row(), task_row()])
    created = []

    def factory(config):
        created.append(config)
        return FakeDaytona()

    with pytest.raises(ValueError, match="duplicate instance_id"):
        asyncio.run(
            benchmark.run_benchmark(matrix_env(path), client_factory=factory, output_root=tmp_path)
        )

    assert created == []
    assert (tmp_path / "validation-error" / "status.json").exists()


def test_run_benchmark_prints_a_matrix_summary(tmp_path, monkeypatch, two_row_dataset, capsys):
    async def fake_pool(jobs, daytona, output_root, **kwargs):
        return [True, False, True, False]

    monkeypatch.setattr(benchmark, "run_pool", fake_pool)

    asyncio.run(benchmark.run_benchmark(matrix_env(two_row_dataset), client_factory=FAKE_CLIENT, output_root=tmp_path))

    out = capsys.readouterr().out
    assert "2/4" in out


def test_run_pool_default_runner_writes_model_scoped_artifacts(tmp_path: Path):
    """The unwired path matters: the default runner must pass the job's model through."""
    jobs = [(task_row(), benchmark.DEEPSEEK_MODEL_NAME)]

    outcomes = asyncio.run(benchmark.run_pool(jobs, FakeDaytona(), tmp_path, batch_size=2))

    assert outcomes == [True]
    status = read_status(tmp_path / benchmark.DEEPSEEK_MODEL_NAME / "owner__repo-1")
    assert status["outcome"] == "success"
    assert status["model_name"] == benchmark.DEEPSEEK_MODEL_NAME


def test_pool_footprint_fits_the_daytona_org_limit():
    """The org caps total concurrent vCPU and memory; exceeding it fails every create."""
    assert benchmark.DEFAULT_BATCH_SIZE * benchmark.CPU_ALLOCATION <= benchmark.ORG_CPU_LIMIT
    assert (
        benchmark.DEFAULT_BATCH_SIZE * benchmark.MEMORY_ALLOCATION_GIB
        <= benchmark.ORG_MEMORY_LIMIT_GIB
    )
