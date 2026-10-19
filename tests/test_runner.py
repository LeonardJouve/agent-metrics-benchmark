"""shared/runner.py: dataset loading, mini-swe-agent config, metrics summary, task x model pool."""
import asyncio
import csv
import json
from pathlib import Path

import pytest

from shared import runner


def task_row() -> dict[str, str]:
    return {
        "instance_id": "owner__repo-1",
        "image": "swebench/task:latest",
        "base_commit": "abc123",
        "problem_statement": "Fix the issue",
    }


MODEL = runner.MODELS[0]


def test_load_rows_requires_data(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("instance_id,image,base_commit,problem_statement\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset has no rows"):
        runner.load_rows(path)


def test_load_rows_reads_oversized_optional_cell(tmp_path: Path):
    """A >128 KiB cell must not break loading; required fields stay intact."""
    huge = "p" * (200 * 1024)
    path = tmp_path / "big.csv"
    fields = [*runner.REQUIRED_FIELDS, "test_patch"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**task_row(), "test_patch": huge})

    row = runner.load_rows(path)[0]

    assert len(row["test_patch"]) > 128 * 1024
    assert {key: row[key] for key in runner.REQUIRED_FIELDS} == task_row()


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
        writer = csv.DictWriter(file, fieldnames=[*runner.REQUIRED_FIELDS, "test_patch"])
        writer.writeheader()
        writer.writerow({**task_row(), "test_patch": "q" * (200 * 1024)})

    assert len(runner.load_rows(path)[0]["test_patch"]) > 128 * 1024
    assert csv.field_size_limit() == before


def test_load_rows_restores_field_size_limit_after_validation_failure(
    tmp_path: Path, stdlib_csv_limit: int
):
    before = csv.field_size_limit()
    path = tmp_path / "big_bad.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(runner.REQUIRED_FIELDS))
        writer.writeheader()
        writer.writerow({**task_row(), "image": ""})

    with pytest.raises(ValueError, match="image"):
        runner.load_rows(path)
    assert csv.field_size_limit() == before


def test_load_rows_restores_field_size_limit_after_open_failure(
    tmp_path: Path, stdlib_csv_limit: int
):
    before = csv.field_size_limit()
    with pytest.raises(OSError):
        runner.load_rows(tmp_path / "missing.csv")
    assert csv.field_size_limit() == before


def test_load_rows_requires_fields(tmp_path: Path):
    path = tmp_path / "missing.csv"
    path.write_text("instance_id,image,base_commit\nabc,t,123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="problem_statement"):
        runner.load_rows(path)


@pytest.mark.parametrize("field", ["instance_id", "image", "base_commit", "problem_statement"])
def test_make_agent_config_requires_fields(field: str):
    row = task_row()
    row[field] = ""
    with pytest.raises(ValueError, match=field):
        runner.make_agent_config(row, MODEL)


def test_make_agent_config_is_local_noninteractive_and_secret_free():
    config = json.loads(runner.make_agent_config(task_row(), MODEL))
    assert config["run"]["task"] == "Fix the issue"
    assert config["agent"]["step_limit"] == 250
    assert config["agent"]["cost_limit"] == 0
    assert config["environment"] == {"environment_class": "local", "cwd": "/testbed", "timeout": 60}
    assert config["model"]["model_name"] == "qwen3.8-flash"
    assert config["model"]["model_kwargs"]["custom_llm_provider"] == "openai"
    assert "OPENAI_API_KEY" not in runner.make_agent_config(task_row(), MODEL).decode()


# Two complete samples plus a torn trailing line: the collector is still appending
# when the file is downloaded, so a partial last line must be skipped, not crash.
METRICS_JSONL = (
    b'{"ts":"2026-09-25T16:50:40Z","epoch":1000.0,"usage_usec":10}\n'
    b'{"ts":"2026-09-25T16:50:41Z","epoch":1001.0,"usage_usec":20}\n'
    b'{"ts":"2026-09-25T16:50:4'
)


def test_make_agent_config_pins_exact_model_and_agent_settings():
    assert json.loads(runner.make_agent_config(task_row(), MODEL)) == {
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


def write_dataset(tmp_path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    path = tmp_path / "tasks.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(task_row()))
        writer.writeheader()
        for row in rows if rows is not None else [task_row()]:
            writer.writerow(row)
    return path


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
        runner.load_rows(path)
    assert isinstance(caught.value, runner.DatasetValidationError)
    assert caught.value.instance_id == "owner__repo-1"


def test_load_rows_empty_dataset_error_carries_no_instance_id(tmp_path):
    path = write_dataset(tmp_path, rows=[])
    with pytest.raises(ValueError, match="no rows") as caught:
        runner.load_rows(path)
    assert not isinstance(caught.value, runner.DatasetValidationError)
    assert getattr(caught.value, "instance_id", None) is None


def test_metrics_summary_uses_median_interval():
    payload = b'{"ts":"a","epoch":10.0}\n{"ts":"b","epoch":10.5}\n{"ts":"c","epoch":12.0}\n'

    assert runner.metrics_summary(payload) == {
        "samples": 3,
        "first_ts": "a",
        "last_ts": "c",
        "interval_s": 1.0,
    }


def test_metrics_summary_skips_torn_last_line():
    assert runner.metrics_summary(METRICS_JSONL)["samples"] == 2


def test_metrics_summary_handles_empty_unparsable_and_single_sample():
    empty = {"samples": 0, "first_ts": None, "last_ts": None, "interval_s": None}
    assert runner.metrics_summary(b"") == empty
    assert runner.metrics_summary(b"not json\n") == empty
    assert runner.metrics_summary(b'{"ts":"a","epoch":1.0}\n') == {
        "samples": 1,
        "first_ts": "a",
        "last_ts": "a",
        "interval_s": None,
    }


def write_rows(path: Path, rows: list[dict[str, str]], extra_field: str | None = None):
    fields = [*runner.REQUIRED_FIELDS] + ([extra_field] if extra_field else [])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_load_rows_returns_every_row_in_order(tmp_path: Path):
    path = tmp_path / "tasks.csv"
    rows = [task_row(), {**task_row(), "instance_id": "other__repo-2"}]
    write_rows(path, rows)

    assert runner.load_rows(path) == rows


def test_load_rows_rejects_duplicate_instance_id(tmp_path: Path):
    path = tmp_path / "dupes.csv"
    write_rows(path, [task_row(), task_row()])

    with pytest.raises(ValueError, match="duplicate instance_id"):
        runner.load_rows(path)


def test_load_rows_reports_the_offending_row_id(tmp_path: Path):
    path = tmp_path / "bad.csv"
    write_rows(path, [task_row(), {**task_row(), "instance_id": "bad__repo-3", "image": ""}])

    with pytest.raises(ValueError, match="image") as excinfo:
        runner.load_rows(path)
    assert excinfo.value.instance_id == "bad__repo-3"


def test_load_rows_restores_field_size_limit(tmp_path: Path, stdlib_csv_limit: int):
    before = csv.field_size_limit()
    path = tmp_path / "big.csv"
    write_rows(path, [task_row()], extra_field="test_patch")
    path.write_text(
        path.read_text(encoding="utf-8") + "x,y,z,w," + "q" * (200 * 1024) + "\n",
        encoding="utf-8",
    )

    assert len(runner.load_rows(path)[1]["test_patch"]) > 128 * 1024
    assert csv.field_size_limit() == before


def test_models_are_pinned_and_ordered():
    assert runner.MODELS == ["qwen3.8-flash", "deepseek-v4-flash-0731"]


def test_make_agent_config_uses_the_requested_model():
    config = json.loads(
        runner.make_agent_config(task_row(), model_name="deepseek-v4-flash-0731")
    )
    assert config["model"]["model_name"] == "deepseek-v4-flash-0731"


def make_tracker():
    return {"live": 0, "peak": 0, "started": [], "finished": []}


async def recording_runner(row, model_name, tracker):
    tracker["live"] += 1
    tracker["peak"] = max(tracker["peak"], tracker["live"])
    tracker["started"].append((row["instance_id"], model_name))
    await asyncio.sleep(0.01)
    tracker["live"] -= 1
    tracker["finished"].append((row["instance_id"], model_name))
    return True


def pool_jobs(count: int = 5) -> list[tuple[dict[str, str], str]]:
    return [
        ({**task_row(), "instance_id": f"owner__repo-{index}"}, runner.MODELS[0])
        for index in range(count)
    ]


def run_pool_with(tracker, jobs, batch_size: int, job_runner=None, **kwargs):
    async def main():
        return await runner.run_pool(
            jobs,
            job_runner or (lambda row, model_name: recording_runner(row, model_name, tracker)),
            batch_size=batch_size,
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

    async def flaky(row, model_name):
        if row["instance_id"] == "owner__repo-1":
            raise RuntimeError("sandbox leaked")
        return await recording_runner(row, model_name, tracker)

    outcomes = run_pool_with(tracker, pool_jobs(3), batch_size=1, job_runner=flaky)

    assert outcomes == [True, False, True]


def test_run_pool_returns_one_outcome_per_job_in_order():
    tracker = make_tracker()

    assert run_pool_with(tracker, pool_jobs(4), batch_size=2) == [True] * 4


def test_build_jobs_pairs_every_row_with_every_model_model_major():
    rows = [task_row(), {**task_row(), "instance_id": "other__repo-2"}]

    assert [(row["instance_id"], model) for row, model in runner.build_jobs(rows)] == [
        ("owner__repo-1", "qwen3.8-flash"),
        ("other__repo-2", "qwen3.8-flash"),
        ("owner__repo-1", "deepseek-v4-flash-0731"),
        ("other__repo-2", "deepseek-v4-flash-0731"),
    ]


def test_run_timeout_is_forty_five_minutes():
    assert runner.RUN_TIMEOUT_SECONDS == 45 * 60


def test_run_pool_returns_false_for_a_hung_job_and_completes_the_rest(tmp_path: Path):
    tracker = make_tracker()

    async def hung(row, model_name):
        if row["instance_id"] == "owner__repo-1":
            await asyncio.sleep(5)
        return await recording_runner(row, model_name, tracker)

    outcomes = run_pool_with(
        tracker, pool_jobs(3), batch_size=1, job_runner=hung, timeout=0.05
    )

    assert outcomes == [True, False, True]
