import csv
import io
import json
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

import build_dataset
from build_dataset import (
    EXPECTED_SOURCE_ROWS,
    MAX_IMAGE_BYTES,
    PAGE_SIZE,
    docker_hub_tag_url,
    fetch_source_rows,
    image_full_size,
    select_rows,
    write_samples,
)

ROWS_BASE = "https://datasets-server.huggingface.co/rows?"


def query_params(url: str) -> dict[str, str]:
    """Decoded single-valued query parameters of a rows endpoint URL."""
    parsed = urlparse(url)
    assert url.startswith(ROWS_BASE), url
    assert (parsed.scheme, parsed.netloc, parsed.path) == (
        "https",
        "datasets-server.huggingface.co",
        "/rows",
    )
    return {key: values[0] for key, values in parse_qs(parsed.query).items()}


def row(repo: str, number: int, difficulty: str = "<15 min fix") -> dict[str, object]:
    return {
        "repo": repo,
        "instance_id": f"{repo.replace('/', '__')}-{number}",
        "difficulty": difficulty,
        "image": f"swebench/{repo.replace('/', '_')}-{number}:latest",
    }


def test_docker_hub_tag_url():
    assert docker_hub_tag_url("swebench/task:latest") == (
        "https://hub.docker.com/v2/repositories/swebench/task/tags/latest/"
    )


def test_docker_hub_tag_url_strips_docker_io_prefix():
    assert docker_hub_tag_url("docker.io/swebench/task:latest") == docker_hub_tag_url(
        "swebench/task:latest"
    )


def test_select_rows_filters_and_balances_repositories():
    rows = [row(repo, number) for repo in ("a/a", "b/b", "c/c") for number in range(6)]
    rows += [row("a/a", 99, "1-4 hours"), row("b/b", 98)]
    sizes = {str(item["image"]): 1_000_000_000 for item in rows}
    sizes[str(rows[-1]["image"])] = 2_000_000_001
    sizes[str(rows[0]["image"])] = None

    selected = select_rows(rows, sizes.__getitem__, count=9, seed=42)

    assert len(selected) == 9
    assert set(Counter(str(item["repo"]) for item in selected).values()) == {3}
    assert all(item["difficulty"] in {"<15 min fix", "15 min - 1 hour"} for item in selected)
    assert all(sizes[str(item["image"])] <= 2_000_000_000 for item in selected)
    assert selected == select_rows(rows, sizes.__getitem__, count=9, seed=42)


def test_select_rows_requires_exact_count():
    with pytest.raises(ValueError, match="need 4 eligible rows; found 3"):
        select_rows([row("a/a", i) for i in range(3)], lambda _: 1, count=4)


def test_select_rows_keeps_exact_size_ceiling_boundary():
    rows = [row("a/a", number) for number in range(3)]
    sizes = {
        "swebench/a_a-0:latest": MAX_IMAGE_BYTES,
        "swebench/a_a-1:latest": MAX_IMAGE_BYTES + 1,
        "swebench/a_a-2:latest": MAX_IMAGE_BYTES - 1,
    }

    selected = select_rows(rows, sizes.__getitem__, count=2, seed=7)

    assert {str(item["instance_id"]) for item in selected} == {"a__a-0", "a__a-2"}
    with pytest.raises(ValueError, match="need 3 eligible rows; found 2"):
        select_rows(rows, sizes.__getitem__, count=3, seed=7)


def test_select_rows_golden_order():
    # Characterization fixture: pins the exact emitted order for a fixed input,
    # seed and count, so neither the per-group shuffle nor the final shuffle can
    # be dropped without failing. Coupled to CPython's random.Random stream.
    rows = [row(repo, number) for repo in ("a/a", "b/b", "c/c") for number in (0, 1)]
    sizes = {str(item["image"]): 10 for item in rows}

    selected = select_rows(rows, sizes.__getitem__, count=6, seed=42)

    assert [str(item["instance_id"]) for item in selected] == [
        "a__a-0",
        "a__a-1",
        "c__c-0",
        "b__b-0",
        "c__c-1",
        "b__b-1",
    ]
    # Same pool, different seed must reorder (guards a no-op shuffle).
    other = select_rows(rows, sizes.__getitem__, count=6, seed=7)
    assert [str(item["instance_id"]) for item in other] != [
        str(item["instance_id"]) for item in selected
    ]


def test_select_rows_does_not_probe_size_for_excluded_difficulty():
    rows = [row("a/a", number) for number in range(4)]
    rows[0]["difficulty"] = "1-4 hours"
    rows[1]["difficulty"] = ">4 hours"
    probed: list[str] = []

    def image_size(image: str) -> int:
        probed.append(image)
        return 10

    selected = select_rows(rows, image_size, count=2, seed=7)

    assert probed == ["swebench/a_a-2:latest", "swebench/a_a-3:latest"]
    assert {str(item["instance_id"]) for item in selected} == {"a__a-2", "a__a-3"}


def test_select_rows_filters_difficulty_size_and_missing_tag():
    rows = [row("a/a", number) for number in range(4)]
    rows[1]["difficulty"] = "1-4 hours"
    sizes = {str(item["image"]): 10 for item in rows}
    sizes["swebench/a_a-2:latest"] = 2_000_000_001
    sizes["swebench/a_a-3:latest"] = None

    selected = select_rows(rows, sizes.__getitem__, count=1, seed=7)
    assert [str(item["instance_id"]) for item in selected] == ["a__a-0"]

    with pytest.raises(ValueError, match="need 2 eligible rows; found 1"):
        select_rows(rows, sizes.__getitem__, count=2, seed=7)


SOURCE_FIELDS = [
    "base_commit", "created_at", "difficulty", "environment_setup_commit",
    "eval_type", "image", "instance_id", "log_parser", "repo", "version",
    "patch", "test_patch", "eval_script", "problem_statement", "hints_text",
    "FAIL_TO_PASS", "PASS_TO_PASS",
]


def source_payload(offset: int = 0, count: int = 1, fields=SOURCE_FIELDS) -> dict:
    rows = [
        {
            **dict.fromkeys(fields, f"value-{offset}"),
            "difficulty": "<15 min fix",
            "repo": f"repo/{offset}",
            "instance_id": f"instance-{offset}",
        }
        for _ in range(count)
    ]
    return {
        "features": [{"name": name} for name in fields],
        "rows": [{"row": item} for item in rows],
    }


def test_fetch_source_rows_pages_and_preserves_schema():
    calls = []

    def fake_json(url: str):
        calls.append(url)
        offset = 0 if "offset=0" in url else 100
        rows = [{"row": dict.fromkeys(SOURCE_FIELDS, f"value-{offset}")}]
        return {"features": [{"name": name} for name in SOURCE_FIELDS], "rows": rows}

    fields, rows = fetch_source_rows(fake_json, expected_rows=2, page_size=1)

    assert fields == SOURCE_FIELDS
    assert len(rows) == 2
    assert [query_params(url)["offset"] for url in calls] == ["0", "1"]
    for url, offset in zip(calls, ("0", "1")):
        params = query_params(url)
        assert params["dataset"] == "SWE-bench/SWE-bench_Verified"
        assert params["config"] == "default"
        assert params["split"] == "test"
        assert params["length"] == "1"
        assert params["offset"] == offset


def test_fetch_source_rows_uses_production_split_pagination_defaults():
    assert (EXPECTED_SOURCE_ROWS, PAGE_SIZE) == (500, 100)
    calls: list[str] = []

    def fake_json(url: str):
        calls.append(url)
        params = query_params(url)
        offset = int(params["offset"])
        rows = [
            {
                "row": {
                    **dict.fromkeys(SOURCE_FIELDS, "value"),
                    "difficulty": "<15 min fix",
                    "repo": "a/a",
                    "instance_id": f"instance-{index}",
                }
            }
            for index in range(offset, offset + int(params["length"]))
        ]
        return {"features": [{"name": name} for name in SOURCE_FIELDS], "rows": rows}

    fields, rows = fetch_source_rows(fake_json)

    assert fields == SOURCE_FIELDS
    assert len(rows) == EXPECTED_SOURCE_ROWS
    assert [query_params(url)["offset"] for url in calls] == ["0", "100", "200", "300", "400"]
    assert {query_params(url)["length"] for url in calls} == {"100"}
    assert {query_params(url)["dataset"] for url in calls} == {"SWE-bench/SWE-bench_Verified"}
    assert {query_params(url)["config"] for url in calls} == {"default"}
    assert {query_params(url)["split"] for url in calls} == {"test"}


def test_fetch_source_rows_clamps_final_page_and_rejects_short_response():
    calls: list[str] = []

    def fake_json(url: str):
        calls.append(url)
        return source_payload(count=1)

    with pytest.raises(ValueError, match="expected 3 source rows; found 2"):
        fetch_source_rows(fake_json, expected_rows=3, page_size=2)

    assert [query_params(url)["offset"] for url in calls] == ["0", "2"]
    assert [query_params(url)["length"] for url in calls] == ["2", "1"]


def test_fetch_source_rows_rejects_duplicate_instance_ids():
    def fake_json(_):
        return source_payload(count=2)

    with pytest.raises(ValueError, match="duplicate instance_id"):
        fetch_source_rows(fake_json, expected_rows=2, page_size=2)


def test_fetch_source_rows_rejects_missing_required_field():
    def fake_json(_):
        return source_payload(fields=[name for name in SOURCE_FIELDS if name != "image"])

    with pytest.raises(ValueError, match="missing required fields"):
        fetch_source_rows(fake_json, expected_rows=1, page_size=1)


def test_image_full_size_reads_docker_hub_value():
    urls: list[str] = []

    def fake_json(url: str):
        urls.append(url)
        return {"full_size": 123}

    assert image_full_size("swebench/task:latest", fake_json) == 123
    assert urls == [docker_hub_tag_url("swebench/task:latest")]


def test_image_full_size_returns_none_for_missing_tag():
    def missing(_):
        raise HTTPError("url", 404, "missing", {}, None)

    assert image_full_size("swebench/missing:latest", missing) is None


def test_image_full_size_propagates_other_http_errors():
    def server_error(_):
        raise HTTPError("url", 500, "boom", {}, None)

    with pytest.raises(HTTPError):
        image_full_size("swebench/task:latest", server_error)


def test_write_samples_preserves_fields(tmp_path: Path):
    rows = []
    for number in range(5):
        item = dict.fromkeys(SOURCE_FIELDS, "")
        item.update(instance_id=str(number), FAIL_TO_PASS=["test"], PASS_TO_PASS={"key": 1})
        rows.append(item)

    sample20 = tmp_path / "sample20.csv"
    write_samples(rows, SOURCE_FIELDS, sample20)

    with sample20.open(encoding="utf-8", newline="") as file:
        sample = list(csv.DictReader(file))
    assert list(sample[0]) == SOURCE_FIELDS
    assert len(sample) == 5
    assert json.loads(sample[0]["FAIL_TO_PASS"]) == ["test"]
    assert json.loads(sample[0]["PASS_TO_PASS"]) == {"key": 1}


def test_write_samples_leaves_no_temporary_files(tmp_path: Path):
    item = dict.fromkeys(SOURCE_FIELDS, "")
    sample20 = tmp_path / "sample20.csv"
    write_samples([item], SOURCE_FIELDS, sample20)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["sample20.csv"]


def test_write_samples_keeps_existing_file_when_a_row_is_incomplete(tmp_path: Path):
    sample20 = tmp_path / "sample20.csv"
    write_samples([dict.fromkeys(SOURCE_FIELDS, "old")], SOURCE_FIELDS, sample20)
    before = sample20.read_text(encoding="utf-8")

    broken = dict.fromkeys(SOURCE_FIELDS, "new")
    del broken["repo"]
    with pytest.raises(KeyError, match="repo"):
        write_samples([broken], SOURCE_FIELDS, sample20)

    assert sample20.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob("*.tmp"))


def test_main_builds_sample20_offline(tmp_path: Path, monkeypatch, capsys):
    hub_calls: list[str] = []

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(url: str, timeout: int | None = None):
        parsed = urlparse(url)
        if parsed.netloc == "datasets-server.huggingface.co":
            query = parse_qs(parsed.query)
            offset = int(query["offset"][0])
            rows = []
            for index in range(offset, offset + int(query["length"][0])):
                repo = f"org{index % 3}/repo{index % 12}"
                rows.append({"row": {
                    **dict.fromkeys(SOURCE_FIELDS, f"text-{index}"),
                    "difficulty": "<15 min fix",
                    "repo": repo,
                    "instance_id": f"instance-{index}",
                    "image": f"swebench/img-{index // 2}:latest",
                    "FAIL_TO_PASS": ["test_one"],
                }})
            body = {"features": [{"name": name} for name in SOURCE_FIELDS], "rows": rows}
        else:
            hub_calls.append(url)
            body = {"full_size": 1_000}
        assert timeout == 60
        return FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(build_dataset, "urlopen", fake_urlopen)
    monkeypatch.setattr(build_dataset, "SAMPLE20_PATH", tmp_path / "sample20.csv")

    assert build_dataset.main() == 0

    with (tmp_path / "sample20.csv").open(encoding="utf-8", newline="") as file:
        sample = list(csv.DictReader(file))
    assert list(sample[0]) == SOURCE_FIELDS
    assert len(sample) == 20
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sample20.csv"]
    assert json.loads(sample[0]["FAIL_TO_PASS"]) == ["test_one"]
    assert len(hub_calls) == len(set(hub_calls)) == 250  # probed once per distinct image
    assert "selected 20 rows from 12 repositories" in capsys.readouterr().out


def test_write_samples_keeps_non_ascii_cells_readable(tmp_path: Path):
    item = dict.fromkeys(SOURCE_FIELDS, "")
    item.update(FAIL_TO_PASS=["修复测试"], problem_statement="naïve")
    sample20 = tmp_path / "sample20.csv"

    write_samples([item], SOURCE_FIELDS, sample20)

    with sample20.open(encoding="utf-8", newline="") as file:
        cells = next(csv.DictReader(file))
    assert json.loads(cells["FAIL_TO_PASS"]) == ["修复测试"]
    # Pins ensure_ascii=False: an escaped cell would still json-load identically.
    assert cells["FAIL_TO_PASS"] == '["修复测试"]'
    assert cells["problem_statement"] == "naïve"


HUB_URL = "https://hub.docker.com/v2/repositories/swebench/task/tags/latest/"


class FakeJSONResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def http_error(code: int, headers: dict[str, str] | None = None) -> HTTPError:
    return HTTPError(HUB_URL, code, f"status {code}", headers or {}, None)


def stub_transport(monkeypatch, outcomes: list[object]) -> list[tuple[str, int | None]]:
    calls: list[tuple[str, int | None]] = []

    def fake_urlopen(url: str, timeout: int | None = None):
        calls.append((url, timeout))
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeJSONResponse(json.dumps(outcome).encode("utf-8"))

    monkeypatch.setattr(build_dataset, "urlopen", fake_urlopen)
    return calls


def stub_sleep(monkeypatch, build_dataset) -> list[float]:
    waits: list[float] = []
    monkeypatch.setattr(build_dataset, "sleep", waits.append)
    return waits


def test_http_json_retries_429_with_exponential_backoff(monkeypatch):
    calls = stub_transport(
        monkeypatch, [http_error(429), http_error(429), {"full_size": 5}]
    )
    waits = stub_sleep(monkeypatch, build_dataset)

    assert build_dataset.http_json(HUB_URL) == {"full_size": 5}
    assert [url for url, _ in calls] == [HUB_URL] * 3
    assert [timeout for _, timeout in calls] == [60, 60, 60]
    assert waits == [1.0, 2.0]


def test_http_json_honours_numeric_retry_after_capped_at_the_ceiling(monkeypatch):
    calls = stub_transport(
        monkeypatch,
        [
            http_error(429, {"Retry-After": "30"}),
            http_error(429, {"Retry-After": "9999"}),
            http_error(429, {"Retry-After": "0"}),
            {"full_size": 5},
        ],
    )
    waits = stub_sleep(monkeypatch, build_dataset)

    assert build_dataset.http_json(HUB_URL) == {"full_size": 5}
    assert len(calls) == 4
    assert waits == [30.0, 60.0, 0.0]


def test_http_json_ignores_non_numeric_retry_after(monkeypatch):
    stub_transport(
        monkeypatch,
        [http_error(429, {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}), {"ok": True}],
    )
    waits = stub_sleep(monkeypatch, build_dataset)

    assert build_dataset.http_json(HUB_URL) == {"ok": True}
    assert waits == [1.0]


def test_http_json_raises_the_final_429_after_bounded_attempts(monkeypatch):
    calls = stub_transport(monkeypatch, [http_error(429)] * build_dataset.HTTP_ATTEMPTS)
    waits = stub_sleep(monkeypatch, build_dataset)

    with pytest.raises(HTTPError) as raised:
        build_dataset.http_json(HUB_URL)

    assert raised.value.code == 429
    assert len(calls) == build_dataset.HTTP_ATTEMPTS
    assert waits == [1.0, 2.0, 4.0]


@pytest.mark.parametrize("code", [404, 403, 500, 503])
def test_http_json_does_not_retry_other_statuses(monkeypatch, code: int):
    calls = stub_transport(monkeypatch, [http_error(code), {"full_size": 5}])
    waits = stub_sleep(monkeypatch, build_dataset)

    with pytest.raises(HTTPError) as raised:
        build_dataset.http_json(HUB_URL)

    assert raised.value.code == code
    assert len(calls) == 1
    assert waits == []


def test_http_json_pins_the_retry_budget(monkeypatch):
    assert build_dataset.HTTP_ATTEMPTS == 4
    assert build_dataset.RETRY_BASE_DELAY_SECONDS == 1.0
    assert build_dataset.RETRY_MAX_DELAY_SECONDS == 60.0
    assert build_dataset.retry_delay(http_error(429), 0) == 1.0
    assert build_dataset.retry_delay(http_error(429), 1) == 2.0
    assert build_dataset.retry_delay(http_error(429), 2) == 4.0
    assert build_dataset.retry_delay(http_error(429), 3) == 8.0
    assert build_dataset.retry_delay(http_error(429, {"Retry-After": "-5"}), 0) == 0.0
    # Default sleep must stay monkeypatchable so the suite never waits for real.
    assert build_dataset.sleep is not None


def test_image_full_size_survives_a_burst_then_reports_size(monkeypatch):
    stub_transport(monkeypatch, [http_error(429), {"full_size": 1_000}])
    waits = stub_sleep(monkeypatch, build_dataset)

    assert image_full_size("swebench/task:latest") == 1_000
    assert waits == [1.0]


def test_image_full_size_stays_loud_when_the_rate_limit_persists(monkeypatch):
    calls = stub_transport(monkeypatch, [http_error(429)] * build_dataset.HTTP_ATTEMPTS)
    stub_sleep(monkeypatch, build_dataset)

    with pytest.raises(HTTPError):
        image_full_size("swebench/task:latest")

    assert len(calls) == build_dataset.HTTP_ATTEMPTS


def test_image_full_size_reports_an_http_error_that_is_not_rate_limiting(monkeypatch):
    stub_transport(monkeypatch, [http_error(404), http_error(429)])
    waits = stub_sleep(monkeypatch, build_dataset)

    assert image_full_size("swebench/gone:latest") is None
    assert waits == []


@pytest.mark.parametrize(
    "hint",
    ["nan", "NaN", "inf", "-inf", "1e99999", "", "   ", "unbounded", "12abc", "1_0"],
)
def test_http_json_falls_back_when_retry_after_is_not_a_plain_number(
    monkeypatch, hint: str
):
    stub_transport(monkeypatch, [http_error(429, {"Retry-After": hint}), {"ok": True}])
    waits = stub_sleep(monkeypatch, build_dataset)

    assert build_dataset.http_json(HUB_URL) == {"ok": True}
    assert waits == [1.0]


def test_retry_delay_only_honours_finite_non_negative_numbers():
    def delay(hint, attempt):
        return build_dataset.retry_delay(http_error(429, {"Retry-After": hint}), attempt)

    # Non-finite is treated as malformed: the exponential schedule decides the wait.
    assert delay("nan", 2) == 4.0
    assert delay("inf", 1) == 2.0
    assert delay("-inf", 0) == 1.0
    assert delay("1e99999", 0) == 1.0
    # Finite hints are honored, clamped to the ceiling and floored at zero.
    assert delay("7", 0) == 7.0
    assert delay("0.5", 2) == 0.5
    assert delay("1e-3", 0) == pytest.approx(0.001)
    assert delay("9999", 0) == 60.0
    assert delay("-5", 3) == 0.0
    assert delay("-0", 3) == 0.0
    # Absent or non-numeric headers also fall back, and the ceiling bounds the schedule.
    assert build_dataset.retry_delay(http_error(429), 3) == 8.0
    assert build_dataset.retry_delay(http_error(429, {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}), 9) == 60.0
