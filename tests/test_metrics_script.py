"""Runs metrics.sh against a fixture cgroup directory with a real bash."""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "metrics.sh"
# Resolve the full path: CreateProcess would otherwise prefer System32\bash.exe (WSL),
# which cannot see the /c/... paths this test passes.
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="bash required")


def posix(path: Path) -> str:
    """Windows path as git-bash sees it: C:\\a\\b -> /c/a/b."""
    text = path.as_posix()
    return f"/{text[0].lower()}{text[2:]}" if len(text) > 2 and text[1] == ":" else text

REQUIRED_KEYS = {
    "ts",
    "epoch",
    "usage_usec",
    "user_usec",
    "system_usec",
    "nr_periods",
    "nr_throttled",
    "throttled_usec",
    "cpu_max",
    "memory_current",
    "memory_max",
    "memory_peak",
    "mem_high",
    "mem_max_events",
    "oom",
    "oom_kill",
    "alloc_cpu",
    "alloc_mem_gib",
    "cgroup_root",
    "cgroup_version",
}


def write_cgroup(root: Path, *, peak: bool = True, cpu_stat: bool = True) -> Path:
    """Fixture cgroup v2 directory. LF endings: real cgroup files never carry CR."""
    root.mkdir(parents=True, exist_ok=True)

    def put(name: str, text: str) -> None:
        (root / name).write_bytes(text.encode("utf-8"))

    if cpu_stat:
        put(
            "cpu.stat",
            "usage_usec 1500000\n"
            "user_usec 1000000\n"
            "system_usec 500000\n"
            "nr_periods 100\n"
            "nr_throttled 3\n"
            "throttled_usec 40000\n",
        )
    put("cpu.max", "max 100000\n")
    put("memory.current", "1048576\n")
    put("memory.max", "max\n")
    put("memory.events", "low 0\nhigh 1\nmax 2\noom 3\noom_kill 4\noom_group_kill 0\n")
    if peak:
        put("memory.peak", "2097152\n")
    return root


def run_collector(tmp_path: Path, cg_root: Path, **env_extra: str):
    out = tmp_path / "metrics.jsonl"
    env = {
        **os.environ,
        "CG_ROOT": posix(cg_root),
        "OUT": posix(out),
        "INTERVAL": "0.05",
        "DURATION": "0.2",
        "ALLOC_CPU": "4",
        "ALLOC_MEM_GIB": "8",
        **env_extra,
    }
    completed = subprocess.run(
        [BASH, posix(SCRIPT)], env=env, capture_output=True, text=True, timeout=60
    )
    lines = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if out.exists() else []
    return completed, out, lines


def test_collector_writes_one_jsonl_sample_per_interval(tmp_path: Path):
    cg_root = write_cgroup(tmp_path / "cg")

    completed, out, lines = run_collector(tmp_path, cg_root)

    assert completed.returncode == 0, completed.stderr
    assert len(lines) >= 3
    deltas = [b["epoch"] - a["epoch"] for a, b in zip(lines, lines[1:])]
    assert all(0.03 <= delta <= 0.3 for delta in deltas)


def test_collector_sample_carries_required_fields_and_fixture_values(tmp_path: Path):
    cg_root = write_cgroup(tmp_path / "cg")

    _, _, lines = run_collector(tmp_path, cg_root)

    sample = lines[0]
    assert REQUIRED_KEYS <= set(sample)
    assert (sample["usage_usec"], sample["user_usec"], sample["system_usec"]) == (
        1500000,
        1000000,
        500000,
    )
    assert (sample["nr_periods"], sample["nr_throttled"], sample["throttled_usec"]) == (
        100,
        3,
        40000,
    )
    assert sample["cpu_max"] == "max 100000"
    assert sample["memory_current"] == 1048576
    assert sample["memory_max"] == "max"
    assert sample["memory_peak"] == 2097152
    assert (sample["mem_high"], sample["mem_max_events"]) == (1, 2)
    assert (sample["oom"], sample["oom_kill"]) == (3, 4)
    assert (sample["alloc_cpu"], sample["alloc_mem_gib"]) == (4, 8)
    assert sample["cgroup_version"] == "v2"
    assert sample["cgroup_root"] == posix(cg_root)


def test_collector_timestamp_is_utc_iso8601_and_epoch_matches(tmp_path: Path):
    cg_root = write_cgroup(tmp_path / "cg")

    _, _, lines = run_collector(tmp_path, cg_root)

    sample = lines[0]
    parsed = datetime.fromisoformat(sample["ts"].replace("Z", "+00:00"))
    assert sample["ts"].endswith("Z")
    assert parsed.utcoffset().total_seconds() == 0
    assert abs(parsed.timestamp() - sample["epoch"]) < 2.0
    assert all(line["epoch"] > line_prev["epoch"] for line_prev, line in zip(lines, lines[1:]))


def test_collector_reports_null_when_memory_peak_is_unavailable(tmp_path: Path):
    cg_root = write_cgroup(tmp_path / "cg", peak=False)

    completed, _, lines = run_collector(tmp_path, cg_root)

    assert completed.returncode == 0, completed.stderr
    assert [line["memory_peak"] for line in lines] == [None] * len(lines)


def test_collector_fails_loudly_without_cgroup_v2_files(tmp_path: Path):
    cg_root = write_cgroup(tmp_path / "cg", cpu_stat=False)

    completed, out, lines = run_collector(tmp_path, cg_root)

    assert completed.returncode == 1
    assert lines and "error" in lines[0]
    assert "cpu.stat" in lines[0]["error"]
    assert out.exists()
