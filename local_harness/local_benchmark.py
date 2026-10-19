"""Experiments 2 and 3: an agent (opencode or mini-swe-agent) in a local Docker harness, one fresh
local container per bash call, so every tool call gets its own system metrics.

Same tasks, models, phases and artifacts as experiment 1 (shared/runner.py), all on local Docker.
Per task:
- a named volume holds /testbed, seeded by Docker from the task image on first mount;
- a harness container (harness.Dockerfile, both agents) mounts it and runs the agent;
- every command goes through tool_client.py (/opt/hook/bin/bash) to the tool server in this
  process: opencode via its `shell` setting (opencode_tools/opencode.json, bash its only tool),
  mini-swe-agent via miniswe_tools/mini_env.py. The server runs it with `docker run --rm` from
  the task image on the same volume. Shared files mean no sync: edits, build artifacts and ignored files all persist.

Usage (from the repo root): uv run python -m local_harness.local_benchmark opencode|miniswe
Results: opencode_tools/runs/ or miniswe_tools/runs/, <model>/<instance_id>/.

Extra artifacts: toolcalls.jsonl (one line per bash call: wall time, CPU, memory peak, exit code)
and toolmetrics.jsonl (metrics.sh 1s samples of every tool container, tagged with `call`).
"""
import asyncio
import base64
import ctypes
import itertools
import json
import os
import shutil
import sys
import time
from asyncio.subprocess import PIPE, STDOUT
from pathlib import Path

import shared.runner as shared

HARNESS_IMAGE = "agent-metrics-harness"
OPENCODE_VERSION = "1.19.2"
HARNESS_CPU, HARNESS_MEMORY_GIB = 1, 2
TOOL_CPU, TOOL_MEMORY_GIB = 1, 1
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # docker build context: the harness copies files from every folder
PROMPT = (
    "Fix the issue below in the repository at /testbed. Edit the source code; do not modify "
    "or add tests. Stop once the fix is in place.\n\n<issue>\n{}\n</issue>"
)
# Runs in each tool container, command on stdin. bashrc activates the task's conda env, as in the
# mini-swe-agent baseline. The summary covers the container's whole life, i.e. just this call.
TOOL_WRAPPER = f"""
CMD=$(cat)
OUT=/metrics/$CALL.jsonl INTERVAL=1 ALLOC_CPU={TOOL_CPU} ALLOC_MEM_GIB={TOOL_MEMORY_GIB} \\
  nohup bash /opt/metrics.sh >/dev/null 2>&1 &
source /root/.bashrc >/dev/null 2>&1
start=$EPOCHREALTIME
bash -c "$CMD" 2>&1
rc=$?
end=$EPOCHREALTIME
{{ read -r peak < /sys/fs/cgroup/memory.peak; }} 2>/dev/null || peak=null
usage=$(awk '/^usage_usec/{{print $2}}' /sys/fs/cgroup/cpu.stat)
printf '{{"exec_start":%s,"exec_end":%s,"cpu_usec":%s,"memory_peak":%s}}\\n' \\
  "$start" "$end" "${{usage:-null}}" "$peak" > /metrics/$CALL.end.json
exit $rc
"""


async def docker(*args: str, stdin: bytes | None = None) -> tuple[int, bytes]:
    """Run the docker CLI; stdout and stderr merged, like a sandbox exec result."""
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdin=PIPE if stdin is not None else None, stdout=PIPE, stderr=STDOUT
    )
    output, _ = await process.communicate(stdin)
    return process.returncode, output


async def start_tool_server(run) -> asyncio.Server:
    """One line of base64 command in, one JSON {output, exit_code} out (see tool_client.py).

    Bound to loopback: Docker Desktop forwards host.docker.internal to it.
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            command = base64.b64decode(await reader.readline()).decode()
            call = asyncio.ensure_future(run(command))
            # The client only waits for the reply, so EOF means it was killed: the agent's own
            # command timeout fired. Cancelling stops the container too, like a local kill would.
            gone = asyncio.ensure_future(reader.read())
            await asyncio.wait({call, gone}, return_when=asyncio.FIRST_COMPLETED)
            if not call.done():
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
                return
            gone.cancel()
            try:
                output, code = call.result()
            except Exception as error:
                output, code = f"[tool container] {type(error).__name__}: {error}\n", 1
            writer.write(json.dumps({"output": output, "exit_code": code}).encode())
            await writer.drain()
        finally:
            writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


async def run_tool(row, tag: str, tools_dir: Path, call: int, command: str) -> tuple[str, int]:
    """One bash call in one fresh container from the task image, on the shared /testbed volume."""
    started = time.monotonic()
    name, code, output = f"tool-{tag}-{call}", None, b""
    try:
        code, output = await docker(
            "run", "--rm", "-i", "--name", name,
            f"--cpus={TOOL_CPU}", f"--memory={TOOL_MEMORY_GIB}g", f"--memory-swap={TOOL_MEMORY_GIB}g",
            "--mount", f"type=volume,src=testbed-{tag},dst=/testbed",
            "--mount", f"type=bind,src={shared.METRICS_SCRIPT_PATH},dst=/opt/metrics.sh,readonly",
            "--mount", f"type=bind,src={tools_dir},dst=/metrics",
            "-w", "/testbed", "-e", f"CALL={call}",
            row["image"], "bash", "-c", TOOL_WRAPPER,
            stdin=command.encode(),
        )
    except asyncio.CancelledError:
        # The client is gone (start_tool_server): kill the container, still log the call.
        await docker("rm", "-f", name)
        output = b"[tool container] killed: the agent's command timeout fired"
        raise
    finally:
        log_tool_call(tools_dir, call, command, code, output, time.monotonic() - started)
    return output.decode(errors="replace"), code


def log_tool_call(tools_dir: Path, call: int, command: str, code, output: bytes, wall_s: float):
    log = {
        "call": call, "ts": time.time(), "command": command, "exit_code": code,
        # docker run end to end: container create, start, command, stop and removal.
        "wall_s": round(wall_s, 3),
    }
    summary = tools_dir / f"{call}.end.json"
    if summary.exists():
        log.update(json.loads(summary.read_text()))
        summary.unlink()
    else:
        # No summary: the wrapper never finished (docker refused the run, or the container was
        # killed). Keep what docker said; the agent saw it too.
        log["docker_output"] = output.decode(errors="replace")[-2000:]
    with open(tools_dir.parent / "toolcalls.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(log) + "\n")


def merge_tool_metrics(tools_dir: Path, run_dir: Path) -> int:
    """tools/<call>.jsonl -> toolmetrics.jsonl tagged with `call`; returns the sample count."""
    lines = []
    for path in sorted(tools_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue  # the container died mid-write: the last line may be torn
            lines.append(json.dumps({"call": int(path.stem), **sample}))
        path.unlink()
    (run_dir / "toolmetrics.jsonl").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return len(lines)


def mini_config(row: dict[str, str], model: str) -> str:
    """Experiment 1's mini-swe-agent config, commands sent to tool containers (mini_env.py)."""
    config = json.loads(shared.make_agent_config(row, model))
    config["environment"]["environment_class"] = "mini_env.ToolContainerEnvironment"
    return json.dumps(config, ensure_ascii=False)  # JSON is YAML


# Per agent: its experiment's runs folder, files written into the harness (via stdin: some problem
# statements exceed Windows' 32k command-line limit), env names copied from ours, the command,
# files exported after the run.
AGENTS = {
    "opencode": {
        "runs": ROOT / "opencode_tools" / "runs",
        "version": f"opencode {OPENCODE_VERSION}",
        "files": lambda row, model: {"/tmp/prompt.md": PROMPT.format(row["problem_statement"])},
        "env": ["OPENCODE_DISABLE_AUTOUPDATE", "QWEN_TOKEN_PLAN_API_KEY"],
        # qwen-token-plan: provider in opencode.json, same token plan as experiment 1.
        # --format json: the event stream doubles as the trajectory.
        "command": lambda model: "/root/.opencode/bin/opencode run --format json "
        f'--model qwen-token-plan/{model} --dir /testbed "$(cat /tmp/prompt.md)"',
        "export": {},
    },
    "miniswe": {
        "runs": ROOT / "miniswe_tools" / "runs",
        "version": f"mini-swe-agent {shared.MINI_VERSION}",
        "files": lambda row, model: {shared.OVERRIDE_PATH: mini_config(row, model)},
        "env": ["OPENAI_API_KEY", "MSWEA_CONFIGURED", "MSWEA_COST_TRACKING"],
        "command": lambda model: "env PYTHONPATH=/opt/hook /opt/minisweagent/bin/mini -c swebench.yaml "
        f"-c {shared.OVERRIDE_PATH} -y --exit-immediately -o {shared.TRAJECTORY_PATH}",
        "export": {shared.TRAJECTORY_PATH: "trajectory.traj.json"},
    },
}


async def run_task(
    row: dict[str, str], agent: str, model: str,
    timeout: int = shared.RUN_TIMEOUT_SECONDS,
) -> bool:
    spec = AGENTS[agent]
    instance_id = row["instance_id"]
    run_dir = spec["runs"] / model / instance_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("toolcalls.jsonl", "toolmetrics.jsonl"):
        (run_dir / stale).unlink(missing_ok=True)  # reruns overwrite, like experiment 1
    tools_dir = (run_dir / "tools").resolve()  # bind-mount sources must be absolute
    tools_dir.mkdir(exist_ok=True)
    tag = f"{agent}-{model}-{instance_id}"  # both models may run the same task at once
    name, volume = f"harness-{tag}", f"testbed-{tag}"
    calls = itertools.count(1)
    status: dict = {
        "instance_id": instance_id,
        "model_name": model,
        "harness": f"{spec['version']} (docker)",
        "container": name,
        "phase": "create",
        "outcome": "failure",
        "error": None,
        "exit_codes": {},
        "phases": {},
        "artifacts": dict.fromkeys(
            ("install_log", "agent_log", "patch_diff", "metrics", "toolcalls", "toolmetrics",
             *(local.split(".")[0] for local in spec["export"].values())),
            False,
        ),
        "metrics": None,
        "tool_calls": 0,
        "cleanup": None,
        "timeout_s": timeout,
        "censored": False,
        "started_at": shared._utc_now(),
    }
    server = None
    try:
        with shared._phase(status, run_dir, "install"):
            # Pull and seed before the harness starts, so its metrics cover only harness work.
            code, pulled = await docker("pull", row["image"])
            if code == 0:
                await remove_task_resources(tag)  # leftovers from a killed run
                code, seeded = await docker(
                    "run", "--rm", "--mount", f"type=volume,src={volume},dst=/testbed",
                    row["image"], "true",
                )
                pulled += seeded
            (run_dir / "install.log").write_bytes(pulled)
            status["exit_codes"]["install"] = code
            status["artifacts"]["install_log"] = True
            if code != 0:
                raise RuntimeError(f"image pull or volume seed exited with code {code}")

        with shared._phase(status, run_dir, "create"):
            code, output = await docker(
                "run", "-d", "--name", name,
                f"--cpus={HARNESS_CPU}",
                f"--memory={HARNESS_MEMORY_GIB}g", f"--memory-swap={HARNESS_MEMORY_GIB}g",
                "--mount", f"type=volume,src={volume},dst=/testbed",
                HARNESS_IMAGE, "sleep", "infinity",
            )
            if code != 0:
                raise RuntimeError(f"docker run failed: {output.decode(errors='replace')}")
            code, output = await docker(
                "exec", "-d",
                "-e", f"OUT={shared.METRICS_REMOTE_OUT}",
                "-e", f"INTERVAL={shared.SAMPLE_INTERVAL_SECONDS}",
                "-e", f"ALLOC_CPU={HARNESS_CPU}", "-e", f"ALLOC_MEM_GIB={HARNESS_MEMORY_GIB}",
                name, "bash", "/opt/hook/metrics.sh",
            )
            if code != 0:
                raise RuntimeError(f"metrics collector failed: {output.decode(errors='replace')}")
            server = await start_tool_server(
                lambda command: run_tool(row, tag, tools_dir, next(calls), command)
            )
            port = server.sockets[0].getsockname()[1]

        with shared._phase(status, run_dir, "agent"):
            for path, content in spec["files"](row, model).items():
                parent = path.rsplit("/", 1)[0]
                await docker("exec", "-i", name, "sh", "-c", f"mkdir -p {parent} && cat > {path}",
                             stdin=content.encode())
            # Capped in-container, like miniswe_daytona.benchmark.agent_command, so the export still runs at the cap.
            code, output = await docker(
                # Bare -e NAME: docker copies the value from our env, keeping it off the command line.
                "exec", *(arg for name_ in spec["env"] for arg in ("-e", name_)),
                "-e", f"TOOL_SERVER=host.docker.internal:{port}", "-w", "/testbed",
                name, "bash", "-c", f"exec timeout -k 30 {int(timeout)} {spec['command'](model)}",
            )
            (run_dir / "agent.log").write_bytes(output)
            status["exit_codes"]["agent"] = code
            status["artifacts"]["agent_log"] = True
            if code in shared.CENSORED_EXIT_CODES:
                status["censored"] = True
                shared._set_error(status, "censored: agent exceeded the wall-clock cap")
            elif code != 0:
                shared._set_error(status, f"agent exited with code {code}")

        with shared._phase(status, run_dir, "export"):
            code, patch = await docker(
                "exec", name, "git", "-C", "/testbed", "diff", "--binary", row["base_commit"]
            )
            (run_dir / "patch.diff").write_bytes(patch)
            status["exit_codes"]["patch"] = code
            status["artifacts"]["patch_diff"] = code == 0
            if code != 0:
                shared._set_error(status, f"patch exited with code {code}")
            for remote, local in spec["export"].items():
                code, content = await docker("exec", name, "cat", remote)
                if code == 0:
                    (run_dir / local).write_bytes(content)
                status["artifacts"][local.split(".")[0]] = code == 0
            code, metrics = await docker("exec", name, "cat", shared.METRICS_REMOTE_OUT)
            if code == 0:
                (run_dir / shared.METRICS_ARTIFACT).write_bytes(metrics)
                status["artifacts"]["metrics"] = True
                status["metrics"] = shared.metrics_summary(metrics)
            else:
                shared._set_error(status, "harness metrics missing")

        if status["error"] is None:
            status["outcome"] = "success"
    except Exception as error:
        shared._set_error(status, error)
    finally:
        failing_phase = status["phase"]
        with shared._phase(status, run_dir, "cleanup"):
            if server is not None:
                server.close()
            error = await remove_task_resources(tag)
            status["cleanup"] = "failed" if error else "deleted"
            if error:
                shared._set_error(status, error)
                status["outcome"] = "failure"
            else:
                status["phase"] = failing_phase
            # After removal: no tool container can still be writing samples.
            status["tool_calls"] = next(calls) - 1
            status["artifacts"]["toolcalls"] = (run_dir / "toolcalls.jsonl").exists()
            status["artifacts"]["toolmetrics"] = merge_tool_metrics(tools_dir, run_dir) > 0
            shutil.rmtree(tools_dir, ignore_errors=True)  # summaries of interrupted calls
        status["finished_at"] = shared._utc_now()
        shared._write_status(run_dir, status)
    return status["outcome"] == "success"


async def remove_task_resources(tag: str) -> str | None:
    """Harness, any tool container still running, then the volume. Returns an error or None."""
    # Repeated name filters are OR'ed; anchored so one task never matches another's prefix.
    code, ids = await docker(
        "ps", "-aq", "--filter", f"name=^harness-{tag}$", "--filter", f"name=^tool-{tag}-"
    )
    if ids.split():
        code, output = await docker("rm", "-f", *ids.decode().split())
        if code != 0:
            return f"docker rm failed: {output.decode(errors='replace')}"
    code, output = await docker("volume", "rm", "-f", f"testbed-{tag}")
    return None if code == 0 else f"docker volume rm failed: {output.decode(errors='replace')}"


async def run_benchmark(agent: str, env=os.environ) -> bool:
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of: {', '.join(AGENTS)}")
    if not env.get("QWEN_TOKEN_PLAN_API_KEY"):
        raise RuntimeError("QWEN_TOKEN_PLAN_API_KEY is not set (export .env first)")
    # Values for the bare `docker exec -e NAME` of AGENTS[*]["env"]. mini-swe-agent's litellm reads
    # the token plan key as OPENAI_API_KEY, like experiment 1's Daytona secret.
    os.environ.update(
        OPENAI_API_KEY=env["QWEN_TOKEN_PLAN_API_KEY"], OPENCODE_DISABLE_AUTOUPDATE="1",
        MSWEA_CONFIGURED="true", MSWEA_COST_TRACKING="ignore_errors",
    )
    if sys.platform == "win32":
        # Keep Windows out of standby: it froze a run for 75 min, then pulls failed without network.
        # DISPLAY too: on Modern Standby laptops, the screen turning off (e.g. locked) sends the
        # whole machine to standby, SYSTEM alone does not stop that.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x1 | 0x2)  # CONTINUOUS|SYSTEM|DISPLAY
    batch_size = shared._batch_size(env)
    rows = shared.load_rows(Path(env.get("DATASET_PATH") or shared.DEFAULT_DATASET_PATH))
    code, output = await docker(
        "build", "-q", "-t", HARNESS_IMAGE, "-f", str(HERE / "harness.Dockerfile"),
        "--build-arg", f"OPENCODE_VERSION={OPENCODE_VERSION}",
        "--build-arg", f"MINI_VERSION={shared.MINI_VERSION}", str(ROOT)
    )
    if code != 0:
        raise RuntimeError(f"harness image build failed:\n{output.decode(errors='replace')}")
    outcomes = await shared.run_pool(
        shared.build_jobs(rows, env.get("MODELS", "").split(",") if env.get("MODELS") else shared.MODELS),
        lambda row, model: run_task(row, agent, model),
        batch_size=batch_size,
    )
    print(f"runs: {sum(outcomes)}/{len(outcomes)} succeeded", flush=True)
    return all(outcomes)


if __name__ == "__main__":
    try:
        sys.exit(0 if asyncio.run(run_benchmark(sys.argv[1] if len(sys.argv) > 1 else "")) else 1)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
