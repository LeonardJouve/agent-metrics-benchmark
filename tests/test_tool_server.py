import asyncio
import base64
import json

from local_harness import local_benchmark
from local_harness import tool_client


def test_client_relays_tool_output_and_exit_code(monkeypatch, capsys):
    calls = []

    async def run(command):
        calls.append(command)
        if command == "boom":
            raise RuntimeError("docker run failed")
        return "it's ünïcode\n", 3

    async def scenario():
        server = await local_benchmark.start_tool_server(run)
        monkeypatch.setenv("TOOL_SERVER", f"127.0.0.1:{server.sockets[0].getsockname()[1]}")
        command = """echo "it's" | grep -c "'" """
        b64 = lambda text: base64.b64encode(text.encode()).decode()
        try:
            # The client blocks, so it runs in a thread while the server serves on this loop.
            assert await asyncio.to_thread(tool_client.main, b64(command)) == 3
            assert capsys.readouterr().out == "it's ünïcode\n"
            assert await asyncio.to_thread(tool_client.main, b64("boom")) == 1
            assert "RuntimeError: docker run failed" in capsys.readouterr().out
        finally:
            server.close()
            await server.wait_closed()
        # Server gone: fail loudly, never run the command in the harness.
        assert await asyncio.to_thread(tool_client.main, b64("echo")) == 1
        assert "tool server unreachable" in capsys.readouterr().out

    asyncio.run(scenario())
    assert calls == ["""echo "it's" | grep -c "'" """, "boom"]


def test_merge_tool_metrics_tags_calls_and_skips_torn_lines(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "10.jsonl").write_text('{"epoch": 3}\n')
    (tools / "2.jsonl").write_text('{"epoch": 1}\n{"epoch": 2}\n{"epo')
    assert local_benchmark.merge_tool_metrics(tools, tmp_path) == 3
    lines = [json.loads(l) for l in (tmp_path / "toolmetrics.jsonl").read_text().splitlines()]
    assert lines == [{"call": 2, "epoch": 1}, {"call": 2, "epoch": 2}, {"call": 10, "epoch": 3}]
    assert not list(tools.iterdir())


def test_server_cancels_the_call_when_the_client_disappears():
    # The agent's command timeout kills tool_client: the container must not run on.
    cancelled = asyncio.Event()

    async def run(command):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def scenario():
        server = await local_benchmark.start_tool_server(run)
        port = server.sockets[0].getsockname()[1]
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(base64.b64encode(b"sleep 60") + b"\n")
            await writer.drain()
            await asyncio.sleep(0.1)
            writer.close()
            await asyncio.wait_for(cancelled.wait(), 5)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
