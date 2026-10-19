"""Experiment 3: mini-swe-agent environment for local_harness/local_benchmark.py
(environment_class: mini_env.ToolContainerEnvironment; copied to /opt/hook in the harness).

mini-swe-agent's local environment, except every command runs through /opt/hook/bin/bash
(tool_client.py), i.e. in a fresh tool container, like opencode's bash. Timeout, output and
submission handling stay mini-swe-agent's own.
"""
import shlex

from minisweagent.environments.local import LocalEnvironment


class ToolContainerEnvironment(LocalEnvironment):
    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = f"exec /opt/hook/bin/bash -c {shlex.quote(action.get('command', ''))}"
        return super().execute({**action, "command": command}, cwd, timeout=timeout)
