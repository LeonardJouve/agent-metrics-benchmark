#!/usr/local/bin/python3 -IS
"""The agents' shell in the harness container, installed as /opt/hook/bin/bash: opencode's `shell`
(opencode_tools/opencode.json), mini-swe-agent's via miniswe_tools/mini_env.py. Both run
`<shell> -c <command>`, so the command reaches us untouched and the model's history shows exactly
what it typed. Named bash so opencode's tool prompt is unchanged.

Sends the base64 command to the runner's tool server (TOOL_SERVER=host:port), which runs it in a
fresh container from the task image. Prints the output and exits with the command's exit code.
Stdlib only and run with `python3 -I -S`, so it starts in milliseconds.
"""
import base64
import json
import os
import shlex
import socket
import sys


def main(b64_command: str) -> int:
    host, port = os.environ["TOOL_SERVER"].rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port))) as conn:
            conn.sendall(b64_command.encode() + b"\n")
            reply = json.loads(conn.makefile("rb").read())
    except (OSError, ValueError) as error:
        # Fail loudly: the command must never run in the harness instead.
        print(f"[tool container] tool server unreachable: {error}")
        return 1
    sys.stdout.write(reply["output"])
    return reply["exit_code"]


if __name__ == "__main__":
    if sys.argv[1:2] != ["-c"]:
        os.execv("/bin/bash", ["bash", *sys.argv[1:]])  # anything but `-c <command>`: real bash
    # The tool container starts in /testbed: carry over the cwd the agent chose.
    command = f"cd {shlex.quote(os.getcwd())} && {sys.argv[2]}"
    sys.exit(main(base64.b64encode(command.encode()).decode()))
