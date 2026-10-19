# Harness for local_benchmark.py, both agents: opencode and mini-swe-agent + git. Commands run in
# sibling containers from the task image, sharing /testbed through a named volume, so no task
# environment is needed here. Versions come from local_benchmark.py (--build-arg).
# Build context: the repo root (see .dockerignore).
FROM python:3.12-slim-bookworm
ARG OPENCODE_VERSION
ARG MINI_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://opencode.ai/install | bash -s -- --version ${OPENCODE_VERSION}
RUN python3 -m venv /opt/minisweagent \
    && /opt/minisweagent/bin/pip install --no-cache-dir mini-swe-agent==${MINI_VERSION}
COPY opencode_tools/opencode.json /root/.config/opencode/opencode.json
COPY shared/metrics.sh miniswe_tools/mini_env.py /opt/hook/
COPY --chmod=755 local_harness/tool_client.py /opt/hook/bin/bash
