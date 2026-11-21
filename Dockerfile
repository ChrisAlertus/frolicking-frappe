# syntax=docker/dockerfile:1

# latest-dev ships uv, plus a shell, and is only used to install packages.
# The runtime stage is the hardened image: no shell and no package manager.
FROM cgr.dev/chainguard/python:latest-dev AS build

USER root
WORKDIR /scripts

# tiktoken: tokenize_demo.py (downloads encoding files on first run)
# requests: stream_ttft.py
# kv_cache_math.py uses only the standard library
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN uv venv /scripts/.venv
ENV PATH="/scripts/.venv/bin:$PATH"
RUN uv pip install --no-cache tiktoken requests

FROM cgr.dev/chainguard/python:latest

WORKDIR /scripts

COPY --from=build --chown=nonroot:nonroot /scripts/.venv /scripts/.venv
COPY --chown=nonroot:nonroot tokenize_demo.py stream_ttft.py kv_cache_math.py ./

ENV PATH="/scripts/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# The base entrypoint is the absolute path /usr/bin/python, which ignores
# PATH and would miss the venv. Clear it so `python <script>` still works.
ENTRYPOINT []
CMD ["python", "tokenize_demo.py"]
