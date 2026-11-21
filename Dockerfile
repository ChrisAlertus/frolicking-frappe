# Python demos (p2/, p3/) on Chainguard's Python image.
#
#   docker build -t llm-demos-py .
#   docker run --rm llm-demos-py p3/kv_math.py
#
# The server-backed demos call llama-server on the laptop; from inside the
# container the laptop is host.docker.internal:
#
#   docker run --rm -e BASE_URL=http://host.docker.internal:8393 \
#     llm-demos-py p3/prefix_cache.py
FROM cgr.dev/chainguard/python:latest-dev AS build
WORKDIR /app
COPY requirements.txt .
RUN python -m venv venv && venv/bin/pip install --no-cache-dir -r requirements.txt

FROM cgr.dev/chainguard/python:latest
WORKDIR /app
COPY --from=build /app/venv venv
COPY p2 p2
COPY p3 p3
ENTRYPOINT ["/app/venv/bin/python"]
