# p2 support-triage demos on Chainguard's Python image.
#
#   docker build -t llm-demos-p2 ./p2
#   docker run --rm llm-demos-p2 tokens_demo.py
#
# Server-backed demos reach the laptop's llama-server via
# host.docker.internal:
#
#   docker run --rm -e BASE_URL=http://host.docker.internal:8080/v1 \
#     llm-demos-p2 stream_meter.py
FROM cgr.dev/chainguard/python:latest-dev AS build
WORKDIR /app
COPY requirements.txt .
RUN python -m venv venv && venv/bin/pip install --no-cache-dir -r requirements.txt

FROM cgr.dev/chainguard/python:latest
WORKDIR /app
COPY --from=build /app/venv venv
COPY . .
ENTRYPOINT ["/app/venv/bin/python"]
