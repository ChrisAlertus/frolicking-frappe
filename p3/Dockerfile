# p3 KV-cache / prefix-cache demos on Chainguard's Python image.
#
#   docker build -t llm-demos-p3 ./p3
#   docker run --rm llm-demos-p3 kv_math.py
#
# prefix_cache.py reaches the laptop's single-slot llama-server via
# host.docker.internal:
#
#   docker run --rm -e BASE_URL=http://host.docker.internal:8393 \
#     llm-demos-p3 prefix_cache.py
FROM cgr.dev/chainguard/python:latest-dev AS build
WORKDIR /app
COPY requirements.txt .
RUN python -m venv venv && venv/bin/pip install --no-cache-dir -r requirements.txt

FROM cgr.dev/chainguard/python:latest
WORKDIR /app
COPY --from=build /app/venv venv
COPY . .
ENTRYPOINT ["/app/venv/bin/python"]
