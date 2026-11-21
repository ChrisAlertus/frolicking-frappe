# p4 vector-search demos on Chainguard's Python image.
#
#   docker build -t llm-demos-p4 ./p4
#   docker run --rm llm-demos-p4 ann_bench.py --n 20000
#
# Full-scale ann_bench.py (1M vectors ≈ 3.1 GB) is OOM-killed by Docker
# Desktop's default ~2 GB VM (exit 137); pass --n or raise the VM limit.
#
# embed_rate.py, hybrid.py and embed_words.py need the laptop's embedding
# server (./p4/setup.sh), reached via host.docker.internal:
#
#   docker run --rm -e EMBED_URL=http://host.docker.internal:8395/v1 \
#     llm-demos-p4 hybrid.py
FROM cgr.dev/chainguard/python:latest-dev AS build
WORKDIR /app
COPY requirements.txt .
RUN python -m venv venv && venv/bin/pip install --no-cache-dir -r requirements.txt

FROM cgr.dev/chainguard/python:latest
WORKDIR /app
COPY --from=build /app/venv venv
COPY . .
ENTRYPOINT ["/app/venv/bin/python"]
