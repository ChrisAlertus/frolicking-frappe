#!/usr/bin/env bash
# p4: python venv plus the embedding server its scripts use.
# Override with EMBED_PORT, EMBED_MODEL env vars.
set -euo pipefail
cd "$(dirname "$0")"
source ../scripts/demo-lib.sh

EMBED_PORT="${EMBED_PORT:-8395}"
EMBED_MODEL="${EMBED_MODEL:-nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0}"
DEMO_DIR=".demo"

require_tools curl llama-server
mkdir -p "$DEMO_DIR"
make_venv "$DEMO_DIR/venv" requirements.txt

start_server llama-embed \
    "$DEMO_DIR/llama-embed.pid" "$DEMO_DIR/llama-embed.log" "$EMBED_PORT" \
    -hf "$EMBED_MODEL" -ngl 99 --embedding --pooling mean --port "$EMBED_PORT"
wait_healthy llama-embed \
    "$DEMO_DIR/llama-embed.pid" "$DEMO_DIR/llama-embed.log" "$EMBED_PORT"

echo "ready: EMBED_URL=http://127.0.0.1:$EMBED_PORT/v1"
