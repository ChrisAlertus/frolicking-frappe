#!/usr/bin/env bash
# p2: download modules, then start the chat server the server-backed demos
# use. Override with PORT, MODEL, CONTEXT env vars.
set -euo pipefail
cd "$(dirname "$0")"
source ../scripts/demo-lib.sh

PORT="${PORT:-8080}"
MODEL="${MODEL:-unsloth/Qwen3.5-4B-GGUF:Q4_K_M}"
CONTEXT="${CONTEXT:-8192}"
DEMO_DIR=".demo"

require_tools go curl llama-server
go mod download
mkdir -p "$DEMO_DIR"

start_server llama-server \
    "$DEMO_DIR/llama-server.pid" "$DEMO_DIR/llama-server.log" "$PORT" \
    -hf "$MODEL" -ngl 99 -fa on -c "$CONTEXT" --port "$PORT"
wait_healthy llama-server \
    "$DEMO_DIR/llama-server.pid" "$DEMO_DIR/llama-server.log" "$PORT"

echo "ready: BASE_URL=http://127.0.0.1:$PORT/v1"
