#!/usr/bin/env bash
# p3: download modules, then start the single-slot chat server the
# prefix-cache demo uses. The embedding server belongs to p4
# (./p4/setup.sh). Override with PREFIX_PORT, PREFIX_MODEL, CONTEXT.
set -euo pipefail
cd "$(dirname "$0")"
source ../scripts/demo-lib.sh

PREFIX_PORT="${PREFIX_PORT:-8393}"
PREFIX_MODEL="${PREFIX_MODEL:-Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M}"
CONTEXT="${CONTEXT:-16384}"
DEMO_DIR=".demo"

require_tools go curl llama-server
go mod download
mkdir -p "$DEMO_DIR"

start_server llama-prefix \
    "$DEMO_DIR/llama-prefix.pid" "$DEMO_DIR/llama-prefix.log" "$PREFIX_PORT" \
    -hf "$PREFIX_MODEL" -ngl 99 -c "$CONTEXT" -np 1 --cache-ram 0 \
    --port "$PREFIX_PORT"
wait_healthy llama-prefix \
    "$DEMO_DIR/llama-prefix.pid" "$DEMO_DIR/llama-prefix.log" "$PREFIX_PORT"

echo "ready: BASE_URL=http://127.0.0.1:$PREFIX_PORT  (prefix_cache)"
