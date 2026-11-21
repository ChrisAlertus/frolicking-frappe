#!/usr/bin/env bash
# Stop the llama-server started by setup.sh and delete its model cache and
# logs. Only kills the pid recorded in .demo/, never another server.
set -euo pipefail
cd "$(dirname "$0")"

DEMO_DIR=".demo"
PID_FILE="$DEMO_DIR/llama-server.pid"

if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
fi

rm -rf "$DEMO_DIR"
echo "demo removed: server stopped, model cache + logs deleted"
