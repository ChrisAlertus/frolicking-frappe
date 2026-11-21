#!/usr/bin/env bash
# Prepare the local demo: download Go modules, then start llama-server in the
# background with its model cache confined to .demo/ so teardown.sh can
# remove everything it created. Override with PORT, MODEL, CONTEXT env vars.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"
MODEL="${MODEL:-unsloth/Qwen3.5-4B-GGUF:Q4_K_M}"
CONTEXT="${CONTEXT:-8192}"
DEMO_DIR=".demo"
PID_FILE="$DEMO_DIR/llama-server.pid"
LOG_FILE="$DEMO_DIR/llama-server.log"
export LLAMA_CACHE="${LLAMA_CACHE:-$PWD/$DEMO_DIR/models}"

for tool in go curl llama-server; do
    command -v "$tool" >/dev/null || {
        echo "missing $tool (brew install go llama.cpp)" >&2
        exit 1
    }
done

(cd p1 && go mod download)
(cd p2 && go mod download)

mkdir -p "$DEMO_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "llama-server already running (pid $(cat "$PID_FILE"))"
elif curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "port $PORT already has a healthy server; not starting another"
else
    server_args=(-hf "$MODEL" -ngl 99 -fa on -c "$CONTEXT" --port "$PORT")
    if command -v python3 >/dev/null; then
        # New session, so the server outlives the shell that launched it.
        # exec keeps the pid, so the pid file still points at the server.
        python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
            llama-server "${server_args[@]}" >"$LOG_FILE" 2>&1 &
    else
        nohup llama-server "${server_args[@]}" >"$LOG_FILE" 2>&1 &
    fi
    echo $! >"$PID_FILE"
    echo "llama-server starting (pid $(cat "$PID_FILE")), logs: $LOG_FILE"
fi

echo "waiting for http://127.0.0.1:$PORT/health ..."
for _ in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "ready: BASE_URL=http://127.0.0.1:$PORT/v1"
        exit 0
    fi
    if [[ -f "$PID_FILE" ]] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "llama-server exited early; see $LOG_FILE" >&2
        exit 1
    fi
    sleep 1
done
echo "timed out waiting for llama-server; see $LOG_FILE" >&2
exit 1
