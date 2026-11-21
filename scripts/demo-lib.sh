# Shared helpers for the per-folder demo setup/teardown scripts. Not meant
# to be run directly; source it from a pN/ script:
#
#   source "$(dirname "$0")/../scripts/demo-lib.sh"
#
# shellcheck shell=bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# All servers share one model cache, so each model is downloaded once no
# matter which folder's setup started the download.
export LLAMA_CACHE="${LLAMA_CACHE:-$REPO_ROOT/.demo/models}"

require_tools() {
    local tool
    for tool in "$@"; do
        command -v "$tool" >/dev/null || {
            echo "missing $tool (brew install go llama.cpp)" >&2
            exit 1
        }
    done
}

# make_venv <venv-dir> <requirements-file>
make_venv() {
    local venv="$1" reqs="$2"
    command -v python3 >/dev/null || return 0
    if [[ ! -x "$venv/bin/python" ]]; then
        python3 -m venv "$venv"
        "$venv/bin/pip" -q install -r "$reqs"
        echo "python venv ready: $venv/bin/python"
    fi
}

# start_server <name> <pid-file> <log-file> <port> [llama-server args...]
start_server() {
    local name="$1" pid_file="$2" log_file="$3" port="$4"
    shift 4
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "$name already running (pid $(cat "$pid_file"))"
    elif curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        echo "port $port already has a healthy server; not starting another"
    else
        if command -v python3 >/dev/null; then
            # New session, so the server outlives the shell that launched it.
            # exec keeps the pid, so the pid file still points at the server.
            python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
                llama-server "$@" >"$log_file" 2>&1 &
        else
            nohup llama-server "$@" >"$log_file" 2>&1 &
        fi
        echo $! >"$pid_file"
        echo "$name starting (pid $(cat "$pid_file")), logs: $log_file"
    fi
}

# wait_healthy <name> <pid-file> <log-file> <port>
wait_healthy() {
    local name="$1" pid_file="$2" log_file="$3" port="$4"
    echo "waiting for $name on http://127.0.0.1:$port/health ..."
    for _ in $(seq 1 600); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "ready: $name on http://127.0.0.1:$port"
            return 0
        fi
        if [[ -f "$pid_file" ]] && ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo "$name exited early; see $log_file" >&2
            return 1
        fi
        sleep 1
    done
    echo "timed out waiting for $name; see $log_file" >&2
    return 1
}

# stop_server <pid-file>: stop the pid recorded in the file, if still alive.
stop_server() {
    local pid_file="$1" pid
    [[ -f "$pid_file" ]] || return 0
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}
