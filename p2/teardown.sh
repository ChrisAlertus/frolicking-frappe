#!/usr/bin/env bash
# Stop p2's chat server and delete its venv, pid and log files. Only kills
# the pid recorded in p2/.demo/, never another server.
set -euo pipefail
cd "$(dirname "$0")"
source ../scripts/demo-lib.sh

stop_server .demo/llama-server.pid
rm -rf .demo
echo "p2 demo removed: server stopped, venv + logs deleted"
