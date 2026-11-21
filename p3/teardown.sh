#!/usr/bin/env bash
# Stop p3's prefix-cache server and delete its venv, pid and log files.
# Only kills the pid recorded in p3/.demo/, never another server.
set -euo pipefail
cd "$(dirname "$0")"
source ../scripts/demo-lib.sh

stop_server .demo/llama-prefix.pid
rm -rf .demo
echo "p3 demo removed: server stopped, venv + logs deleted"
