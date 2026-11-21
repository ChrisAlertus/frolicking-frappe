#!/usr/bin/env bash
# Stop p4's embedding server and delete its pid and log files. Only kills
# the pid recorded in p4/.demo/, never another server.
set -euo pipefail
cd "$(dirname "$0")"
source ../scripts/demo-lib.sh

stop_server .demo/llama-embed.pid
rm -rf .demo
echo "p4 demo removed: server stopped, logs deleted"
