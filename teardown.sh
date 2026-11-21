#!/usr/bin/env bash
# Tear down every demo folder by running each p*/teardown.sh, then delete
# the shared model cache. Only pids recorded in the folders' .demo/ dirs
# are stopped; a server you launched yourself is left alone.
set -euo pipefail
cd "$(dirname "$0")"

for teardown in p*/teardown.sh; do
    [[ -f "$teardown" ]] || continue # no teardown needed, or literal glob
    echo "== $teardown"
    "./$teardown"
done
rm -rf .demo
echo "demo removed: servers stopped, shared model cache deleted"
