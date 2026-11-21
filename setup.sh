#!/usr/bin/env bash
# Set up every demo folder by running each p*/setup.sh. Each folder manages
# its own venv, servers, and pid/log files under pN/.demo/; the shared
# helpers live in scripts/demo-lib.sh and all servers share the model cache
# in .demo/models. Run one folder only with e.g. ./p3/setup.sh.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p .demo/models
for setup in p*/setup.sh; do
    echo "== $setup"
    "./$setup"
done
