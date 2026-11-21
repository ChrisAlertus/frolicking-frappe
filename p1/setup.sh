#!/usr/bin/env bash
# p1 needs no server, just the Go modules. There is no teardown: `go mod
# download` only fills Go's own module cache.
set -euo pipefail
cd "$(dirname "$0")"
command -v go >/dev/null || {
    echo "missing go (brew install go)" >&2
    exit 1
}
go mod download
