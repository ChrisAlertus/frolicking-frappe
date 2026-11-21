#!/usr/bin/env bash
# p1 needs no servers and no venv, just the Go modules. There is no
# teardown: `go mod download` only fills Go's own module cache.
set -euo pipefail
cd "$(dirname "$0")"
go mod download
