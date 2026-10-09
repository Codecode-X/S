#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export TASKS=frozenlake
exec bash "$ROOT/scripts/common/main_experiment.sh" "$@"
