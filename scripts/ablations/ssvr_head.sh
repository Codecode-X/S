#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export ABLATIONS=ssvr_head
export EXPERIMENT_KIND=ablation
export RUN_ANALYSIS=0
exec bash "$ROOT/scripts/common/ablation_experiment.sh" "$@"
