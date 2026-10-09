#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export ABLATIONS=ssvr
export EXPERIMENT_KIND=invert
export RUN_ANALYSIS=1
export ABLATION_SCOPE=planning
exec bash "$ROOT/scripts/common/ablation_experiment.sh" "$@"
