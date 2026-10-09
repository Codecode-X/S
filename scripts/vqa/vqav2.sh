#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export ABLATIONS=ssvr
export EXPERIMENT_KIND=vqa
export RUN_ANALYSIS=1
export ABLATION_SCOPE=planning
export ABLATION_SCOPE=full
exec bash "$ROOT/scripts/common/ablation_experiment.sh" "$@"
