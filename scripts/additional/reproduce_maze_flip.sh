#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/additional/reproduce_maze_flip.sh
Full reproduction: build or reuse data -> cache initial images -> train -> evaluate the final checkpoint -> summarize.
Environment variables follow maze_flip.sh; OVERWRITE=1 rebuilds the data.
RUN_DIR must be a new directory for each run.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }
export DATA_ROOT="${DATA_ROOT:-$ROOT/dataset}"
export MAZE_FLIP_DATA_ROOT="${MAZE_FLIP_DATA_ROOT:-$DATA_ROOT/maze_flip/tokenized_dataset}"
args=(--train_source "${MAZE_TRAIN_SOURCE:-$DATA_ROOT/maze/tokenized_dataset/SFT_random/train_dataset.jsonl}"
      --validation_source "${MAZE_VALIDATION_SOURCE:-$DATA_ROOT/maze/tokenized_dataset/SFT/test_dataset.jsonl}"
      --output_root "$MAZE_FLIP_DATA_ROOT")
if [[ "${OVERWRITE:-0}" == 1 ]]; then
  args+=(--overwrite)
fi
python tools/data/build_maze_flip_dataset.py "${args[@]}"
exec bash "$ROOT/scripts/additional/maze_flip.sh" all
