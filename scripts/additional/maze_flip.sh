#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/additional/maze_flip.sh [prepare|all|train|eval]
  prepare   Build or reuse MazeFlip data from Maze
  all       Prepare images, train, evaluate and summarize using existing MazeFlip data (default)
  train / eval follow the main experiment entry point

Data defaults to DATA_ROOT/maze_flip/tokenized_dataset.
MAZE_TRAIN_SOURCE / MAZE_VALIDATION_SOURCE override the original Maze data paths.
MAZE_FLIP_DATA_ROOT overrides the output directory; OVERWRITE=1 rebuilds the data.
Model, GPU and learning-rate settings follow scripts/common/main_experiment.sh.
EOF
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
export DATA_ROOT="${DATA_ROOT:-$ROOT/dataset}"
export MAZE_FLIP_DATA_ROOT="${MAZE_FLIP_DATA_ROOT:-$DATA_ROOT/maze_flip/tokenized_dataset}"
case "${1:-all}" in
  prepare)
    shift
    [[ $# -eq 0 ]] || { usage >&2; exit 2; }
    args=(--train_source "${MAZE_TRAIN_SOURCE:-$DATA_ROOT/maze/tokenized_dataset/SFT_random/train_dataset.jsonl}"
          --validation_source "${MAZE_VALIDATION_SOURCE:-$DATA_ROOT/maze/tokenized_dataset/SFT/test_dataset.jsonl}"
          --output_root "$MAZE_FLIP_DATA_ROOT")
    [[ "${OVERWRITE:-0}" != 1 ]] || args+=(--overwrite)
    exec python tools/data/build_maze_flip_dataset.py "${args[@]}"
    ;;
  all|train|eval) ;;
  *) usage >&2; exit 2 ;;
esac
export TASKS=maze_flip
export TRAIN_DATASET="${TRAIN_DATASET:-$MAZE_FLIP_DATA_ROOT/SFT_random/train_dataset.jsonl}"
export TEST_DATASET="${TEST_DATASET:-$MAZE_FLIP_DATA_ROOT/SFT/test_dataset.jsonl}"
export EVAL_DATASET="${EVAL_DATASET:-$TEST_DATASET}"
exec bash "$ROOT/scripts/common/main_experiment.sh" "$@"
