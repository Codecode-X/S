#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
CHECKPOINT="${2:-}"
if [[ "${TASK}" != "frozenlake" && "${TASK}" != "maze" && "${TASK}" != "minibehaviour" && "${TASK}" != "maze_flip" ]] || [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: bash scripts/common/evaluate.sh <frozenlake|maze|minibehaviour|maze_flip> <checkpoint>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the base Qwen2.5-VL model}"
QWEN_PROCESSOR_PATH="${QWEN_PROCESSOR_PATH:-${QWEN_MODEL_PATH}}"
QWEN_VISUAL_IMAGE_ROOT="${QWEN_VISUAL_IMAGE_ROOT:-output/qwen_visual_initial_maps_vqvae}"
TEST_DATASET="${TEST_DATASET:-dataset/${TASK}/tokenized_dataset/SFT/test_dataset.jsonl}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
USE_KV_CACHE="${USE_KV_CACHE:-0}"
[[ "$USE_KV_CACHE" == 0 || "$USE_KV_CACHE" == 1 ]] || { echo 'USE_KV_CACHE must be 0 or 1' >&2; exit 2; }
OUTPUT_DIR="${OUTPUT_DIR:-output/evals/${TASK}/$(basename "${CHECKPOINT}")}"

[[ -d "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${TEST_DATASET}" ]] || { echo "Missing dataset: ${TEST_DATASET}" >&2; exit 1; }
[[ -d "${QWEN_VISUAL_IMAGE_ROOT}" ]] || { echo "Missing image cache: ${QWEN_VISUAL_IMAGE_ROOT}" >&2; exit 1; }

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ssvr-eval.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
if [[ ! "${CHECKPOINT_NAME}" =~ ^checkpoint-[0-9]+$ ]]; then
  CHECKPOINT_STEP="$(python - "${CHECKPOINT}/trainer_state.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(int(json.load(handle)["global_step"]))
PY
)"
  CHECKPOINT_NAME="checkpoint-${CHECKPOINT_STEP}"
fi
ln -s "$(realpath "${CHECKPOINT}")" "${TMP_ROOT}/${CHECKPOINT_NAME}"

ARGS=(
  --ckpt_root "${TMP_ROOT}"
  --base_model "${QWEN_MODEL_PATH}"
  --backbone_type qwen25vl
  --qwen_visual_input
  --qwen_processor_path "${QWEN_PROCESSOR_PATH}"
  --qwen_visual_image_root "${QWEN_VISUAL_IMAGE_ROOT}"
  --test_dataset "${TEST_DATASET}"
  --output_dir "${OUTPUT_DIR}"
  --doc_path "${OUTPUT_DIR}/metrics.md"
  --torch_dtype bfloat16
  --task "${TASK}"
)
[[ -z "${EVAL_MAX_SAMPLES:-1000}" ]] || ARGS+=(--max_samples "${EVAL_MAX_SAMPLES:-1000}")
[[ "${QWEN_PROCESSOR_USE_FAST:-false}" != "true" ]] || ARGS+=(--qwen_processor_use_fast)
if [[ "$USE_KV_CACHE" == 1 ]]; then
  ARGS+=(--use_kv_cache)
else
  ARGS+=(--no-use_kv_cache)
fi

accelerate launch --num_processes "${NUM_GPUS}" --main_process_port "${MASTER_PORT}" src/ssvr/evaluation/run.py "${ARGS[@]}"
