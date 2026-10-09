#!/usr/bin/env bash
set -euo pipefail

log_stage() {
  printf '\n[%s] [IMAGE-PREP] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log_stage "1/4 Parse image cache settings"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

TASKS="${TASKS:-frozenlake,maze,minibehaviour}"
OUT_ROOT="${QWEN_VISUAL_IMAGE_ROOT:-output/qwen_visual_initial_maps_vqvae}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FILES=()
log_stage "2/4 Collect and validate dataset files"
IFS=',' read -r -a TASK_LIST <<< "${TASKS}"
for task in "${TASK_LIST[@]}"; do
  [[ "${task}" == "frozenlake" || "${task}" == "maze" || "${task}" == "minibehaviour" || "${task}" == "maze_flip" ]] || { echo "Unsupported task: ${task}" >&2; exit 1; }
  for path in "${TRAIN_DATASET:-${DATA_ROOT:-dataset}/${task}/tokenized_dataset/SFT_random/train_dataset.jsonl}" "${EVAL_DATASET:-${DATA_ROOT:-dataset}/${task}/tokenized_dataset/SFT/train_dataset.jsonl}" "${TEST_DATASET:-${DATA_ROOT:-dataset}/${task}/tokenized_dataset/SFT/test_dataset.jsonl}"; do
    [[ ! -f "${path}" ]] || FILES+=("${path}")
  done
done
[[ "${#FILES[@]}" -gt 0 ]] || { echo "No datasets found for ${TASKS}" >&2; exit 1; }

COMMON=(--jsonl "${FILES[@]}" --output_root "${OUT_ROOT}" --batch_size "${BATCH_SIZE}")
[[ -z "${VQVAE_DIR:-}" ]] || COMMON+=(--vqvae_dir "${VQVAE_DIR}")
[[ "${OVERWRITE:-0}" != "1" ]] || COMMON+=(--overwrite)

if [[ "${NUM_GPUS}" -le 1 ]]; then
  log_stage "3/4 Generate the image cache in one process"
  python3 src/ssvr/data/prepare_images.py "${COMMON[@]}" --device "${DEVICE:-cuda:0}"
  log_stage "4/4 Image cache preparation completed:${OUT_ROOT}"
  exit 0
fi

log_stage "3/4 Launch ${NUM_GPUS} image decoding shards"
PIDS=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  CUDA_VISIBLE_DEVICES="${gpu}" python3 src/ssvr/data/prepare_images.py "${COMMON[@]}" --device cuda:0 --num_shards "${NUM_GPUS}" --shard_index "${gpu}" &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAILED=1
done
if [[ "${FAILED}" == "0" ]]; then
  log_stage "4/4 All image cache shards completed:${OUT_ROOT}"
else
  log_stage "4/4 Image cache preparation failed"
fi
exit "${FAILED}"
