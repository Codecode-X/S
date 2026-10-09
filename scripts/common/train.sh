#!/usr/bin/env bash
set -euo pipefail

log_stage() {
  printf '\n[%s] [STAGE] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log_stage "1/7 Parse task and runtime settings"
TASK="${1:-}"
if [[ "${TASK}" != "frozenlake" && "${TASK}" != "maze" && "${TASK}" != "minibehaviour" && "${TASK}" != "maze_flip" ]]; then
  echo "Usage: bash scripts/common/train.sh <frozenlake|maze|minibehaviour|maze_flip>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

log_stage "2/7 Validate model, dataset and output settings"
: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to Qwen2.5-VL-7B-Instruct}"
QWEN_PROCESSOR_PATH="${QWEN_PROCESSOR_PATH:-${QWEN_MODEL_PATH}}"
QWEN_VISUAL_IMAGE_ROOT="${QWEN_VISUAL_IMAGE_ROOT:-output/qwen_visual_initial_maps_vqvae}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29621}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
STOP_AFTER_EPOCHS="${STOP_AFTER_EPOCHS:-5}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1.5e-4}"
SEED="${SEED:-2026}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1000}"

case "${TASK}" in
  frozenlake)
    ACTION_TEXT_LABELS="${ACTION_TEXT_LABELS:-UP,DOWN,LEFT,RIGHT}"
    ;;
  maze|maze_flip)
    ACTION_TEXT_LABELS="${ACTION_TEXT_LABELS:-UP,DOWN,LEFT,RIGHT}"
    ;;
  minibehaviour)
    ACTION_TEXT_LABELS="${ACTION_TEXT_LABELS:-UP,DOWN,LEFT,RIGHT,Pick,DROP}"
    ;;
esac

TRAIN_DATASET="${TRAIN_DATASET:-dataset/${TASK}/tokenized_dataset/SFT_random/train_dataset.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-dataset/${TASK}/tokenized_dataset/SFT/test_dataset.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/models/${TASK}/qwen25vl_ssvr_t_reproduce/text_token_output}"
RUN_NAME="${RUN_NAME:-ssvr_t_${TASK}_epoch5_reproduce}"

for path in "${TRAIN_DATASET}" "${EVAL_DATASET}"; do
  [[ -f "${path}" ]] || { echo "Missing dataset: ${path}" >&2; exit 1; }
  ! head -n 1 "${path}" | grep -q '^version https://git-lfs.github.com/spec/v1' || { echo "Dataset is a Git LFS pointer: ${path}" >&2; exit 1; }
done
printf '[CONFIG] task=%s model=%s train_data=%s eval_data=%s output=%s gpus=%s learning_rate=%s\n' \
  "${TASK}" "${QWEN_MODEL_PATH}" "${TRAIN_DATASET}" "${EVAL_DATASET}" "${OUTPUT_DIR}" "${NUM_GPUS}" "${LEARNING_RATE}"

if [[ "${SKIP_IMAGE_PREP:-0}" != "1" ]]; then
  log_stage "3/7 Prepare the Qwen visual image cache"
  TASKS="${TASK}" NUM_GPUS="${NUM_GPUS}" QWEN_VISUAL_IMAGE_ROOT="${QWEN_VISUAL_IMAGE_ROOT}" bash scripts/common/prepare_images.sh
else
  log_stage "3/7 Skip image cache preparation (SKIP_IMAGE_PREP=1)"
fi

if [[ "${WAIT_FOR_IDLE_GPUS:-0}" == "1" ]]; then
  log_stage "4/7 Wait for idle GPUs"
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required for WAIT_FOR_IDLE_GPUS=1" >&2; exit 1; }
  while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v limit="${GPU_IDLE_MEM_MB:-1024}" '$1 > limit { busy=1 } END { exit !busy }'; do
    sleep "${GPU_CHECK_INTERVAL_SECONDS:-60}"
  done
  log_stage "GPUs are idle"
else
  log_stage "4/7 Skip waiting for idle GPUs"
fi

log_stage "5/7 Build training launch arguments"
ARGS=(
  --model_path "${QWEN_MODEL_PATH}"
  --processor_path "${QWEN_PROCESSOR_PATH}"
  --image_root "${QWEN_VISUAL_IMAGE_ROOT}"
  --dataset "${TRAIN_DATASET}"
  --eval_dataset "${EVAL_DATASET}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --action_text_labels "${ACTION_TEXT_LABELS}"
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${PER_DEVICE_BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --eval_max_samples "${EVAL_MAX_SAMPLES}"
  --seed "${SEED}"
)

ARGS+=(--stop_after_epochs "${STOP_AFTER_EPOCHS}")
[[ "${QWEN_PROCESSOR_USE_FAST:-false}" != "true" ]] || ARGS+=(--processor_use_fast)

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200000}"
log_stage "6/7 Launch distributed training with Accelerate"
accelerate launch --num_processes "${NUM_GPUS}" --main_process_port "${MASTER_PORT}" --mixed_precision bf16 --dynamo_backend no src/ssvr/training/sft.py "${ARGS[@]}"
log_stage "7/7 Training completed; output directory:${OUTPUT_DIR}"
