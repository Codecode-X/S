#!/usr/bin/env bash


set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"


_vp_restore_xtrace=0
case "$-" in *x*) _vp_restore_xtrace=1; set +x ;; esac
if [[ -z "${DASHSCOPE_API_KEY:-}" && -r "$ROOT/.secrets/dashscope_api_key" ]]; then
    IFS= read -r DASHSCOPE_API_KEY < "$ROOT/.secrets/dashscope_api_key" || true
    export DASHSCOPE_API_KEY
fi
if [[ "$_vp_restore_xtrace" == 1 ]]; then set -x; fi
unset _vp_restore_xtrace

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/home/usr/UU/m-x/Qwen2.5-VL-7B-Instruct}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/dataset}"
export QWEN_VISUAL_IMAGE_ROOT="${QWEN_VISUAL_IMAGE_ROOT:-$ROOT/output/qwen_visual_initial_maps_vqvae}"
export VQA_DATA_ROOT="${VQA_DATA_ROOT:-/home/usr/UU/data/VQAv2_val/downloads}"
export VQA_MAX_SAMPLES="${VQA_MAX_SAMPLES:-1000}"
export VQA_SAMPLE_SEED="${VQA_SAMPLE_SEED:-2026}"
export VQA_SAMPLE_STRATEGY="${VQA_SAMPLE_STRATEGY:-random}"
export VQA_MAX_NEW_TOKENS="${VQA_MAX_NEW_TOKENS:-64}"
export VQA_MAX_IMAGE_SIZE="${VQA_MAX_IMAGE_SIZE:-256}"
export VQA_PROMPT="${VQA_PROMPT:-Answer directly.}"
export VQA_JUDGE_MODEL="${VQA_JUDGE_MODEL:-qwen3.6-flash}"
export VQA_JUDGE_CONCURRENCY="${VQA_JUDGE_CONCURRENCY:-16}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export STOP_AFTER_EPOCHS="${STOP_AFTER_EPOCHS:-5}"
export SEED="${SEED:-2026}"

export REUSE_MAIN_MODELS="${REUSE_MAIN_MODELS:-0}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-0}"
export IMAGE_SCALES="${IMAGE_SCALES:-0.7,0.8,0.9,1.0,1.1,1.2,1.3}"

: "${ABLATIONS:?Use a scripts/<category>/<experiment>.sh entry point}"
export ABLATIONS
exec "${PYTHON:-python3}" "$ROOT/tools/experiments/gru_ablation_runner.py" "$@"
