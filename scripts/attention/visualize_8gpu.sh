#!/usr/bin/env bash
set -euo pipefail


log_stage() {
  printf '\n[%s] [IMPLICIT-ATTENTION] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/usr/miniconda3/envs/visualplanning/bin/python3.12}"
NUM_GPUS="${NUM_GPUS:-8}"
BACKBONE_TYPE="${BACKBONE_TYPE:-qwen25vl}"

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/attention/visualize_8gpu.sh <MODEL_PATH> <DATASET>" >&2
  echo "DATASET must be one of: frozenlake, maze, minibehaviour" >&2
  exit 1
fi

MODEL_PATH="$1"
DATASET="$2"

case "${DATASET}" in
  frozenlake)
    TEST_DATASET="${TEST_DATASET:-${DATA_ROOT:-dataset}/frozenlake/tokenized_dataset/SFT/test_dataset.jsonl}"
    ;;
  maze)
    TEST_DATASET="${TEST_DATASET:-${DATA_ROOT:-dataset}/maze/tokenized_dataset/SFT/test_dataset.jsonl}"
    ;;
  minibehaviour)
    TEST_DATASET="${TEST_DATASET:-${DATA_ROOT:-dataset}/minibehaviour/tokenized_dataset/SFT/test_dataset.jsonl}"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}; expected frozenlake|maze|minibehaviour." >&2
    exit 1
    ;;
esac

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_PATH directory: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/local_action_config.json" ]]; then
  echo "Missing local_action_config.json under MODEL_PATH: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TEST_DATASET}" ]]; then
  echo "Missing test dataset: ${TEST_DATASET}" >&2
  exit 1
fi

ACTION_OUTPUT_MODE="$("${PYTHON_BIN}" - "${MODEL_PATH}/local_action_config.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("action_output_mode", "action_head"))
PY
)"
EXPECTED_OUTPUT_MODE="text_token"
if [[ "${BACKBONE_TYPE}" == "lvm" ]]; then
  EXPECTED_OUTPUT_MODE="action_head"
fi
if [[ "${ACTION_OUTPUT_MODE}" != "${EXPECTED_OUTPUT_MODE}" ]]; then
  echo "Unexpected action_output_mode=${ACTION_OUTPUT_MODE}; expected ${EXPECTED_OUTPUT_MODE} for ${BACKBONE_TYPE}" >&2
  exit 1
fi

if [[ "${BACKBONE_TYPE}" == "lvm" ]]; then
  BASE_MODEL="${BASE_MODEL:-/home/usr/UU/VisualPlanning/models/LVM_ckpts}"
  PROCESSOR_PATH="${PROCESSOR_PATH:-${BASE_MODEL}}"
else
  MODEL_ROOT="${MODEL_ROOT:-/home/usr/UU/m-x}"
  if [[ "${MODEL_PATH}" == *"/3B/"* || "${MODEL_PATH}" == *"Qwen2.5-VL-3B"* ]]; then
    BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct}"
  else
    BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct}"
  fi
  PROCESSOR_PATH="${PROCESSOR_PATH:-${BASE_MODEL}}"
fi
IMAGE_ROOT="${IMAGE_ROOT:-output/qwen_visual_initial_maps_vqvae}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTRIBUTION_MODE="${ATTRIBUTION_MODE:-grad_x_attention}"
ATTN_LAYER="${ATTN_LAYER:-all}"
ATTN_HEAD="${ATTN_HEAD:-mean}"
IMAGE_SCALE="${IMAGE_SCALE:-1.0}"
MAX_SCAN_SAMPLES="${MAX_SCAN_SAMPLES:-}"
MAX_STEPS="${MAX_STEPS:-}"
CASES_PER_BUCKET="${CASES_PER_BUCKET:-3}"
MIN_CORRECT_MOVE_STEPS="${MIN_CORRECT_MOVE_STEPS:-0}"
MINIBEHAVIOUR_INTERACTION_ACTION_MASKING="${MINIBEHAVIOUR_INTERACTION_ACTION_MASKING:-false}"
MINIBEHAVIOUR_SPLIT_ATTRIBUTION="${MINIBEHAVIOUR_SPLIT_ATTRIBUTION:-false}"
RESCAN_CASES="${RESCAN_CASES:-0}"

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Missing Qwen base model: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "Missing Qwen visual image root: ${IMAGE_ROOT}" >&2
  echo "Build it first with: TASKS=${DATASET} bash scripts/common/prepare_images.sh" >&2
  exit 1
fi

SAFE_MODEL_ID="$(realpath -m "${MODEL_PATH}" | sed 's#^/##; s#[^A-Za-z0-9._-]#_#g')"
OUTPUT_DIR="${OUTPUT_DIR:-EXP_DOCS/attention_visualizations/${DATASET}_implicit_attention_${SAFE_MODEL_ID}}"
MARKDOWN_PATH="${MARKDOWN_PATH:-EXP_DOCS/${DATASET}_implicit_attention_${SAFE_MODEL_ID}.md}"
SHARD_ROOT="${OUTPUT_DIR}/selection_shards"
VISUALIZE_SCRIPT="scripts/attention/visualize_8gpu.sh"
printf -v REPRODUCE_COMMAND '%q ' "TEST_DATASET=${TEST_DATASET}" "PYTHON_BIN=${PYTHON_BIN}" "BASE_MODEL=${BASE_MODEL}" "PROCESSOR_PATH=${PROCESSOR_PATH}" "IMAGE_ROOT=${IMAGE_ROOT}" "CASES_PER_BUCKET=${CASES_PER_BUCKET}" "NUM_GPUS=${NUM_GPUS}" "OUTPUT_DIR=${OUTPUT_DIR}" "MARKDOWN_PATH=${MARKDOWN_PATH}" bash "${VISUALIZE_SCRIPT}" "${MODEL_PATH}" "${DATASET}"

common_args=(
  tools/visualization/visualize_implicit_attention.py
  --checkpoint "${MODEL_PATH}"
  --base_model "${BASE_MODEL}"
  --backbone_type "${BACKBONE_TYPE}"
  --processor_path "${PROCESSOR_PATH}"
  --test_dataset "${TEST_DATASET}"
  --image_root "${IMAGE_ROOT}"
  --image_scale "${IMAGE_SCALE}"
  --torch_dtype "${TORCH_DTYPE}"
  --attribution_mode "${ATTRIBUTION_MODE}"
  --attn_layer "${ATTN_LAYER}"
  --attn_head "${ATTN_HEAD}"
  --cases_per_bucket "${CASES_PER_BUCKET}"
  --min_correct_move_steps "${MIN_CORRECT_MOVE_STEPS}"
  --reproduce_command "${REPRODUCE_COMMAND}"
)
if [[ "${MINIBEHAVIOUR_INTERACTION_ACTION_MASKING}" == "true" || "${MINIBEHAVIOUR_INTERACTION_ACTION_MASKING}" == "1" ]]; then
  common_args+=(--minibehaviour_interaction_action_masking)
fi
if [[ "${MINIBEHAVIOUR_SPLIT_ATTRIBUTION}" == "true" || "${MINIBEHAVIOUR_SPLIT_ATTRIBUTION}" == "1" ]]; then
  common_args+=(--minibehaviour_split_attribution)
fi
if [[ -n "${MAX_SCAN_SAMPLES}" ]]; then
  common_args+=(--max_scan_samples "${MAX_SCAN_SAMPLES}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  common_args+=(--max_steps "${MAX_STEPS}")
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${MARKDOWN_PATH}")"

echo "============================================"
echo " SSVR ${BACKBONE_TYPE} Attention Visualization 8GPU"
echo "============================================"
echo " Model:       ${MODEL_PATH}"
echo " Dataset:     ${DATASET}"
echo " Base model:  ${BASE_MODEL}"
echo " Backbone:    ${BACKBONE_TYPE}"
echo " Test data:   ${TEST_DATASET}"
echo " Image root:  ${IMAGE_ROOT}"
echo " Output dir:  ${OUTPUT_DIR}"
echo " Markdown:    ${MARKDOWN_PATH}"
echo " GPUs:        ${NUM_GPUS}"
echo " Attribution: ${ATTRIBUTION_MODE}; layer=${ATTN_LAYER}; head=${ATTN_HEAD}"
echo " Cases/bucket:${CASES_PER_BUCKET}"
echo " Min correct move steps: ${MIN_CORRECT_MOVE_STEPS}"
echo " MB interaction masking: ${MINIBEHAVIOUR_INTERACTION_ACTION_MASKING}"
echo " MB split attribution: ${MINIBEHAVIOUR_SPLIT_ATTRIBUTION}"
echo "============================================"

log_stage "1/3 Scan and select correct/incorrect cases for each level"
if [[ -f "${OUTPUT_DIR}/selected_cases.json" && "${RESCAN_CASES}" != "1" ]]; then
  echo "Reusing existing selected cases: ${OUTPUT_DIR}/selected_cases.json"
  echo "Set RESCAN_CASES=1 to force a new 8-GPU case scan."
else
rm -rf "${SHARD_ROOT}"
mkdir -p "${SHARD_ROOT}"

pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  shard_dir="${SHARD_ROOT}/shard_${gpu}"
  mkdir -p "${shard_dir}"
  (
    if CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${common_args[@]}" \
      --output_dir "${shard_dir}" \
      --markdown_path "${shard_dir}/report.md" \
      --device cuda:0 \
      --num_sample_shards "${NUM_GPUS}" \
      --sample_shard_index "${gpu}" \
      --select_only; then
      touch "${shard_dir}/.success"
    else
      touch "${shard_dir}/.failed"
      exit 1
    fi
  ) >"${shard_dir}/select.log" 2>&1 &
  pids+=("$!")
done

echo "Scanning cases across ${NUM_GPUS} GPU shards..."
while true; do
  completed="$(find "${SHARD_ROOT}" -mindepth 2 -maxdepth 2 \
    \( -name '.success' -o -name '.failed' \) -type f | wc -l)"
  failed_count="$(find "${SHARD_ROOT}" -mindepth 2 -maxdepth 2 \
    -name '.failed' -type f | wc -l)"
  bar_width=32
  filled="$((completed * bar_width / NUM_GPUS))"
  empty="$((bar_width - filled))"
  filled_bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  empty_bar="$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  percent="$((completed * 100 / NUM_GPUS))"
  printf '\rSelection [%s%s] %3d%% (%d/%d shards, failed=%d)' \
    "${filled_bar}" "${empty_bar}" "${percent}" "${completed}" "${NUM_GPUS}" "${failed_count}"
  if [[ "${completed}" -ge "${NUM_GPUS}" ]]; then
    printf '\n'
    break
  fi
  sleep 2
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one selection shard failed. Logs are under ${SHARD_ROOT}/shard_*/select.log" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${SHARD_ROOT}" "${OUTPUT_DIR}/selected_cases.json" "${OUTPUT_DIR}/scan_selected_cases.csv" "${CASES_PER_BUCKET}" <<'PY'
import csv
import json
import sys
from pathlib import Path

shard_root = Path(sys.argv[1])
selected_out = Path(sys.argv[2])
scan_out = Path(sys.argv[3])
cases_per_bucket = max(1, int(sys.argv[4]))

selected = {}
scan_rows = []
for path in sorted(shard_root.glob("shard_*/selected_cases.json")):
    raw = json.loads(path.read_text(encoding="utf-8"))
    for level, buckets in raw.items():
        dst = selected.setdefault(str(level), {})
        for bucket, values in buckets.items():
            if not isinstance(values, list):
                values = [values]
            merged = dst.setdefault(bucket, [])
            for index in values:
                index = int(index)
                if index not in merged:
                    merged.append(index)
for path in sorted(shard_root.glob("shard_*/scan_selected_cases.csv")):
    if path.stat().st_size == 0:
        continue
    with path.open("r", encoding="utf-8", newline="") as f:
        scan_rows.extend(dict(row) for row in csv.DictReader(f))

selected = {
    str(level): {
        bucket: sorted(indices)[:cases_per_bucket]
        for bucket, indices in sorted(buckets.items())
    }
    for level, buckets in sorted(selected.items(), key=lambda item: int(item[0]))
}
selected_out.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
if scan_rows:
    fields = list(scan_rows[0].keys())
    with scan_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scan_rows)
else:
    scan_out.write_text("", encoding="utf-8")
print(f"Merged selected cases: {selected_out}")
print(json.dumps(selected, indent=2, ensure_ascii=False))
PY
fi

log_stage "2/3 Compute grad x attention at each step and render heatmaps"
CUDA_VISIBLE_DEVICES="${FINAL_GPU:-0}" "${PYTHON_BIN}" "${common_args[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --markdown_path "${MARKDOWN_PATH}" \
  --device cuda:0

echo "Wrote markdown: ${MARKDOWN_PATH}"
echo "Wrote images:   ${OUTPUT_DIR}"
log_stage "3/3 Visualization completed"
