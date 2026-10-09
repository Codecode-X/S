#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
usage() {
  cat <<'EOF'
Usage: bash scripts/main/<frozenlake|maze|minibehaviour>.sh [all|train|eval]
      bash scripts/additional/maze_flip.sh [all|train|eval]
  all   Prepare images -> train -> evaluate the final checkpoint -> summarize (default)
  train Prepare images -> train and save weights and logs
  eval  Prepare images -> evaluate models/VP-mist/<task>_epoch5 -> summarize

Environment variables:
  TASKS=<task selected by this entry point>
  QWEN_MODEL_PATH=<auto-detect a local base model, otherwise use Qwen/Qwen2.5-VL-7B-Instruct>
  QWEN_PROCESSOR_PATH=<same as the model path>
  NUM_GPUS=8 PER_DEVICE_BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1(the current training entry point only supports 1)
  LEARNING_RATE=1.5e-4                VP-mist training learning rate
  NUM_EPOCHS=10 STOP_AFTER_EPOCHS=5 SEED=2026 EVAL_MAX_SAMPLES=1000
  DATA_ROOT=<project>/dataset CHECKPOINT_ROOT=<project>/models/VP-mist
  RUN_DIR=<new result directory> QWEN_VISUAL_IMAGE_ROOT=<shared image cache>
  SKIP_IMAGE_PREP=1                    Skip decoding when a complete image cache exists
  VQVAE_DIR=<local VQ-VAE>              Download from Hugging Face if unset
  USE_KV_CACHE=0                       Set to 1 to cache the planning inference prefix
You can also set TRAIN_DATASET / EVAL_DATASET / TEST_DATASET / CHECKPOINT.
RUN_DIR must not exist; each run uses a separate directory.
EOF
}
MODE=all
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if [[ $# -gt 0 ]]; then MODE="$1"; shift; fi
[[ $# -eq 0 && "$MODE" =~ ^(all|train|eval)$ ]] || { usage >&2; exit 2; }
export MODE
: "${TASKS:?Use a scripts/main/*.sh or scripts/additional/maze_flip.sh entry point}"
export TASKS
[[ "$TASKS" =~ ^(frozenlake|maze|minibehaviour|maze_flip)$ ]] || {
  echo 'Each entry point runs one task; invoke the corresponding experiment scripts separately.' >&2
  exit 2
}
if [[ -z "${QWEN_MODEL_PATH:-}" ]]; then
  QWEN_MODEL_PATH="$(python - "${CHECKPOINT_ROOT:-$ROOT/models/VP-mist}" <<'PY'
import json, sys
from pathlib import Path
for adapter in sorted(Path(sys.argv[1]).glob('*/backbone/adapter_config.json')):
    candidate = json.loads(adapter.read_text()).get('base_model_name_or_path', '')
    if candidate and (Path(candidate)/'config.json').is_file():
        print(candidate)
        break
else:
    print('Qwen/Qwen2.5-VL-7B-Instruct')
PY
)"
fi
export QWEN_MODEL_PATH
export QWEN_PROCESSOR_PATH="${QWEN_PROCESSOR_PATH:-$QWEN_MODEL_PATH}"
export QWEN_VISUAL_IMAGE_ROOT="${QWEN_VISUAL_IMAGE_ROOT:-$ROOT/output/qwen_visual_initial_maps_vqvae}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/dataset}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/models/VP-mist}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}" STOP_AFTER_EPOCHS="${STOP_AFTER_EPOCHS:-5}"
export LEARNING_RATE="${LEARNING_RATE:-1.5e-4}"
export QWEN_PROCESSOR_USE_FAST="${QWEN_PROCESSOR_USE_FAST:-false}"
export SEED="${SEED:-2026}" EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1000}"
export MASTER_PORT="${MASTER_PORT:-29621}"
export SWANLAB_MODE="${SWANLAB_MODE:-disabled}" PYTHONUNBUFFERED=1
export SKIP_IMAGE_PREP="${SKIP_IMAGE_PREP:-0}"
export USE_KV_CACHE="${USE_KV_CACHE:-0}"
export RUN_DIR="${RUN_DIR:-$ROOT/output/reproductions/$(date -u +%Y%m%dT%H%M%SZ)_${MODE}_$$}"
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"; export RUN_DIR
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/metadata"
exec > >(tee -a "$RUN_DIR/logs/pipeline.log") 2>&1
finish() {
  local rc=$?
  trap - EXIT
  printf 'exit_code=%s\nfinished_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$RUN_DIR/status.txt"
  if [[ $rc -eq 0 ]]; then echo "Completed: $RUN_DIR"; else echo "Run failed (exit code ${rc}); see $RUN_DIR/logs/pipeline.log" >&2; fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Result directory: %s\nMode: %s\n' "$RUN_DIR" "$MODE"
python - <<'PY'
import json, os, platform
from pathlib import Path
keys = '''MODE TASKS QWEN_MODEL_PATH QWEN_PROCESSOR_PATH QWEN_VISUAL_IMAGE_ROOT
DATA_ROOT CHECKPOINT_ROOT NUM_GPUS PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE NUM_EPOCHS STOP_AFTER_EPOCHS SEED EVAL_MAX_SAMPLES MASTER_PORT SWANLAB_MODE SKIP_IMAGE_PREP
CUDA_VISIBLE_DEVICES VQVAE_DIR TRAIN_DATASET EVAL_DATASET TEST_DATASET CHECKPOINT
RESUME_FROM_CHECKPOINT QWEN_PROCESSOR_USE_FAST RUN_DIR USE_KV_CACHE'''.split()
keys += ['MAZE_FLIP_DATA_ROOT']
cfg = {k: os.environ.get(k) for k in keys}
if cfg['USE_KV_CACHE'] not in ('0', '1'): raise SystemExit('USE_KV_CACHE must be 0 or 1')
cfg['effective_global_batch'] = int(cfg['NUM_GPUS']) * int(cfg['PER_DEVICE_BATCH_SIZE']) * int(cfg['GRADIENT_ACCUMULATION_STEPS'])
if int(cfg['GRADIENT_ACCUMULATION_STEPS']) != 1:
    raise SystemExit('The VP-mist training entry point requires gradient_accumulation_steps=1')
cfg['model'] = dict(project='VP-mist', state_encoder='recurrent_implicit_state', state_token_mode='append', action_output_mode='text_token', explicit_position_input=False)
if cfg['TASKS'] == 'maze_flip':
    cfg['experiment'] = dict(name='MazeFlip: Dynamic Observation Frames',
                             observation='initial_map_only',
                             left_right_move='toggle_vertical_flip',
                             up_down_move='toggle_horizontal_flip',
                             action_coordinate_frame='current_view')
cfg['implementation_defaults'] = dict(lr=float(cfg['LEARNING_RATE']), warmup_ratio=0.1, scheduler='cosine', alpha=0.7, lora_r=32, lora_alpha=64, lora_dropout=0.1)
cfg['python'] = platform.python_version()
cfg['scope'] = 'Reference experiment workflow with the VP-mist recurrent implicit-state model; paper scores are comparison values only.'
Path(os.environ['RUN_DIR'], 'metadata/config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False)+'\n')
if cfg['effective_global_batch'] != 128:
    print('Note: global batch size differs from the default 128; the training trajectory will change.')
PY

BASE_TRAIN="${TRAIN_DATASET:-}"; BASE_EVAL="${EVAL_DATASET:-}"; BASE_TEST="${TEST_DATASET:-}"
task="$TASKS"
train="${BASE_TRAIN:-$DATA_ROOT/$task/tokenized_dataset/SFT_random/train_dataset.jsonl}"
test="${BASE_TEST:-$DATA_ROOT/$task/tokenized_dataset/SFT/test_dataset.jsonl}"
val="${BASE_EVAL:-$test}"
python - "$task" "$train" "$val" "$test" <<'PY'
import json, os, sys
from pathlib import Path
task, train, val, test = sys.argv[1:]
records = {role: str(Path(name).resolve()) for role, name in
           {'train': train, 'validation': val, 'test': test}.items()}
Path(os.environ['RUN_DIR'], 'metadata', task+'_inputs.json').write_text(json.dumps(records, indent=2)+'\n')
PY
run_stage() {
  local label="$1"; shift
  { printf '%q ' "$@"; printf '\n'; } >> "$RUN_DIR/commands.sh"
  echo "[$(date -u +%FT%TZ)] $label"
  "$@" 2>&1 | tee "$RUN_DIR/logs/$label.log"
}
printf '#!/usr/bin/env bash\n' > "$RUN_DIR/commands.sh"
export TRAIN_DATASET="${BASE_TRAIN:-$DATA_ROOT/$task/tokenized_dataset/SFT_random/train_dataset.jsonl}"
export TEST_DATASET="${BASE_TEST:-$DATA_ROOT/$task/tokenized_dataset/SFT/test_dataset.jsonl}"
export EVAL_DATASET="${BASE_EVAL:-$TEST_DATASET}"
if [[ "$SKIP_IMAGE_PREP" != 1 ]]; then
  if [[ "$MODE" == eval ]]; then
    run_stage "${task}_prepare" env TASKS="$task" TRAIN_DATASET="$TEST_DATASET" EVAL_DATASET="$TEST_DATASET" bash scripts/common/prepare_images.sh
  else
    run_stage "${task}_prepare" env TASKS="$task" bash scripts/common/prepare_images.sh
  fi
elif [[ ! -d "$QWEN_VISUAL_IMAGE_ROOT" ]]; then
  echo "Image cache does not exist: $QWEN_VISUAL_IMAGE_ROOT" >&2; exit 1
fi
if [[ "$MODE" == all || "$MODE" == train ]]; then
  run_stage "${task}_train" env SKIP_IMAGE_PREP=1 OUTPUT_DIR="$RUN_DIR/models/$task" RUN_NAME="vp_mist_${task}_seed${SEED}" bash scripts/common/train.sh "$task"
  ckpt="$(python - "$RUN_DIR/models/$task" <<'PY'
from pathlib import Path
import re, sys
paths = [p for p in Path(sys.argv[1]).glob('checkpoint-*') if re.fullmatch(r'checkpoint-\d+', p.name) and (p/'trainer_state.json').is_file()]
if not paths: raise SystemExit('Training did not produce a valid checkpoint')
print(max(paths, key=lambda p: int(p.name.split('-')[-1])).resolve())
PY
)"
else
  ckpt="${CHECKPOINT:-$CHECKPOINT_ROOT/${task}_epoch5}"
  ckpt="$(realpath "$ckpt")"
fi
printf '%s\n' "$ckpt" > "$RUN_DIR/metadata/${task}_checkpoint.txt"
if [[ "$MODE" != train ]]; then
  run_stage "${task}_evaluate" env OUTPUT_DIR="$RUN_DIR/evals/$task" bash scripts/common/evaluate.sh "$task" "$ckpt"
fi
if [[ "$MODE" == train ]]; then exit 0; fi
python - <<'PY'
import csv, json, os
from pathlib import Path
root = Path(os.environ['RUN_DIR'])
reference = {'frozenlake': (99.5,99.8), 'maze': (99.5,99.6), 'minibehaviour': (96.0,97.9)}
rows = []
for task in os.environ['TASKS'].split(','):
    p = root/'evals'/task/'summary.json'
    evaluated = json.loads(p.read_text())
    if len(evaluated) != 1: raise SystemExit(f'Expected one evaluated checkpoint per task: {p}')
    result = evaluated[0]

    result['path'] = (root/'metadata'/f'{task}_checkpoint.txt').read_text().strip()
    p.write_text(json.dumps(evaluated, indent=2)+'\n')
    ref_em, ref_pr = reference.get(task, (None, None))
    rows.append(dict(task=task, checkpoint=result['path'], samples=result['samples'],
                     em_pct=result['em']*100, pr_pct=result['pr']*100,
                     paper_em_pct=ref_em, paper_pr_pct=ref_pr,
                     delta_em_pp=result['em']*100-ref_em if ref_em is not None else None,
                     delta_pr_pp=result['pr']*100-ref_pr if ref_pr is not None else None,
                     legal_rate=result['legal_rate'], entropy=result['entropy']))
(root/'results.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False)+'\n')
with (root/'results.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines = ['# SSVR / VP-mist experiment results', '',
         'Model: VP-mist recurrent implicit state. Paper SSVR scores are reference values. Measured scores are percentages; differences are percentage points.',
         'The final training checkpoint is evaluated without selecting weights by test-set scores.', '',
         '| Task | Samples | EM % | PR % | Paper EM % | Paper PR % | Delta EM | Delta PR |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
def display(value, spec):
    return format(value, spec) if value is not None else '—'
for r in rows:
    lines.append(f"| {r['task']} | {r['samples']} | {r['em_pct']:.3f} | {r['pr_pct']:.3f} | {display(r['paper_em_pct'], '.1f')} | {display(r['paper_pr_pct'], '.1f')} | {display(r['delta_em_pp'], '+.3f')} | {display(r['delta_pr_pp'], '+.3f')} |")
lines += ['', 'Training and evaluation use the current VP-mist model implementation.',
          'Baselines, robustness, VQA, ablations and visualization use separate experiment entry points.',
          'Configuration, dataset paths and checkpoint paths are in metadata/; stage logs are in logs/.',
          'SEED controls seed_everything; Trainer retains the default seed 42.']
(root/'results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
