#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "${ROBOTWIN_ENV_FILE:-}" ]]; then source "$ROBOTWIN_ENV_FILE"; fi
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT/src:$ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname "$(command -v "$PYTHON")"):$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export EMBODIMENT_REGISTRY="$ROOT/configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json"
export ROBOTWIN_NORM_STATS="${ROBOTWIN_NORM_STATS:-$ROOT/assets/stats/robotwin.json}"
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_PROJECT="${WANDB_PROJECT:-phoenix-mwa}"
export TORCH_COMPILE_ENABLED=false HYDRA_FULL_ERROR=1 NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
: "${WAM_PRETRAIN_CHECKPOINT:?}" "${STVAE_CHECKPOINT:?}" "${DINO_ENCODER_CHECKPOINT:?}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?}" "${ROBOTWIN_DATA_ROOT:?}" "${ROBOTWIN_TEXT_CACHE:?}"
NNODES="${NNODES:-8}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ "$NNODES" != 1 ]]; then
  NODE_RANK="${NODE_RANK:-${RANK:?Set NODE_RANK or RANK for each node}}"
else
  NODE_RANK="${NODE_RANK:-0}"
fi
for value in "$NNODES" "$NPROC_PER_NODE" "$NODE_RANK"; do [[ "$value" =~ ^[0-9]+$ ]]; done
(( NNODES > 0 && NPROC_PER_NODE > 0 && NODE_RANK < NNODES ))
WORLD=$((NNODES*NPROC_PER_NODE))
[[ "$WORLD" == "${EXPECTED_WORLD_SIZE:-64}" ]] || { echo 'World-size mismatch' >&2; exit 2; }
if (( NNODES > 1 )); then : "${MASTER_ADDR:?}"; fi
export RUN_ID="${RUN_ID:?Choose a new run name}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]
OUTPUT="${WAM_OUTPUT_ROOT:-$ROOT/runs}/robotwin/$RUN_ID"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output $OUTPUT" >&2; exit 2; }
unset RESUME_STATE WANDB_RUN_ID WANDB_RESUME
for arg in "$@"; do
  key="${arg%%=*}"; key="${key#++}"; key="${key#+}"; key="${key#~}"
  case "$key" in
    task|output_dir|resume|hydra.run.dir|hydra.sweep.dir)
      echo "Reserved override: $key; use the supported recipe and RUN_ID/WAM_OUTPUT_ROOT" >&2; exit 2;;
  esac
done
ARGS=(task=robotwin_posttrain "output_dir=$OUTPUT"
  "batch_size=${PER_DEVICE_BATCH_SIZE:-8}" "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}"
  "max_steps=${MAX_STEPS:-null}" "save_every=${SAVE_EVERY:-5000}"
  "wandb.mode=$WANDB_MODE" "wandb.enabled=${WANDB_ENABLED:-true}" "$@")
exec "$PYTHON" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_joint96_posttrain_zero1.yaml \
  --num_machines "$NNODES" --num_processes "$WORLD" --machine_rank "$NODE_RANK" \
  --main_process_ip "${MASTER_ADDR:-127.0.0.1}" --main_process_port "${MASTER_PORT:-24491}" \
  --deepspeed_multinode_launcher standard scripts/train.py "${ARGS[@]}"
