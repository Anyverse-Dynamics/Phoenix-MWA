#!/usr/bin/env bash
# Explicit NVIDIA/Accelerate entrypoint. Does not start unless invoked.
set -euo pipefail
if [[ "${ROBOCASA_SMOKE:-0}" != 0 || -n "${ROBOCASA_SMOKE_SELECTION:-}" ]]; then
    echo 'RoboCasa smoke runs moved to Phoenix-devtools; refusing to start formal training.' >&2
    exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${PYTHON:?}" "${RUN_ID:?Use a new run name}" "${ROBOCASA_PACKAGE:?}" "${ROBOCASA_CACHE:?}"
: "${WAM_PRETRAIN_CHECKPOINT:?}" "${CSVAE_CHECKPOINT:?}" "${DINO_ENCODER_CHECKPOINT:?}" "${DIFFSYNTH_MODEL_BASE_PATH:?}"
export ROBOCASA_REGISTRY="$ROOT/configs/embodiment_registry/robocasa_gr1_delta80_slot18_v1.json"
export PATH="$(dirname "$PYTHON"):$PATH" PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export WANDB_MODE="${WANDB_MODE:-disabled}" TORCH_COMPILE_ENABLED=false WAM_TIMING_STEPS=0
export NCCL_DEBUG=WARN HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RESUME_STATE FASTWAM_LOW_VRAM STOP_AFTER_STEP WANDB_RUN_ID WANDB_RESUME
unset WAM_TIMING_DIR WAM_TIMING_START_STEP
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]
NNODES="${NNODES:-1}" NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
for value in "$NNODES" "$NPROC_PER_NODE" "$NODE_RANK"; do [[ "$value" =~ ^[0-9]+$ ]]; done
(( NNODES > 0 && NPROC_PER_NODE > 0 && NODE_RANK < NNODES ))
WORLD="$((NNODES * NPROC_PER_NODE))"
[[ "$WORLD" == "${EXPECTED_WORLD_SIZE:?}" ]] || { echo 'World-size mismatch' >&2; exit 2; }
if (( NNODES > 1 )); then : "${MASTER_ADDR:?}"; fi
OUT="${WAM_OUTPUT_ROOT:-$ROOT/runs}/robocasa/$RUN_ID"
ENTRY=scripts/train.py
ARGS=(task=robocasa_posttrain "output_dir=$OUT")
for arg in "$@"; do
    key="${arg%%=*}"; key="${key#++}"; key="${key#+}"; key="${key#~}"
    case "$key" in output_dir|task|resume|hydra.run.dir|hydra.sweep.dir)
        echo "Reserved override: $key" >&2; exit 2;; esac
done
ARGS+=("$@")
[[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
LAUNCH=("$PYTHON" -m accelerate.commands.launch
    --config_file scripts/accelerate_configs/accelerate_joint96_posttrain_zero1.yaml
    --num_machines "$NNODES" --num_processes "$WORLD" --machine_rank "$NODE_RANK"
    --main_process_ip "${MASTER_ADDR:-127.0.0.1}" --main_process_port "${MASTER_PORT:-24621}"
    --deepspeed_multinode_launcher standard "$ENTRY" "${ARGS[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${LAUNCH[@]}"; printf '\n'; exit 0; fi
"$PYTHON" -c 'import sys; from wam.posttrain.resources import validate_launch; validate_launch("robocasa", sys.argv[1:])' "${ARGS[@]:1}"
exec "${LAUNCH[@]}"
