#!/usr/bin/env bash
# Run once per DLC node. No credentials, data or models are installed/copied here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
PYTHON="$(command -v "$PYTHON")"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
export EMBODIMENT_REGISTRY="${EMBODIMENT_REGISTRY:-$ROOT/configs/embodiment_registry/local_robot_arm18_capacity28_v1.json}"
export LIBERO_FLU_BI_NORM_STATS="$ROOT/assets/stats/phoenix_libero.json"
export LIBERO_PAIR_BINDING="${LIBERO_PAIR_BINDING:-${LIBERO_STVAE_CACHE:?}/JOINT96_PAIR_BINDING.json}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-phoenix-mwa}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" HYDRA_FULL_ERROR=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset RESUME_STATE FASTWAM_LOW_VRAM STOP_AFTER_STEP
unset WANDB_RUN_ID WANDB_RESUME WAM_TIMING_DIR WAM_TIMING_START_STEP
export WAM_TIMING_STEPS=0 TORCH_COMPILE_ENABLED=false
export RUN_ID="${RUN_ID:?Choose a NEW RUN_ID; existing outputs are refused}"
: "${WAM_PRETRAIN_CHECKPOINT:?}" "${STVAE_CHECKPOINT:?}" "${DINO_ENCODER_CHECKPOINT:?}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?}" "${LIBERO_DATA_ROOT:?}" "${LIBERO_TEXT_CACHE:?}"
: "${LIBERO_WAN_VAE_CACHE:?}" "${LIBERO_STVAE_CACHE:?}"
NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
for value in "$NNODES" "$NPROC_PER_NODE" "$NODE_RANK"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Invalid topology' >&2; exit 2; }
done
(( NNODES > 0 && NPROC_PER_NODE > 0 && NODE_RANK < NNODES ))
WORLD="$((NNODES * NPROC_PER_NODE))"
[[ "$WORLD" == "${EXPECTED_WORLD_SIZE:-32}" ]] || { echo 'World-size mismatch' >&2; exit 2; }
if (( NNODES > 1 )); then : "${MASTER_ADDR:?}"; fi
OUTPUT="${WAM_OUTPUT_ROOT:-$ROOT/runs}/libero/$RUN_ID"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'RUN_ID must be a simple name' >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 2; }
for arg in "$@"; do
    key="${arg%%=*}"
    key="${key#++}"; key="${key#+}"; key="${key#~}"
    case "$key" in
        output_dir|task|resume|hydra.run.dir|hydra.sweep.dir)
            echo "Reserved override: $key; use RUN_ID / WAM_OUTPUT_ROOT and the supported task" >&2; exit 2;;
    esac
done
ARGS=(task=libero_posttrain "output_dir=$OUTPUT"
      "wandb.mode=$WANDB_MODE" "wandb.enabled=${WANDB_ENABLED:-true}" "$@")
LAUNCH=("$PYTHON" -m accelerate.commands.launch
    --config_file "$ROOT/scripts/accelerate_configs/accelerate_joint96_posttrain_zero1.yaml"
    --num_machines "$NNODES" --num_processes "$WORLD" --machine_rank "$NODE_RANK"
    --main_process_ip "${MASTER_ADDR:-127.0.0.1}" --main_process_port "${MASTER_PORT:-24455}"
    --deepspeed_multinode_launcher standard "$ROOT/scripts/train.py" "${ARGS[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${LAUNCH[@]}"; printf '\n'
    exit 0
fi
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/.cache/huggingface/datasets}"
if [[ "$WANDB_MODE" == online && "${WANDB_ENABLED:-true}" == true ]]; then
    : "${WANDB_API_KEY:?Inject credentials through the scheduler; no hardcoded secret file}"
fi
"$PYTHON" scripts/preflight_libero.py --resources "${ARGS[@]:1}"
exec "${LAUNCH[@]}"
