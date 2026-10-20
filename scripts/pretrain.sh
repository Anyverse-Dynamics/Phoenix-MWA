#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PHOENIX_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/resources/weights"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${PYTHON:-python}"
if [[ "${PHOENIX_SMOKE:-0}" != 0 || "${DRY_RUN:-0}" != 0 ]]; then
    echo 'Development launch flags are unsupported by this formal entry.' >&2
    exit 2
fi
NNODES="${NNODES:-28}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${RANK:-}}"
: "${RUN_ID:?Set a new shared RUN_ID on all nodes}"
: "${MASTER_ADDR:?Set MASTER_ADDR on all nodes}"
MASTER_PORT="${MASTER_PORT:-24319}"
if [[ "$NNODES" != 28 || "$NPROC_PER_NODE" != 8 ]]; then
    echo 'Formal recipe requires 28 x 8 GPUs.' >&2
    exit 2
fi
if [[ ! "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES )); then
    echo 'Set a distinct NODE_RANK in [0, NNODES) on every node.' >&2
    exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo 'RUN_ID must be a single path-safe experiment name.' >&2
    exit 2
fi
# Only rank 0 owns creation; later workers may see its newly created directory.
if [[ "$NODE_RANK" == 0 && -e "$ROOT/runs/$RUN_ID" ]]; then
    echo "Refusing to overwrite $ROOT/runs/$RUN_ID; choose a new RUN_ID." >&2
    exit 2
fi
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE)) LOCAL_WORLD_SIZE="$NPROC_PER_NODE"
exec "$PYTHON" -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/accelerate_local_joint96_zero1.yaml \
    --num_machines "$NNODES" --num_processes "$WORLD_SIZE" \
    --machine_rank "$NODE_RANK" --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" --deepspeed_multinode_launcher standard \
    scripts/pretrain.py "$@"
