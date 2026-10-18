#!/usr/bin/env bash
# Explicit four-GPU, three-update acceptance; never used by formal training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${PYTHON:?}" "${RUN_ID:?Use a fresh smoke ID}" "${WAM_PRETRAIN_CHECKPOINT:?}"
export PATH="$(dirname "$PYTHON"):$PATH" PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export WANDB_MODE=disabled WAM_TIMING_STEPS=0 TORCH_COMPILE_ENABLED=false
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
export EMBODIMENT_REGISTRY="$ROOT/configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json"
export LIBERO_FLU_BI_NORM_STATS="$ROOT/assets/stats/phoenix_libero.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${SMOKE_GPUS:-0,1,2,3}"
unset RESUME_STATE FASTWAM_LOW_VRAM STOP_AFTER_STEP WANDB_RUN_ID WANDB_RESUME
unset WAM_TIMING_DIR WAM_TIMING_START_STEP
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]
OUT="$ROOT/validation_runs/$RUN_ID"
[[ ! -e "$OUT" ]] || { echo "Refusing existing smoke output: $OUT" >&2; exit 2; }
"$PYTHON" - <<'PY'
import os, subprocess
gpus = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
assert len(gpus) == len(set(gpus)) == 4, 'Smoke requires exactly four explicitly selected GPUs'
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
usage = {int(line.split(',')[0]): int(line.split(',')[1]) for line in rows.splitlines()}
assert all(g in usage and usage[g] < 1024 for g in gpus), 'Selected GPU is absent or occupied'
PY
# Verify the configured initialization and resource compatibility first.
"$PYTHON" scripts/preflight_libero.py --resources
exec "$PYTHON" -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/accelerate_joint96_posttrain_zero1.yaml \
    --num_machines 1 --num_processes 4 --machine_rank 0 \
    --main_process_ip 127.0.0.1 --main_process_port "${MASTER_PORT:-24511}" \
    --deepspeed_multinode_launcher standard scripts/smoke_train_libero.py \
    task=libero_posttrain "output_dir=$OUT" \
    batch_size=1 gradient_accumulation_steps=1 max_steps=3 num_epochs=50 \
    num_workers=2 log_every=1 save_every=0 eval_every=0 \
    save_training_state=false save_on_train_end=true max_weight_checkpoints=1 \
    wandb.enabled=false wandb.mode=disabled
