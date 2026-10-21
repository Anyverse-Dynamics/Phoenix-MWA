#!/usr/bin/env bash
# Minimal RoboTwin validation: training, both evaluation modes, or both in order.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  printf '%s\n' \
    'Usage: bash scripts/run_robotwin_smoke.sh [train|eval|all] [options]' \
    '  train: four GPUs, batch 1/device, two updates; accepts training overrides' \
    '  eval:  task 0 / episode 0 in C2C and C2R; accepts evaluation options' \
    '  all:   train then eval on ROBOTWIN_SMOKE_GPU (default 0); no options' \
    'No mode defaults to all. Evaluation uses ROBOTWIN_CHECKPOINT, not the smoke output.'
}

run_train() (
  # Keep training-only environment overrides out of the following evaluation.
  export NNODES=1 NPROC_PER_NODE=4 NODE_RANK=0 EXPECTED_WORLD_SIZE=4
  export MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-24591}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export PER_DEVICE_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=2 SAVE_EVERY=2
  export WANDB_MODE=disabled WANDB_ENABLED=false
  export RUN_ID="${RUN_ID:-robotwin_train_smoke_$(date +%Y%m%d_%H%M%S)}"
  exec bash "$ROOT/scripts/train_robotwin.sh" num_workers=2 log_every=1 eval_every=0 save_training_state=false "$@"
)

run_eval() {
  # Complete the original episode in each mode; never shorten simulator horizons.
  local mode
  for mode in c2c c2r; do
    bash "$ROOT/scripts/eval_robotwin.sh" --mode "$mode" --max-tasks 1 --episodes 1 "$@"
  done
}

MODE="${1:-all}"
if (( $# > 0 )); then shift; fi
case "$MODE" in
  train) run_train "$@" ;;
  eval) run_eval "$@" ;;
  all)
    if (( $# > 0 )); then usage >&2; exit 2; fi
    run_train
    run_eval --gpus "${ROBOTWIN_SMOKE_GPU:-0}"
    ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
