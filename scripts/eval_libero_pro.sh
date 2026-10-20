#!/usr/bin/env bash
# LIBERO-Pro runs in a separate process; it never replaces the Plus installation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ "${1:-full}" == full && $# -le 1 ]] || { echo 'Use full for benchmark evaluation' >&2; exit 2; }
: "${LIBERO_PRO_REPO_ROOT:?}" "${LIBERO_BENCHMARK_ROOT:?}" "${EVAL_OUTPUT:?}"
: "${JOINT96_EVAL_RESOURCE_DIR:?}"
: "${DINO_ENCODER_CHECKPOINT:?}" "${CSVAE_CHECKPOINT:?}" "${DIFFSYNTH_MODEL_BASE_PATH:?}"
export PYTHONPATH="$ROOT/src:$ROOT:$LIBERO_PRO_REPO_ROOT"
export WAM_LIBERO_REPO=LIBERO-PRO EVAL_SELECTION=full EVAL_ENV_BACKEND=inprocess
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export EVAL_RENDER_BACKEND="${EVAL_RENDER_BACKEND:-egl}"
export MUJOCO_GL="$EVAL_RENDER_BACKEND" PYOPENGL_PLATFORM="$EVAL_RENDER_BACKEND"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false NUMBA_DISABLE_JIT=1
export DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1
exec "${PYTHON:-python}" -m eval.libero_pro.runner --phase "${1:-full}" \
    --output "$EVAL_OUTPUT" --gpus "${EVAL_GPUS:-0,1,2,3,4,5,6,7}" \
    --workers-per-gpu "${WORKERS_PER_GPU:-1}" \
    --shards "${EVAL_SHARDS:-1}" --shard-rank "${EVAL_SHARD_RANK:-0}"
