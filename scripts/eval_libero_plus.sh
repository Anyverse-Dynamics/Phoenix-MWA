#!/usr/bin/env bash
# Explicit resources; never sources a key file or an old checkout's environment.
set -euo pipefail
if [[ -n "${EVAL_SELECTION:-}" && "${EVAL_SELECTION}" != full ]]; then
    echo 'Reduced LIBERO panels moved to Phoenix-devtools; use full for benchmark evaluation.' >&2
    exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ "${1:-full}" == full && $# -le 1 ]] || { echo 'Use full for benchmark evaluation' >&2; exit 2; }
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/LIBERO-plus${EVAL_EXTRA_PYTHONPATH:+:$EVAL_EXTRA_PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export EVAL_RENDER_BACKEND="${EVAL_RENDER_BACKEND:-egl}"
case "$EVAL_RENDER_BACKEND" in
    egl) ;;
    osmesa) unset MUJOCO_EGL_DEVICE_ID ;;
    *) echo 'EVAL_RENDER_BACKEND must be egl or osmesa' >&2; exit 2 ;;
esac
export MUJOCO_GL="$EVAL_RENDER_BACKEND" PYOPENGL_PLATFORM="$EVAL_RENDER_BACKEND"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1
: "${JOINT96_EVAL_RESOURCE_DIR:?}"
: "${EVAL_OUTPUT:?}" "${LIBERO_BENCHMARK_ROOT:?}"
: "${DINO_ENCODER_CHECKPOINT:?}" "${CSVAE_CHECKPOINT:?}" "${DIFFSYNTH_MODEL_BASE_PATH:?}"
exec "${PYTHON:-python}" -m eval.libero_plus.runner --phase "${1:-full}" \
    --output "$EVAL_OUTPUT" --gpus "${EVAL_GPUS:-0,1,2,3,4,5,6,7}" \
    --workers-per-gpu "${WORKERS_PER_GPU:-1}" \
    --shards "${EVAL_SHARDS:-1}" --shard-rank "${EVAL_SHARD_RANK:-0}"
