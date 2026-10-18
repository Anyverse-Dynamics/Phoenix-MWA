#!/usr/bin/env bash
# One policy server and one simulator process; this shell owns both lifetimes.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -u
: "${CHECKPOINT:?Set the posttraining weight checkpoint}"
: "${TRAIN_CONFIG:?Set the saved training config.yaml}"
export ROBOCASA_PACKAGE="${ROBOCASA_PACKAGE:-$ROOT/assets/robocasa}"
export ROBOCASA_SCENE_SEEDS="${ROBOCASA_SCENE_SEEDS:-$ROOT/assets/robocasa/scene_seed_list.json}"
: "${ENV_NAME:?Set the registered gr1_unified environment name}"
: "${EVAL_ROOT:?Set a new evaluation output directory}"
: "${POLICY_PYTHON:?Use the accepted NVIDIA policy environment}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?}" "${ROBOCASA_SCENE_SEEDS:?}"
export SCENE_SEED_MODE=seed_list SCENE_SEED_LIST="$ROBOCASA_SCENE_SEEDS"
if [[ "${ROBOCASA_DIAGNOSTIC_SMOKE:-0}" != 1 ]]; then
    "$POLICY_PYTHON" "$ROOT/tools/require_robocasa_acceptance.py"
else
    [[ "${N_EPISODES:-0}" =~ ^[123]$ ]] || { echo 'Diagnostic smoke is limited to 1..3 episodes' >&2; exit 2; }
fi
[[ ! -e "$EVAL_ROOT" ]] || { echo "Refusing existing EVAL_ROOT: $EVAL_ROOT" >&2; exit 2; }
mkdir -p "$EVAL_ROOT"
if [[ "${ROBOCASA_DIAGNOSTIC_SMOKE:-0}" != 1 ]]; then
    "$POLICY_PYTHON" "$ROOT/tools/require_robocasa_acceptance.py" --output "$EVAL_ROOT/evaluation_acceptance.json"
fi
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/robocasa_delta_numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/robocasa_delta_mpl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
source "$ROOT/scripts/robocasa_sim_env.sh"
PORT="${PORT:-39100}"
SEED="${SEED:-7}"
EXECUTE_STEPS="${EXECUTE_STEPS:-12}"
EXTRA=()
if [[ "${ALLOW_SMOKE:-0}" == 1 ]]; then EXTRA+=(--allow-smoke); fi
SCENE_ARGS=()
if [[ -n "${SCENE_SEED_LIST:-}" ]]; then SCENE_ARGS+=(--scene-seed-list "$SCENE_SEED_LIST"); fi
"$POLICY_PYTHON" tools/serve_robocasa_joint96.py \
  --checkpoint "$CHECKPOINT" --config "$TRAIN_CONFIG" --package "$ROBOCASA_PACKAGE" \
  --device "${DEVICE:-cuda:0}" --port "$PORT" --sim-root "$SIM_ROOT" \
  --execute-steps "$EXECUTE_STEPS" --denoising-steps "${DENOISING_STEPS:-10}" \
  --seed "$SEED" "${EXTRA[@]}" > "$EVAL_ROOT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$SIM_PYTHON" tools/eval_robocasa_joint96.py \
  --host 127.0.0.1 --port "$PORT" --sim-root "$SIM_ROOT" \
  --expected-checkpoint "$CHECKPOINT" --expected-package "$ROBOCASA_PACKAGE" \
  --output "$EVAL_ROOT/results" --env "$ENV_NAME" --episodes "${N_EPISODES:-50}" \
  --execute-steps "$EXECUTE_STEPS" --max-episode-steps "${MAX_EPISODE_STEPS:-720}" \
  --seed "$SEED" --scene-seed-mode "${SCENE_SEED_MODE:-legacy_unseeded}" "${SCENE_ARGS[@]}" 2>&1 | tee "$EVAL_ROOT/client.log"
