#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "${ROBOTWIN_ENV_FILE:-}" ]]; then source "$ROBOTWIN_ENV_FILE"; fi
export PYTHONPATH="$ROOT/src:$ROOT" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export EMBODIMENT_REGISTRY="$ROOT/configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json"
exec "${PYTHON:-python}" -m eval.robotwin.launch "$@"
