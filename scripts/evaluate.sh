#!/usr/bin/env bash
# Public entrypoint. Resource settings are explicit shell files owned by the user.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${1:-}" == --help || $# -lt 2 ]]; then
    echo 'Usage: bash scripts/evaluate.sh {libero-plus|libero-pro|robocasa-gr1} RESOURCE_ENV [full|smoke|preflight]'
    echo 'Generate RESOURCE_ENV with tools/prepare_eval.py; see docs/EVALUATION.md.'
    exit 0
fi
benchmark="$1"; resources="$2"; phase="${3:-full}"
case "$benchmark" in libero-plus|libero-pro|robocasa-gr1) ;; *) echo 'Unsupported benchmark' >&2; exit 2;; esac
[[ -f "$resources" ]] || { echo "Missing resource configuration: $resources" >&2; exit 2; }
# shellcheck disable=SC1090
source "$resources"
if [[ -n "${PPU_SDK_ROOT:-}" ]]; then
    [[ -f "$PPU_SDK_ROOT/envsetup.sh" ]] || { echo 'Invalid PPU_SDK_ROOT' >&2; exit 2; }
    source "$PPU_SDK_ROOT/envsetup.sh" cuda
fi
case "$benchmark" in
  libero-pro)
    case "$phase" in full|smoke|preflight) ;; *) echo 'Invalid LIBERO-Pro phase' >&2; exit 2;; esac
    exec bash scripts/eval_libero_pro.sh "$phase"
    ;;
  libero-plus)
    case "$phase" in full|smoke|preflight) ;; *) echo 'Invalid LIBERO-Plus phase' >&2; exit 2;; esac
    export EVAL_SELECTION=full
    exec bash scripts/eval_libero_plus.sh "$phase"
    ;;
  robocasa-gr1)
    case "$phase" in
      full) ;;
      smoke) export N_EPISODES=1 ROBOCASA_TASK_LIMIT=1 ;;
      preflight) export PREFLIGHT_ONLY=1 ;;
      *) echo 'Invalid RoboCasa phase' >&2; exit 2;;
    esac
    exec bash scripts/eval_robocasa_joint96_full24.sh
    ;;
esac
