#!/usr/bin/env bash
# Public entrypoint. Resource settings are explicit shell files owned by the user.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${1:-}" == --help || $# -lt 2 ]]; then
    echo 'Usage: bash scripts/evaluate.sh {libero-plus|libero-pro|robocasa-gr1} RESOURCE_ENV [full]'
    echo 'Generate RESOURCE_ENV with tools/prepare_eval.py; see docs/EVALUATION.md.'
    exit 0
fi
benchmark="$1"; resources="$2"; phase="${3:-full}"
[[ "$phase" == full && $# -le 3 ]] || { echo 'Only the full benchmark is available here' >&2; exit 2; }
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
    exec bash scripts/eval_libero_pro.sh "$phase"
    ;;
  libero-plus)
    if [[ -n "${EVAL_SELECTION:-}" && "${EVAL_SELECTION}" != full ]]; then
        echo 'Reduced LIBERO panels moved to Phoenix-devtools.' >&2; exit 2
    fi
    export EVAL_SELECTION=full
    exec bash scripts/eval_libero_plus.sh "$phase"
    ;;
  robocasa-gr1)
    exec bash scripts/eval_robocasa_joint96_full24.sh
    ;;
esac
