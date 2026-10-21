#!/usr/bin/env bash
# Run from any directory; paths are resolved from this repository.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?Usage: install_simulator_env.sh robocasa|robotwin ENV_DIR}"
ENV_DIR="${2:?Supply a separate simulator environment directory}"
case "$MODE" in robocasa|robotwin) ;; *) echo "Unknown simulator: $MODE" >&2; exit 2;; esac
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "${BASE_PYTHON:-python3.10}" -m venv "$ENV_DIR"
fi
grep -q '^include-system-site-packages = false$' "$ENV_DIR/pyvenv.cfg" || {
  echo 'Use an isolated venv without inherited site-packages' >&2; exit 2;
}
PY="$ENV_DIR/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
if [[ "$MODE" == robotwin ]]; then
  : "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the matching simulator source and assets}"
  command -v nvcc >/dev/null || { echo 'Install the CUDA development toolkit and expose nvcc on PATH' >&2; exit 2; }
  "$PY" -m pip install --upgrade pip wheel 'setuptools<70' ninja 'cmake<4'
  "$PY" -m pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
fi
"$PY" -m pip install -r "$ROOT/configs/$MODE-sim-requirements.txt"
if [[ "$MODE" == robotwin ]]; then
  # The simulator release can omit Git metadata; this is package version metadata,
  # not a model checkpoint-step or file-digest validation.
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.7.8
  export MAX_JOBS="${MAX_JOBS:-2}"
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    TORCH_CUDA_ARCH_LIST="$("$PY" -c 'import torch; assert torch.cuda.is_available(), "Expose the target GPU or set TORCH_CUDA_ARCH_LIST for cross-compilation"; print(";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))')"
    export TORCH_CUDA_ARCH_LIST
  fi
  echo "Building CuRobo for CUDA architectures: $TORCH_CUDA_ARCH_LIST"
  "$PY" -m pip install --no-build-isolation -e "$ROBOTWIN_ROOT/envs/curobo"
fi
"$PY" -m pip check
"$PY" -c 'import site, sys; assert not site.ENABLE_USER_SITE; print("Simulator interpreter:", sys.executable)'
