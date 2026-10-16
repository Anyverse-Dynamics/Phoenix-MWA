#!/usr/bin/env bash
# Separate no-Torch simulator environment; never install Robosuite1.5 into LIBERO1.4.
ROBOCASA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ROBOCASA_RESOURCES:?}"
export SIM_ROOT="$ROBOCASA_REPO/third_party/robocasa_runner"
export SIM_PYTHON="${SIM_PYTHON:-$ROBOCASA_RESOURCES/environments/robocasa-sim/bin/python}"
export ROBOCASA_SOURCE="$ROBOCASA_RESOURCES/simulator/robocasa-gr1-tabletop-tasks"
export ROBOSUITE_SOURCE="$ROBOCASA_RESOURCES/simulator/robosuite"
export PYTHONPATH="$ROBOCASA_REPO/src:$ROBOCASA_REPO:$ROBOCASA_SOURCE:$ROBOSUITE_SOURCE:$SIM_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/phoenix_robocasa_numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/phoenix_robocasa_mpl}"
test -x "$SIM_PYTHON"
for file in "$ROBOCASA_SOURCE/robocasa/__init__.py" "$ROBOSUITE_SOURCE/robosuite/__init__.py"; do
    test -s "$file" || { echo "Missing simulator source: $file" >&2; return 2; }
done
