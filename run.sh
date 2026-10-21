#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python_bin="${PYTHON:-python3}"
command_name="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command_name" in
  server) exec "$python_bin" -m phoenix.server "$@" ;;
  client) exec "$python_bin" -m phoenix.client "$@" ;;
  prepare) exec "$python_bin" -m phoenix.prepare "$@" ;;
  train) exec bash "$project_root/scripts/train_phoenix.sh" "$@" ;;
  data) exec "$python_bin" -m phoenix.data "$@" ;;
  env)
    mode="${1:-rtc}"
    if [[ $# -gt 0 ]]; then shift; fi
    case "$mode" in rtc) watermark=8; fixed=true;; sync) watermark=0; fixed=false;; *) exit 2;; esac
    exec "$python_bin" -m prometheus.workflows.run_env --config-name run_env/phoenix \
      env.action_hz=30 env.max_chunk_horizon=32 env.policy_low_watermark="$watermark" \
      env.fixed_chunk_switch="$fixed" "$@"
    ;;
  help|-h|--help)
    echo 'Prometheus robot runtime'
    echo 'Bundled example: Phoenix Training RTC (H32 / video9 / denoise10)'
    echo 'Usage: PYTHON=/path/to/python bash run.sh data|train|server|client|prepare|env [arguments]'
    echo 'Client --mode rtc uses training-time prefix conditioning; --mode sync is the comparison.'
    echo 'Model assets are external. The env command initializes hardware resources.'
    ;;
  *) echo 'Unknown Prometheus command' >&2; exit 2 ;;
esac
