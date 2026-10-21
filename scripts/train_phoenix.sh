#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python_bin="${PYTHON:-python3}"
processes="${NPROC_PER_NODE:-1}"
nodes="${NNODES:-1}"
if [[ "$processes" == 1 && "$nodes" == 1 ]]; then
  exec "$python_bin" -m phoenix.train "$@"
fi
exec "$python_bin" -m torch.distributed.run \
  --nnodes "$nodes" --nproc-per-node "$processes" --node-rank "${NODE_RANK:-0}" \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT:-29500}" \
  --module phoenix.train "$@"
