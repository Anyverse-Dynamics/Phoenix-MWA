#!/usr/bin/env bash
# Bound concurrent task workers while preserving the legacy rank-to-GPU assignment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CHECKPOINT:?Set a stable posttraining weight checkpoint}"
: "${TRAIN_CONFIG:?Set the saved training config.yaml}"
: "${ROBOCASA_PACKAGE:?Set the matching numeric/text package}"
: "${OUTROOT:?Set a new evaluation output directory}"
: "${POLICY_PYTHON:?}" "${ROBOCASA_SCENE_SEEDS:?}"
export SCENE_SEED_LIST="$ROBOCASA_SCENE_SEEDS"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
NUM_GPUS_TOTAL="${NUM_GPUS_TOTAL:-1}"
BASE_PORT_START="${BASE_PORT_START:-39100}"
RANK_STAGGER_SECONDS="${RANK_STAGGER_SECONDS:-30}"
export N_EPISODES="${N_EPISODES:-50}" SEED="${SEED:-7}"
export SCENE_SEED_MODE=seed_list
export EXECUTE_STEPS="${EXECUTE_STEPS:-12}" DENOISING_STEPS="${DENOISING_STEPS:-10}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
export CHECKPOINT TRAIN_CONFIG ROBOCASA_PACKAGE
[[ "$NUM_GPUS_TOTAL" =~ ^[1-9][0-9]*$ && "$NUM_GPUS_TOTAL" -le 24 ]]
[[ "$BASE_PORT_START" =~ ^[1-9][0-9]*$ && "$BASE_PORT_START" -le 65512 ]]
[[ "$RANK_STAGGER_SECONDS" =~ ^[0-9]+$ ]]
[[ "$N_EPISODES" =~ ^[1-9][0-9]*$ ]]
[[ "$SCENE_SEED_MODE" == legacy_unseeded || "$SCENE_SEED_MODE" == fixed || "$SCENE_SEED_MODE" == seed_list ]] || {
  echo 'SCENE_SEED_MODE must be legacy_unseeded, fixed or seed_list.' >&2; exit 2;
}
if [[ "$SCENE_SEED_MODE" == seed_list ]]; then
  : "${SCENE_SEED_LIST:?Set a saved scene seed list JSON manifest}"
  [[ -f "$SCENE_SEED_LIST" ]]
  export SCENE_SEED_LIST
elif [[ -n "${SCENE_SEED_LIST:-}" ]]; then
  echo 'SCENE_SEED_LIST requires SCENE_SEED_MODE=seed_list.' >&2; exit 2
fi
[[ -f "$CHECKPOINT" && -f "$TRAIN_CONFIG" && -f "$ROBOCASA_PACKAGE/package.json" ]]
MAX_WORKERS=24
if [[ -n "${WORKERS_PER_GPU:-}" ]]; then
  [[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
    echo 'WORKERS_PER_GPU must be a positive integer.' >&2; exit 2;
  }
  MAX_WORKERS=$((NUM_GPUS_TOTAL * WORKERS_PER_GPU))
  if ((MAX_WORKERS > 24)); then MAX_WORKERS=24; fi
fi
source "$ROOT/scripts/robocasa_sim_env.sh"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "$POLICY_PYTHON" tools/require_robocasa_acceptance.py
fi

TASKS=(
  PnPCupToDrawerClose PnPPotatoToMicrowaveClose PnPMilkToMicrowaveClose
  PnPBottleToCabinetClose PnPWineToCabinetClose PnPCanToDrawerClose
  PosttrainPnPNovelFromCuttingboardToBasketSplitA
  PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
  PosttrainPnPNovelFromCuttingboardToPanSplitA
  PosttrainPnPNovelFromCuttingboardToPotSplitA
  PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA
  PosttrainPnPNovelFromPlacematToBasketSplitA
  PosttrainPnPNovelFromPlacematToBowlSplitA
  PosttrainPnPNovelFromPlacematToPlateSplitA
  PosttrainPnPNovelFromPlacematToTieredshelfSplitA
  PosttrainPnPNovelFromPlateToBowlSplitA
  PosttrainPnPNovelFromPlateToCardboardboxSplitA
  PosttrainPnPNovelFromPlateToPanSplitA
  PosttrainPnPNovelFromPlateToPlateSplitA
  PosttrainPnPNovelFromTrayToCardboardboxSplitA
  PosttrainPnPNovelFromTrayToPlateSplitA
  PosttrainPnPNovelFromTrayToPotSplitA
  PosttrainPnPNovelFromTrayToTieredbasketSplitA
  PosttrainPnPNovelFromTrayToTieredshelfSplitA
)
TASK_LIMIT="${ROBOCASA_TASK_LIMIT:-24}"
[[ "$TASK_LIMIT" =~ ^[1-9][0-9]*$ && "$TASK_LIMIT" -le 24 ]] || { echo 'ROBOCASA_TASK_LIMIT must be 1..24' >&2; exit 2; }
TASKS=("${TASKS[@]:0:TASK_LIMIT}")
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ ! -e "$OUTROOT" ]] || { echo "Refusing existing OUTROOT: $OUTROOT" >&2; exit 2; }
  mkdir -p "$OUTROOT"
  "$POLICY_PYTHON" tools/require_robocasa_acceptance.py --output "$OUTROOT/evaluation_acceptance.json"
  exec > >(tee -a "$OUTROOT/launcher.top.log") 2>&1
  if [[ "$SCENE_SEED_MODE" == seed_list ]]; then
    cp -- "$SCENE_SEED_LIST" "$OUTROOT/scene_seed_list.json"
    export SCENE_SEED_LIST="$OUTROOT/scene_seed_list.json"
    PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" "$SIM_PYTHON" - "$SCENE_SEED_LIST" "$N_EPISODES" <<'PY'
import json, sys
from wam.robocasa_simulation import load_scene_seed_list
print(json.dumps(load_scene_seed_list(sys.argv[1], int(sys.argv[2]))), flush=True)
PY
  fi
  ENV_NAMES=()
  for task in "${TASKS[@]}"; do
    ENV_NAMES+=("gr1_unified/${task}_GR1ArmsAndWaistFourierHands_Env")
  done
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$SIM_PYTHON" tools/check_robocasa_simulator.py \
    --sim-root "$SIM_ROOT" --env "${ENV_NAMES[@]}" --seed "$SEED"
  [[ "${PREFLIGHT_ONLY:-0}" != 1 ]] || exit 0
fi
echo "checkpoint=$CHECKPOINT tasks=${#TASKS[@]} gpus=$NUM_GPUS_TOTAL episodes/task=$N_EPISODES"
echo "seed=$SEED policy_seed_mode=fixed_per_replan execute=$EXECUTE_STEPS denoise=$DENOISING_STEPS max_steps=$MAX_EPISODE_STEPS"
echo "scene_seed_mode=$SCENE_SEED_MODE"
PIDS=()
TASK_RANKS=()
STATUS=0
cleanup() {
  for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for rank in "${!TASKS[@]}"; do
  slot=$((rank % MAX_WORKERS))
  if [[ -n "${PIDS[$slot]:-}" ]]; then
    if ! wait "${PIDS[$slot]}"; then
      echo "FAILED rank=${TASK_RANKS[$slot]} (see rank${TASK_RANKS[$slot]}.launcher.log)"
      STATUS=1
    fi
    unset 'PIDS[slot]'
  fi
  gpu=$((rank % NUM_GPUS_TOTAL))
  port=$((BASE_PORT_START + rank))
  task="gr1_unified/${TASKS[$rank]}_GR1ArmsAndWaistFourierHands_Env"
  echo "rank=$rank gpu=$gpu worker=$slot port=$port task=$task"
  [[ "${DRY_RUN:-0}" != 1 ]] || continue
  setsid env CUDA_VISIBLE_DEVICES="$gpu" DEVICE=cuda:0 MUJOCO_EGL_DEVICE_ID="$gpu" \
    PORT="$port" ENV_NAME="$task" EVAL_ROOT="$OUTROOT/rank$rank" \
    bash "$ROOT/scripts/eval_robocasa_joint96.sh" > "$OUTROOT/rank$rank.launcher.log" 2>&1 &
  PIDS[$slot]=$!
  TASK_RANKS[$slot]=$rank
  if ((rank + 1 < ${#TASKS[@]})); then sleep "$RANK_STAGGER_SECONDS"; fi
done
[[ "${DRY_RUN:-0}" != 1 ]] || exit 0
for slot in "${!PIDS[@]}"; do
  rank="${TASK_RANKS[$slot]}"
  if ! wait "${PIDS[$slot]}"; then echo "FAILED rank=$rank (see rank$rank.launcher.log)"; STATUS=1; fi
  unset 'PIDS[slot]'
done
"${POLICY_PYTHON:-python}" - "$OUTROOT" "$N_EPISODES" "${TASKS[@]}" <<'PY' || STATUS=1
import json
import hashlib
import sys
from pathlib import Path

root, expected = Path(sys.argv[1]), int(sys.argv[2])
rows, errors = [], []
seed_path = root / "scene_seed_list.json"
seed_data = seed_path.read_bytes() if seed_path.exists() else None
seed_list = json.loads(seed_data)["episode_seeds"][:expected] if seed_data is not None else None
for rank, name in enumerate(sys.argv[3:]):
    task = f"gr1_unified/{name}_GR1ArmsAndWaistFourierHands_Env"
    try:
        result = json.loads((root / f"rank{rank}/results/result.json").read_text())
        if not isinstance(result, list) or len(result) != 2 or result[0] != task:
            raise ValueError("wrong task/result schema")
        successes = result[1]
        if not isinstance(successes, list) or len(successes) != expected or any(type(v) is not bool for v in successes):
            raise ValueError("incomplete or invalid episode results")
        if seed_list is not None:
            config = json.loads((root / f"rank{rank}/results/config.json").read_text())
            audit = [json.loads(line) for line in
                     (root / f"rank{rank}/results/scene_seeds/env_0.jsonl").read_text().splitlines()]
            episodes = [row for row in audit if row.get("evaluation_episode")]
            if ([row["scene_seed"] for row in episodes] != seed_list or
                    [row["backend_seed"] for row in episodes] != seed_list or
                    [row["episode_index"] for row in episodes] != list(range(expected))):
                raise ValueError("actual backend episode seeds differ from saved seed list")
        rows.append(dict(rank=rank, task_name=task, n_episodes=len(successes),
                         n_successes=sum(successes), success_rate=sum(successes)/len(successes)))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(dict(rank=rank, task_name=task, error=str(exc)))
n = sum(row["n_episodes"] for row in rows)
s = sum(row["n_successes"] for row in rows)
summary = dict(status="FINAL" if not errors else "INCOMPLETE", per_task=rows, errors=errors,
               overall=dict(n_tasks=len(rows), n_episodes=n, expected_episodes=len(sys.argv[3:])*expected,
                            n_successes=s, success_rate=s/n if n else None))
summary["evaluation_acceptance"] = json.loads((root / "evaluation_acceptance.json").read_text())
if seed_list is not None:
    summary["scene_seed_list"] = dict(episode_seeds=seed_list,
        hash_validation=False, backend_seeds_verified=not errors)
(root / "merged_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"status": summary["status"], **summary["overall"]}))
sys.exit(bool(errors))
PY
exit "$STATUS"
