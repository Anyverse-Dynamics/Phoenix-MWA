# Source from the project root. Replace resource locations with your downloads.
export PYTHON="${PYTHON:-python}"
export ROBOTWIN_SIM_PYTHON="${ROBOTWIN_SIM_PYTHON:-resources/robotwin/sim-env/bin/python}"
# Training-only inputs below are not read by eval_robotwin.sh.
export WAM_PRETRAIN_CHECKPOINT="${WAM_PRETRAIN_CHECKPOINT:-$PWD/checkpoints/phoenix_pretrained.pt}"
export ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-$PWD/data/robotwin}"
export ROBOTWIN_TEXT_CACHE="${ROBOTWIN_TEXT_CACHE:-$PWD/data/cache/robotwin-text}"
# Inference resources. Keep caller overrides, including legacy deployment envs.
export DINO_ENCODER_CHECKPOINT="${DINO_ENCODER_CHECKPOINT:-$PWD/checkpoints/encoders/phoenix_encoder.pt}"
export STVAE_CHECKPOINT="${STVAE_CHECKPOINT:-$PWD/checkpoints/encoders/phoenix_stvae.pt}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$PWD/checkpoints}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$PWD/resources/robotwin/RoboTwin}"
export ROBOTWIN_FROZEN_C2C="${ROBOTWIN_FROZEN_C2C:-$PWD/resources/robotwin/frozen/c2c}"
export ROBOTWIN_FROZEN_C2R="${ROBOTWIN_FROZEN_C2R:-$PWD/resources/robotwin/frozen/c2r}"
export ROBOTWIN_CHECKPOINT="${ROBOTWIN_CHECKPOINT:-$PWD/checkpoints/robotwin/policy.pt}"
export ROBOTWIN_INFERENCE_CONFIG="${ROBOTWIN_INFERENCE_CONFIG:-${ROBOTWIN_TRAIN_CONFIG:-$PWD/configs/inference/phoenix_robotwin.yaml}}"
export ROBOTWIN_EVAL_STATS="${ROBOTWIN_EVAL_STATS:-$PWD/assets/stats/robotwin.json}"
