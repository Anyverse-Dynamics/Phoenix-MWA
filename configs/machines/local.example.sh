#!/usr/bin/env bash
# Source from the repository root; override resources through environment variables.
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-checkpoints}"
export WAM_PRETRAIN_CHECKPOINT="${WAM_PRETRAIN_CHECKPOINT:-checkpoints/pretrain.pt}"
# Required: match the step stored in WAM_PRETRAIN_CHECKPOINT; no historical default.
: "${EXPECTED_PRETRAIN_STEP:?Set the actual initialization checkpoint step}"
export EXPECTED_PRETRAIN_STEP
export STVAE_CHECKPOINT="${STVAE_CHECKPOINT:-checkpoints/stvae.pt}"
export DINO_ENCODER_CHECKPOINT="${DINO_ENCODER_CHECKPOINT:-checkpoints/dino.pt}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-data/libero}"
export LIBERO_TEXT_CACHE="${LIBERO_TEXT_CACHE:-data/cache/libero/text}"
export LIBERO_WAN_VAE_CACHE="${LIBERO_WAN_VAE_CACHE:-data/cache/libero/wanvae}"
export LIBERO_STVAE_CACHE="${LIBERO_STVAE_CACHE:-data/cache/libero/stvae}"
export LIBERO_PAIR_BINDING="${LIBERO_PAIR_BINDING:-$LIBERO_STVAE_CACHE/JOINT96_PAIR_BINDING.json}"
