#!/usr/bin/env bash
# Source from the repository root; override resources through environment variables.
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-checkpoints}"
export WAM_PRETRAIN_CHECKPOINT="${WAM_PRETRAIN_CHECKPOINT:-checkpoints/pretrain.pt}"
# Checkpoint progress is read automatically; no expected-step setting is needed.
export CSVAE_CHECKPOINT="${CSVAE_CHECKPOINT:-resources/encoders/csvae.pt}"
export DINO_ENCODER_CHECKPOINT="${DINO_ENCODER_CHECKPOINT:-resources/encoders/dino110k.pt}"
# FastWAM LeRobot v2.1 archives; see README.md#libero-training-data.
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${WAM_DATA_ROOT:-data}/libero_mujoco3.3.2}"
export LIBERO_TEXT_CACHE="${LIBERO_TEXT_CACHE:-data/cache/libero/text}"
export LIBERO_WAN_VAE_CACHE="${LIBERO_WAN_VAE_CACHE:-data/cache/libero/wanvae}"
export LIBERO_CSVAE_CACHE="${LIBERO_CSVAE_CACHE:-data/cache/libero/csvae}"
export LIBERO_PAIR_BINDING="${LIBERO_PAIR_BINDING:-$LIBERO_CSVAE_CACHE/JOINT96_PAIR_BINDING.json}"
