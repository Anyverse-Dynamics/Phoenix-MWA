# Data contracts

## LIBERO

Use FastWAM's preprocessed [LIBERO dataset](https://huggingface.co/datasets/yuanty/LIBERO-fastwam):
the four LeRobot **v2.1** `.tar.gz` archives for MuJoCo **3.3.2**, not the separate
`lerobot_v30/` directories or the upstream raw HDF5 demonstrations. See the
[download and extraction commands](../README.md#libero-training-data).

LIBERO combines spatial, object, goal and long-horizon suites. Put the four
`libero_*_no_noops_lerobot` directories under `data/libero_mujoco3.3.2/`, or set
`LIBERO_DATA_ROOT` to the parent directory of an existing copy. Each suite
contains `data/` (numeric Parquet records), `videos/` and `meta/`.
The cached recipe uses main/wrist camera slots 0/2, padding the unused slot.
RGB offsets are 0,4,...,32; action offsets are 0,...,31.

Wan (48 channels) and CS-VAE (48 channels) form the 96-channel visual representation. Text/visual caches and completion/binding manifests must match the encoder weights and data metadata. Cache generation is a separate preparation step, not an implicit training side effect.

The FastWAM dataset download does not include those caches. Keep
`LIBERO_TEXT_CACHE`, `LIBERO_WAN_VAE_CACHE`, `LIBERO_STVAE_CACHE` and
`LIBERO_PAIR_BINDING` configured separately; changing the dataset path does not
regenerate or rebind an existing cache.

LIBERO uses FLU/B=I canonical80 actions. RoboCasa GR1 uses Delta80 slot18. Never interchange normalization statistics or coordinate conventions. Only path metadata in shipped statistics is sanitized; numerical values are preserved.

RoboCasa's main-camera cache window has shape `(96,3,16,16)`; the reader restores absent camera slots. See YAML recipes for splitting and masking rules. Keep model configuration, numeric normalization statistics and codec settings with the weights. File digest fields in existing artifacts are optional metadata, not runtime admission checks.

## RoboTwin

`robotwin_posttrain` consumes the clean 2,500-episode subset (the first 50 episodes
of each 550-episode task block), with three cameras resized/cropped to 256×256.
RGB augmentation and frozen DINO/CS-VAE encoding run online. The task-indexed
text cache is still required. Native joint14 action/state is mapped to canonical80
with ARX5 embodiment slot 2; do not reuse LIBERO or RoboCasa statistics.
The recipe predicts 32 actions and does not use memory conditioning.

C2C/C2R evaluation consumes separate frozen task/episode seeds, unseen instructions
and scene/observation records. Training data is not an evaluation seed source.
See [RoboTwin resource requirements](RESOURCES.md#robotwin-training-and-evaluation-resources).

LIBERO-Plus and LIBERO-Pro use the same four-suite LIBERO training data; their
own BDDL and initialization resources are used only for their respective tests.
