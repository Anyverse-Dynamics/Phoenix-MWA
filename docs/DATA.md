# Data contracts

LIBERO combines spatial, object, goal and long-horizon LeRobot suites. Put the four `libero_*_no_noops_lerobot` directories under `data/libero/`. The cached recipe uses main/wrist camera slots 0/2, padding the unused slot. RGB offsets are 0,4,...,32; action offsets are 0,...,31.

Wan (48 channels) and CS-VAE (48 channels) form the 96-channel visual representation. Text/visual caches and completion/binding manifests must match the encoder weights and data metadata. Cache generation is a separate preparation step, not an implicit training side effect.

LIBERO uses FLU/B=I canonical80 actions. RoboCasa GR1 uses Delta80 slot18. Never interchange normalization statistics or coordinate conventions. Only path metadata in shipped statistics is sanitized; numerical values are preserved.

RoboCasa's main-camera cache window has shape `(96,3,16,16)`; the reader restores absent camera slots. See YAML recipes for splitting and masking rules. Keep model configuration, numeric normalization statistics and codec settings with the weights. File digest fields in existing artifacts are optional metadata, not runtime admission checks.
