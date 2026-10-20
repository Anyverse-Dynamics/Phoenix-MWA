# Training

`scripts/train_libero.sh` selects `configs/task/libero_posttrain.yaml`: weights-only initialization, four-suite cached data, canonical80 actions, 32-action chunks and embodiment soft prompts. Configure resources with `configs/machines/local.example.sh`.

`scripts/train_robocasa.sh` selects `configs/task/robocasa_posttrain.yaml`. Supply the resource variables required by the launcher. Model and data shapes, coordinate conventions and normalization must match the selected checkpoint.

Distributed launchers run once per node. Set `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` and `EXPECTED_WORLD_SIZE` consistently. Use a fresh `RUN_ID`. Global batch equals per-device batch times world size times gradient accumulation.

Set `WAM_PRETRAIN_CHECKPOINT` before sourcing the machine resource example. Initialization, policy and encoder steps are read from checkpoints for reporting only; legacy expected-step settings are ignored. Missing progress metadata is recorded as unknown, not inferred from filenames. Model dimensions, encoder/cache semantics, coordinate transforms, normalization and prompt slots remain checked. Training duration and optimizer/scheduler state consistency checks are unchanged.

Phoenix pretraining exports and Phoenix-MWA initialization use
`phoenix.local_joint96.weights.v1`. Set `CSVAE_CHECKPOINT` to
`resources/encoders/csvae.pt` and `DINO_ENCODER_CHECKPOINT` to
`resources/encoders/dino110k.pt`, or override both with matching local files.
Cached LIBERO training uses `LIBERO_CSVAE_CACHE` and its matching pair binding.
The encoder and cache metadata must use the current `csvae` names.

The public task configs contain default hyperparameters, not fixed experimental requirements. Override them through the training launcher, for example `batch_size=4 gradient_accumulation_steps=2 max_steps=25000 save_every=5000 learning_rate=5e-5`.

Evaluation uses each policy's saved configuration through `tools/prepare_eval.py`; training recipe defaults do not override a released policy's inference configuration.

Outputs default to `runs/`. LIBERO and RoboTwin default to offline W&B; RoboCasa defaults to disabled logging. Supply credentials through scheduler environment variables if online logging is enabled; never commit credential files. Full-state resume is not exposed by this release. Installation launches no jobs.

## Recipe defaults and benchmark mapping

There are three training recipes, not four: LIBERO-Plus and LIBERO-Pro both
evaluate the four-suite LIBERO-trained policy. Their perturbations are evaluation
inputs, not additional training suites.

| Recipe | Evaluation targets | Batch/device; accumulation | Schedule; initial LR | Save interval |
| --- | --- | --- | --- | --- |
| `libero_posttrain` | LIBERO-Plus, LIBERO-Pro | 8; 1 | 50 epochs; 5e-5 | 5,000 updates |
| `robocasa_posttrain` | RoboCasa GR1 | 8; 1 | 60,000 updates; 1e-4 | 5,000 updates |
| `robotwin_posttrain` | RoboTwin C2C and C2R | 8; 1 | 50 epochs; 5e-5 | 5,000 updates |

All recipes predict 32 actions with benchmark-specific canonical80 mappings.
LIBERO uses FastWAM's preprocessed LeRobot v2.1 dataset; its text and visual
caches are separate prerequisites. See [DATA.md](DATA.md) for each recipe's
data requirements.

## LIBERO training

Download and extract the four FastWAM LeRobot v2.1 archives following
[LIBERO data preparation](../README.md#libero-training-data). The default data
root is `data/libero_mujoco3.3.2/`; to reuse an existing copy, export
`LIBERO_DATA_ROOT` before sourcing the machine example. Prepare matching caches
separately, then run from the repository root after installing `.[train,logging]`:

```bash
source configs/machines/local.example.sh
export PYTHON="$(command -v python)"
RUN_ID=libero_run NNODES=1 NPROC_PER_NODE=8 EXPECTED_WORLD_SIZE=8 \
  bash scripts/train_libero.sh
```

Supply four LeRobot suites, text/Wan/CS-VAE caches and pair-binding metadata.

## RoboCasa GR1 training

```bash
source configs/machines/local.example.sh
export PYTHON="$(command -v python)"
export ROBOCASA_PACKAGE=resources/robocasa-package
export ROBOCASA_CACHE=data/cache/robocasa
RUN_ID=robocasa_run NNODES=1 NPROC_PER_NODE=8 EXPECTED_WORLD_SIZE=8 \
  bash scripts/train_robocasa.sh
```

The package supplies numeric/text resources and normalization; the cache supplies
paired visual features. This uses native GR1 Delta80/slot18, not LIBERO coordinates.

## RoboTwin training

```bash
source configs/machines/robotwin.example.sh
RUN_ID=robotwin_run NNODES=1 NPROC_PER_NODE=8 EXPECTED_WORLD_SIZE=8 \
  bash scripts/train_robotwin.sh
```

This selects `configs/task/robotwin_posttrain.yaml`: clean 2,500-episode training,
three cameras, online RGB augmentation and frozen encoding, task-indexed text
cache, joint14-to-canonical80 actions and no memory conditioning. Offline visual
caches and a running simulator are not needed for training.

For 64-GPU training, run on every node with
`NNODES=8 NPROC_PER_NODE=8 EXPECTED_WORLD_SIZE=64`, a shared
new `RUN_ID`, scheduler-provided `MASTER_ADDR` and node-specific `NODE_RANK`.
RoboTwin also accepts `PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`,
`MAX_STEPS` and `SAVE_EVERY`. Edit its resource example or override values after
sourcing it. See [RoboTwin contracts and evaluation](ROBOTWIN.md).

See [EVALUATION.md](EVALUATION.md) for all four independent evaluation entries.
