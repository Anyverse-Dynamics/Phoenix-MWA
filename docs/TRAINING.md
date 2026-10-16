# Training

`scripts/train_libero.sh` selects `configs/task/libero_posttrain.yaml`: weights-only initialization, four-suite cached data, canonical80 actions, 32-action chunks and embodiment soft prompts. Configure resources with `configs/machines/local.example.sh`.

`scripts/train_robocasa.sh` selects `configs/task/robocasa_posttrain.yaml`. Supply the resource variables required by the launcher and its preflight. Structural checks remain enforced; runtime file hash checks are disabled. Numerical behavior still needs validation on the target accelerator.

Distributed launchers run once per node. Set `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` and `EXPECTED_WORLD_SIZE` consistently. Use a fresh `RUN_ID`. Global batch equals per-device batch times world size times gradient accumulation.

Set `WAM_PRETRAIN_CHECKPOINT` and `EXPECTED_PRETRAIN_STEP` explicitly before sourcing the machine resource example. The latter must match the step inside the initialization checkpoint; no default checkpoint step is assumed. It is the pretraining initialization step, not the target post-training step. Model dimensions, encoder/cache compatibility, coordinate transforms, normalization and prompt slots remain checked.

The public task configs contain default hyperparameters, not fixed experimental requirements. Override them through the training launcher, for example `batch_size=4 gradient_accumulation_steps=2 max_steps=25000 save_every=5000 learning_rate=5e-5`. `scripts/preflight_libero.py --resources` accepts the same Hydra overrides and validates the resolved configuration. GPU training smoke remains explicitly bounded to four GPUs and three updates, using the configured initialization checkpoint.

Evaluation uses each policy's saved configuration through `tools/prepare_eval.py`; training recipe defaults do not override a released policy's inference configuration.

Outputs default to `runs/`. W&B is offline by default. Supply credentials through scheduler environment variables if online logging is enabled; never commit credential files. Full-state resume is not exposed by this release. Installation launches no jobs.
