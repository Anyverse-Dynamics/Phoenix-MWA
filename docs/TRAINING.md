# Phoenix Pretraining

Run commands from the repository root after completing the
[installation](../README.md#getting-started) and [resource preparation](RESOURCES.md).
The final recipe is [`configs/phoenix_pretrain.yaml`](../configs/phoenix_pretrain.yaml).
This checkout uses CSVAE format v2 and `bundled://dino-csvae-v2`. Prepare
`resources/weights/csvae/csvae.pt` before launching. Earlier encoder
checkpoints and full training states are incompatible with this format.

Exported policy weights set `joint96_metadata.schema` to
`phoenix.local_joint96.weights.v1`, matching Phoenix-MWA. This policy schema is
separate from the CSVAE encoder's `format_version`. Earlier policy exports need
an explicit metadata conversion before use; tensor values do not need to change.
The data and full-state resume schemas retain their existing identifiers.

## Distributed Launch

The formal launcher requires **28 nodes × 8 GPUs**, batch 20 per GPU and one
gradient accumulation step, giving a global batch of 4,480. It uses BF16,
DeepSpeed ZeRO-1 without communication overlap, and seed 42.

Activate the same compatible environment on every node. Make the checkout,
resource package, output directory and raw dataset paths available consistently
across the allocation. Run once per node:

```bash
export PYTHON="$(command -v python)"
export NNODES=28 NPROC_PER_NODE=8
export NODE_RANK="${NODE_RANK:?Set a distinct rank from 0 to 27}"
export MASTER_ADDR="${MASTER_ADDR:?Set the rank-0 address}"
export MASTER_PORT=24319 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUN_ID=phoenix_final_001
unset RESUME_STATE
bash scripts/pretrain.sh
```

All nodes use the same `RUN_ID`, address and port. Rank 0 refuses an existing
output directory. Outputs are written to `runs/$RUN_ID/`.

The launcher sets `PHOENIX_ROOT`, the local model root, `PYTHONPATH` and offline
model-loading flags. It does not source a private environment or credentials.

## Final Recipe

`scripts/pretrain.sh` uses the following settings throughout a new run:

- 256,013 optimizer updates and the same fixed cosine schedule horizon.
- 10,000 warmup updates; peak learning rate 1e-4 and minimum 1e-6.
- Weight decay 0.01 and gradient clipping at 1.0.
- Main-view masking 3%, sensor noise 7.5%, language augmentation 70%.
- Clean-main-view protection in each four-sample block.
- Save and validation every 5,000 updates, plus final weights and full state.

The mask and sensor-noise events are mutually exclusive and require a real
auxiliary view. Single-real-view samples are neither masked nor noised by this
augmentation. Clean-block protection reserves a clean eligible main view in
each block of four. The recipe retains the original RGB dual-version selection,
excluded subsets, paired-view identities, language selection and train-only
normalization.

Visual encoding is online: each view supplies nine 256 × 256 RGB frames sampled
from a 33-timestamp window. The model predicts 32 canonical80 actions. WanVAE,
Phoenix Encoder and CS-VAE remain frozen; the video/action DiTs, state encoder
and embodiment prompts are trainable. UMT5 contexts are read from the supplied
cache.

The loss is `0.5 × Wan latent loss + 0.5 × CS-VAE latent loss + action loss`.
There is no additional encoder distillation, memory, VLM or track-teacher
objective in this recipe.

## Same-Recipe Resume

Use a complete state produced by the current CSVAE format v2 checkout, the same
topology and recipe, and a new output run name:

```bash
export RESUME_STATE="$PWD/runs/phoenix_final_001/checkpoints/state/step_005000"
export RUN_ID=phoenix_final_resume_001
bash scripts/pretrain.sh
```

Run on every node with the distributed variables set as above. Full-state resume
restores model, optimizer, scheduler, sampler progress and rank-specific RNG
state. A weights-only checkpoint is not a replacement for the state directory.
Data, normalization, encoder, sampler, completion, topology and recipe checks
remain enforced. Saved contracts include resolved resource paths; use matching
resource locations and a compatible model configuration. Runtime loading does
not verify source or checkpoint file hashes.

## Logging

W&B is disabled by default. To enable online logging, append these overrides to
the training command and provide credentials through the environment:

```bash
bash scripts/pretrain.sh \
  wandb.enabled=true wandb.mode=online \
  wandb.workspace=YOUR_ENTITY wandb.project=phoenix-pretrain
```

The launcher neither copies nor sources a secrets file. Keep the same recipe
overrides on every distributed worker.
