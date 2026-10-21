# Encoder pretraining

**English** | [简体中文](README_zh.md) | [Repository overview](../../README.md)

Phoenix's visual encoder combines DINOv3-L/16 with an eight-block spatiotemporal
adapter. A frozen VGGT-Ω teacher supplies geometric supervision; a frozen
V-JEPA 2.1 teacher supplies temporal feature targets. This module trains that
encoder, before CS-VAE or world–action model training.

## Model and objective

Each example is an RGB clip, center-cropped to a square and resized to 256 × 256.
DINO produces a 16 × 16 patch grid with 1,024 channels per frame. The adapter
uses 16 attention heads, 16 register tokens, spatial attention within each frame
and temporal attention across frames. Relative frame positions preserve the
sampling stride. It returns `[B, T, 256, 1024]` features.

The default recipe initializes DINO from the `aggregator.patch_embed.` weights
in the VGGT-Ω checkpoint. It alternates teachers **once per optimizer update**;
all microbatches of an accumulated update use the same teacher.

- **VGGT-Ω:** supervise adapter blocks 1, 3, 5 and 7 against the teacher's
  intermediate features, and decode student features through the frozen depth
  head. The final relation term ramps to 0.1 over the first 5% of training.
- **V-JEPA 2.1:** apply a spatial tube mask through DINO's mask-token input,
  retaining the full adapter patch grid. The predictor replaces masked features
  with learned mask tokens and predicts temporally pooled teacher features.
  Prediction and target are layer-normalized before the L1 feature loss.
  Context loss ramps to 0.5 over the first 20% of optimizer updates. During the
  first 20% of V-JEPA updates, mask coverage grows from 35–55% to 50–90%,
  alternating small and large spatial block profiles.
- **EMA:** update the deployment student after each optimizer update, with
  momentum 0.9999. Teachers, projectors and the predictor are not deployed. Reconstructed constants
  such as image normalization buffers are copied exactly, so export and resume
  do not reset constants that have drifted through floating-point averaging.

Training and deployment share `wam.encoder.multi_teacher_distillation`.
It uses dense masking with a curriculum and alternates teacher objectives.

## Install and provide resources

Use the repository's Python 3.10 environment and compatible PyTorch build.
The core dependencies already include Accelerate, OmegaConf, PyAV and PyArrow.
The shared encoder is part of the Phoenix package. W&B is optional (`pip install -e '.[logging]'`).
The default video decoder is PyAV; Decord, OpenCV and TorchCodec are optional
backends and require their corresponding packages.

DINO source is already bundled. Provide local checkouts of the two teacher
implementations and their official pretrained weights. Teacher code and model
weights remain under their respective licenses; they are not copied into this
module. Use the following exact source revisions:

| Teacher source | Revision |
| --- | --- |
| VGGT-Ω | `39a0cb8af88554f15ddcb5354cd52bde588fa014` |
| V-JEPA 2.1 | `204698b45b3712590f06245fbfba32d3be539812` |
| DINOv3, bundled | `6876159a11b4df116f30f667f8c9888617df0751` |

Set resource locations in your shell or replace the environment interpolations
in a private copy of `configs/encoder_pretrain/dinov3_multiteacher.yaml`:

```bash
export PHOENIX_VGGT_SOURCE="$PWD/resources/vggt-omega"
export PHOENIX_VJEPA_SOURCE="$PWD/resources/vjepa2"
export PHOENIX_VGGT_CHECKPOINT="$PWD/checkpoints/vggt_omega_1b_512.pt"
export PHOENIX_VJEPA_CHECKPOINT="$PWD/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
export PHOENIX_EGO_DATA="$PWD/data/ego_example"
export PHOENIX_ROBOT_DATA="$PWD/data/robot_example"
export PYTHONPATH="$PWD/src"
```

The two dataset entries are examples, not the original training corpus.
Set each entry's `video_key` to its primary camera and `sampling_group` to
`ego_general` or `robot`. A dataset may use one `path`, multiple `paths`, or
explicit `subdatasets` patterns. LeRobot V3 `meta/info.json`, episode Parquet
metadata and videos are required. The loader discovers all `dtype: video`
features; auxiliary cameras are sampled as separate clip streams.

The recipe samples lengths 4/8/12/16 and strides 1/2, with 32 frames per device
per microbatch. Temperature sampling targets ego/general:robot = 40:60 and
primary:auxiliary = 50:50. These targets require eligible streams in the
corresponding groups; an infeasible mixture fails instead of silently changing
the proportions. Adjust the targets to match your corpus.

## Preflight and launch

Preflight only resolves the configuration, checks source revisions, verifies
weight-file presence and inspects video metadata. It neither loads teacher
weights nor decodes the full corpus:

```bash
python -m wam.encoder_pretrain preflight --config configs/encoder_pretrain/dinov3_multiteacher.yaml
```

Launch training explicitly with a new run ID:

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m wam.encoder_pretrain train \
  --config configs/encoder_pretrain/dinov3_multiteacher.yaml \
  run_id=encoder_run_01
```

The checked-in recipe uses 160,000 updates. Training starts only with the
`train` command. W&B is disabled by default; enable `tracking.enabled=true`
and configure `wandb_project`/`wandb_entity` if desired. JSONL metrics are saved
locally regardless of W&B.

## Resume and export

Run outputs live under `run_root_dir/run_id`:

- `config.full.yaml`: resolved configuration.
- `metrics.jsonl`: losses, learning rates and sampling composition.
- `checkpoints/step_XXXXXXXX.pt`: online/EMA student, objectives, optimizer,
  scheduler, sampler and per-rank RNG state.
- `ema_student_final.pt`: student-only deployment weights, metadata and config.

A nonempty run directory requires an explicit `trainer.resume_from=...`.
Resume requires the same world size, gradient accumulation, model, data and
training schedule. Logging/checkpoint intervals may change. Teacher weights
are reloaded from the configured paths, so keep those files immutable. Catalog
caches contain pickled Python objects: use only caches created by this module
in your own trusted output location, and change `catalog_cache_version` if
source metadata changes. The cache uses filesystem locks; shared multi-node
storage must provide cross-node locking before a distributed launch.

Full resume checkpoints use `phoenix.encoder_pretrain.resume.v1`. Exported
students use `phoenix.encoder.student.v1` and retain the runtime feature contract.
In `OnlineDinov3DistilledEncoder`, use `checkpoint_state="student"` for
`ema_student_final.pt`, or `checkpoint_state="ema_student"` for a full training
checkpoint. The export includes adapter configuration for the existing loader.

## Module map

| Location | Responsibility |
| --- | --- |
| `wam.encoder_pretrain.__main__` | Explicit preflight/train command |
| `config.py`, `build.py` | Resource validation and construction of the fixed recipe |
| `model.py` | Phoenix training checkpoint contract over the shared runtime |
| `data/` | LeRobot V3 clips, metadata catalog, multi-view sampling and video decode |
| `trainer.py`, `multiview_trainer.py` | Optimizer-step alternation, accumulation, EMA, resume and metrics |
| `wam.encoder.multi_teacher_distillation` | Shared student, adapter and teacher losses |
| `wam.encoder.external_sources` | Pinned teacher source loading |
| `wam.third_party.dino_runtime.dinov3` | Pinned DINO source |
