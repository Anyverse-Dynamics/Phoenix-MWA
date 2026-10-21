# Training RTC

The training entry point is `phoenix/train.py`, launched through `bash run.sh train`. The model predicts a 32-step action chunk jointly with a 9-frame video window. Training inputs span 33 raw frames at 30 Hz; video is sampled at offsets 0, 4, …, 32.

## Modules

| Module | Responsibility |
| --- | --- |
| `phoenix/data/` | Convert recordings, freeze windows, generate language and visual caches |
| `phoenix/dataset.py` | Read aligned parquet with RGB windows or frozen visual latents |
| `phoenix/training_data.py` | Map joint14 to canonical80 and shard deterministic batches |
| `phoenix/training.py` | Prepare prefix conditions, timestep tensors, suffix masks, and RTC metadata |
| `phoenix/losses.py` | Joint video/action objective |
| `phoenix/trainer.py` | Optimization, held-out evaluation, checkpoint publication, and resume |
| `phoenix/train.py` | Configuration, assets, model construction, and process setup |

## Window manifest

Generate the manifest and caches with the [data preparation guide](DATA_PREPARATION.md). The complete path is Prometheus recording → LeRobot v3 → frozen manifest → text/visual cache → Training RTC. Existing LeRobot v3 datasets may enter at the manifest stage.

The manifest is a UTF-8 JSON object with schema `phoenix.windows.v1`:

| Field | Contents |
| --- | --- |
| `fps` | `30` |
| `state_key`, `action_key` | Parquet column names for physical joint14 values |
| `tasks` | Nested mapping `tasks[root_id][task_index]` to instruction text |
| `files` | Mapping of parquet paths to `{bytes, mtime_ns}` |
| `episodes` | Episode descriptors |

Each episode contains `split` (`train` or `val`), `length`, `fps`, `dataset_from_index`, `episode_index`, `root_id`, `row_groups`, and `views`.

- A row group is `{path, row_group, start, end}`. The global index range is half-open: `[start,end)`.
- A view is `{path, bytes, mtime_ns, start_s, end_s}`. Its time range locates the episode within the video file.
- The three views must be ordered main camera, left wrist, right wrist.
- Use string `root_id` values and string task-index keys in the JSON mappings.
- Paths must be accessible on every training node; relative paths resolve from the repository working directory.

Parquet columns include `index`, `frame_index`, `episode_index`, `timestamp`, `task_index`, and the selected state/action columns. State and action rows each contain 14 finite physical values ordered left arm joints, left gripper, right arm joints, right gripper.

The reader checks file identities, contiguous indices, episode boundaries, and 30 Hz timestamps. It rejects windows crossing instruction boundaries. At episode ends, it repeats the final RGB frame and masks invalid image/action supervision. Each window retains a real first frame and future visual supervision.

## Normalization

Use `phoenix.joint14.normalization.v1` with `gripper_unit` and `state`/`action` objects, each containing `mean[14]` and positive `std[14]`. Select statistics consistent with the initialization and deployment coordinate system.

```bash
python -m phoenix.training_inputs \
  --normalization assets/normalization.json \
  --output assets/phoenix_normalization.json --gripper-unit m
```

This command copies numeric values into the shared schema. It does not fit new statistics, convert units, or overwrite existing files.

## Text cache

The index uses schema `phoenix.text_index.v1` and contains:

- `manifest_sha256`: SHA256 of the exact window manifest file.
- `prompt_template`: `A video recorded from a robot's point of view executing the following instruction: {task}`.
- `prompts`: mapping from the complete prompt's UTF-8 SHA256 to `{path, sha256}`.

Cache paths resolve relative to the index file. Each Torch dictionary contains BF16 `context[128,4096]`, boolean `context_mask[128]`, and the matching `prompt` string.

## Visual cache

Set `data.visual_cache_index` or `PHOENIX_VISUAL_CACHE_INDEX` to a completed `phoenix.visual_cache.h32.v1` index. The reader consumes `[96,3,16,48]` BF16 latents, checks per-row hashes, and retains the original image/action padding masks. Source data, frozen weights, preprocessing, current-frame conditioning, and cache fingerprints must match training. Set the option to `null` for online RGB encoding. See [cache generation and sharding](DATA_PREPARATION.md#4-cache-visual-windows).

## Objective and trainable modules

The default prefix sampler is 100% uniform over the inclusive range `[0,max_prefix_steps]`, with `max_prefix_steps=10` and `no_prefix_probability=0.0`. There is no additional forced zero-prefix branch. For a full valid window, each of P0–P10 has probability `1/11`; shorter padded windows bound the range to leave a valid supervised suffix.

Known prefix values are clean actions at timestep zero. Suffix values use the action scheduler's noise and Flow Matching target. Prefix and padding elements are excluded from action supervision.

```text
loss = 0.5 × VAE loss + 0.5 × STVAE loss + Action loss
```

Each branch retains per-sample normalization and scheduler timestep weighting. The VAE and visual encoder are frozen. MoT, embodiment soft prompts, and state encoders follow the model's trainable-module settings. Registry identity is inherited from the selected checkpoint.

The 10-step setting is for inference denoising. Training samples its own diffusion timesteps.

## Configuration and launch

Set the resource variables listed in the [Phoenix example guide](PHOENIX_RTC.md#training), then select a new output directory:

```bash
bash run.sh train --check-config
bash run.sh train output_dir=runs/phoenix_training_rtc
```

`--check-config` resolves configuration, computes asset fingerprints, and reads the manifest/index contracts without constructing the full model or starting optimization.

Use dot-list overrides for batch size, gradient accumulation, learning rate, and other fields:

```bash
NPROC_PER_NODE=8 bash run.sh train \
  training.batch_size=1 training.gradient_accumulation_steps=2 \
  output_dir=runs/phoenix_training_rtc
```

DDP is the default distributed mode. `distributed.deepspeed_zero1=true` selects ZeRO-1 when a compatible DeepSpeed installation is available. Multiple nodes use `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. All nodes access the same assets and output directory.

The batch sampler shards the global order once, drops incomplete accumulation groups, and records consumed local batches. Do not apply another distributed sampler to this loader.

## Checkpoint publication and resume

Checkpoints are written to `checkpoints/step_XXXXXX/` under `output_dir`. They contain inference-compatible `weights.pt`, optimizer and learning-rate states, per-rank RNG states, and `trainer_state.json`. `COMPLETE.json` records the full file set and SHA256 values after state publication. `latest.json` points to a completed checkpoint. Existing checkpoints are not overwritten or rotated away.

```bash
NPROC_PER_NODE=8 bash run.sh train \
  output_dir=runs/phoenix_training_rtc \
  resume_state=/path/to/checkpoint_directory
```

Resume requires matching assets, model definition, RTC settings, training schedule, and world size. It restores optimizer, scheduler, rank RNG, epoch, and consumed batches. Weights-only initialization and full-state resume are separate operations.

For inference, use the completed checkpoint's `weights.pt` and the byte-identical normalization file exported to the run directory. Fill their SHA256 values into `configs/phoenix.yaml` together with the matching model definition and instruction context.
