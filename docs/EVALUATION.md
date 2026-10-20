# Evaluation

Install the policy/LIBERO environment and the separate RoboCasa/RoboTwin simulator
environments using [INSTALLATION.md](INSTALLATION.md). Prepare resource files with
the policy environment's Python; set `--sim-python` for RoboCasa and
`ROBOTWIN_SIM_PYTHON` for RoboTwin to their own venv launchers. Do not resolve those
Python symlinks to base interpreters. Reusing benchmark assets or checkpoints is
independent of installing fresh Python dependencies.

## Checkpoint compatibility

LIBERO-Plus, LIBERO-Pro, RoboCasa and RoboTwin read policy, initialization and encoder progress
from checkpoint metadata for reporting only. No fixed training step is required;
missing progress is reported as unknown. Legacy expected-step options are ignored.
Model schema, tensor names/shapes, action space, normalization, camera ordering,
coordinate transforms and embodiment prompt semantics remain checked. Changing
a checkpoint's recorded step does not change its inference configuration.

For LIBERO-Plus, select the actual file with `JOINT96_EVAL_CHECKPOINT_FILENAME`
(default `policy.pt`, otherwise a unique `.pt` in the resource directory).
Multiple candidates require an explicit filename; filenames are not parsed as
training steps. Training duration, simulator episode limits and optimizer-state
consistency checks are unchanged.

All examples are relative to the repository root. Use a policy checkpoint with its matching inference configuration, normalization statistics and frozen encoders.

The source includes configurations for the two released policies:

| Policy | Inference configuration | Normalization statistics |
| --- | --- | --- |
| LIBERO | `configs/inference/phoenix_libero.yaml` | `assets/stats/phoenix_libero.json` |
| RoboCasa | `configs/inference/phoenix_robocasa.yaml` | `assets/stats/phoenix_robocasa.json` |

These are standalone OmegaConf files, not Hydra training recipes. They retain
model, processor and compatibility fields, but no optimizer, training dataset or
visual-cache paths. `data.train/val` are retained for the existing contract checks;
evaluation does not instantiate those datasets. Resource flags passed to
`prepare_eval.py` relocate the model's encoder and registry paths. For other
checkpoints, supply their own matching configuration and statistics.

Before preparing either benchmark, [download WanVAE, T5 and the tokenizer](../README.md#wanvae-t5-and-tokenizer)
following the [required directory layout](RESOURCES.md#wanvae-t5-and-tokenizer).
The commands below assume their repository-ID directories are under
`checkpoints/`. If using another root, replace each `--model-base checkpoints`
with `--model-base "$DIFFSYNTH_MODEL_BASE_PATH"`; the preparer records the absolute
root in `eval.env.sh`. The launchers do not download missing models.

## LIBERO-Plus

```bash
python tools/prepare_eval.py --benchmark libero-plus \
  --checkpoint checkpoints/libero/policy.pt --config configs/inference/phoenix_libero.yaml \
  --stats assets/stats/phoenix_libero.json \
  --dino resources/encoders/dino110k.pt --csvae resources/encoders/csvae.pt \
  --model-base checkpoints --benchmark-root resources/LIBERO-plus/libero/libero \
  --output resources/libero
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/evaluate.sh libero-plus resources/libero/eval.env.sh full
```

Use a fresh runtime directory for each evaluation run. The launcher refuses to mix new runs with existing launched outputs. `EVAL_GPUS` selects accelerators; `WORKERS_PER_GPU` defaults to 1. `EVAL_SHARDS` and `EVAL_SHARD_RANK` select disjoint partitions.

Full evaluation uses 10,030 registered tasks with one episode per task, the first benchmark-provided initial state, seed 42, 30 settling steps, 10 denoising steps and replan 10 from a 32-action prediction. The cap is 400 steps, or 700 for `libero_10`. BDDL language is authoritative, including language perturbations. Plans, errors and per-task outcomes are recorded.

## RoboCasa GR1

Install policy transport dependencies in the policy environment with `pip install -e '.[robocasa-policy]'`. Keep the simulator in its separate environment.

RoboCasa evaluation reads `inference.normalization_stats` directly from its
configuration, defaulting to the source file `assets/stats/phoenix_robocasa.json`.
Config-relative paths are resolved against the repository root. The preparer
records an absolute path; `--stats PATH` can explicitly select matching statistics
for another policy (CLI-relative paths use the current working directory).
The package's `norm.json` is neither required nor read, even if present. A missing
configured statistics file fails without falling back to the package. Older
configs without this field also use the source asset; use `--stats` for a policy
with different statistics. The small inference manifest, task list and original
50 scene seeds are in `assets/robocasa/`; training still uses its own full package
and statistics. Evaluation no longer reads training episode metadata,
`text/COMPLETE.json` or prebuilt T5 embeddings.

```bash
python tools/prepare_eval.py --benchmark robocasa-gr1 \
  --checkpoint checkpoints/robocasa/policy.pt --config configs/inference/phoenix_robocasa.yaml \
  --dino resources/encoders/dino110k.pt --csvae resources/encoders/csvae.pt --model-base checkpoints \
  --resources resources/robocasa \
  --sim-python resources/robocasa/environments/robocasa-sim/bin/python \
  --output resources/robocasa-eval
NUM_GPUS_TOTAL=8 N_EPISODES=50 \
  bash scripts/evaluate.sh robocasa-gr1 resources/robocasa-eval/eval.env.sh full
```

The default `--package` is `assets/robocasa` and the default `--scene-seeds` is
`assets/robocasa/scene_seed_list.json`. The full launcher reads the ordered 24
tasks from `assets/robocasa/tasks.json`. Each task uses the same saved 50 episode
seeds (1,200 episodes total). Seeds fix randomized
initial scenes, not policy noise, and do not guarantee cross-platform pixel
identity. The task manifest selects simulator environments; actual instructions
always come from `annotation.human.coarse_action`, not a fixed manifest string.

T5 runs online when an exact prompt is first encountered, with the existing
128-token BF16 convention (zero padded context, full attention mask). Subsequent
uses read memory or the prepared runtime's `text_cache/`. Direct policy loading
defaults to `.cache/robocasa_t5`; configure `inference.text_cache_dir: null` for
memory-only caching. The cache keys include prompt, encoder/tokenizer file
identity, encoding convention and platform. File identities use paths, sizes
and timestamps, not content integrity hashes. Cache entries are written atomically
under a per-key lock; corrupt entries fail visibly. Caches are local runtime
artifacts, not public source. Online encoding means local model computation,
not an online API: download matching T5/tokenizer weights before offline runs.
The first new instruction takes extra time and T5 adds device-memory overhead;
check the target hardware before choosing workers per GPU.

`N_EPISODES=50`, `MAX_EPISODE_STEPS=720`, `EXECUTE_STEPS=12`,
`DENOISING_STEPS=10` are defaults. Scene seeds are verified against the saved
manifest during aggregation. The separate policy evaluation seed is 7.

## LIBERO-Pro

Use an external [LIBERO-PRO checkout](https://github.com/Zxy-MLlab/LIBERO-PRO) and the official [Pro resources](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro). Do not install Pro over the Plus package. Each entry selects its own benchmark in a fresh process.

Prepare `resources/libero-pro-benchmark/` with `bddl_files/`, `init_files/` from the official four-case data and `assets/` from LIBERO-PRO (symlinks are supported). The four-case data revision is `c86fc3b8293185a6f373677018ff3e37f8391602`; use matching simulator assets. Keep the checkout and assets under ignored resource directories.

Prepare a runtime directory and launch all 8,000 episodes on eight GPUs:

```bash
python tools/prepare_eval.py --benchmark libero-pro \
  --checkpoint checkpoints/libero/policy.pt --config configs/inference/phoenix_libero.yaml \
  --stats assets/stats/phoenix_libero.json \
  --dino resources/encoders/dino110k.pt --csvae resources/encoders/csvae.pt --model-base checkpoints \
  --libero-repo-root resources/LIBERO-PRO \
  --benchmark-root resources/libero-pro-benchmark --output resources/libero-pro-full
EVAL_GPUS=0,1,2,3,4,5,6,7 WORKERS_PER_GPU=1 EVAL_SHARDS=1 EVAL_SHARD_RANK=0 \
  bash scripts/evaluate.sh libero-pro resources/libero-pro-full/eval.env.sh full
```

Full mode runs 160 tasks × 50 official initial states = 8,000 episodes. The Environment category is not included in this four-case panel. If `resources/libero-pro-full` already exists, choose a fresh path and update both `--output` and the `eval.env.sh` argument. Full results are saved under `resources/libero-pro-full/evaluation/`.

The result index is `official_task_id * 50 + trial`; every trial uses its own official initial state. Instructions come from BDDL `:language`, not filenames. Seed 42, 30 settling steps, chunk 32, replan 10, 10 denoising steps, and 400/700-step limits match the reference protocol. Actions are decoded against the current observation at each simulator step. No videos are saved. `summary_gam.json` only reports `avg_16_settings` when the complete 8,000-episode panel has no technical errors.

`--libero-repo-root` selects external Pro code; `--benchmark-root` selects the directory containing `assets/`, `bddl_files/` and `init_files/`. Neither changes the bundled Plus package. The example uses the released LIBERO policy's source inference configuration. See [resource acquisition](RESOURCES.md#libero-pro-benchmark) and [installation](INSTALLATION.md#libero-pro).

`EVAL_GPUS` selects GPUs and `WORKERS_PER_GPU` defaults to 1. Full runs support `EVAL_SHARDS` and `EVAL_SHARD_RANK`; a shard summary is not a full-panel result.

### Output files

Outputs are saved in the prepared resource directory's `evaluation/`:

- `plan.json`: tasks, instructions and trial indices.
- `results/<suite>/task_<index>.json`: episode outcome and any technical error.
- `COMPLETE.json`: `passed=true` only after all selected episodes and workers complete without technical errors.
- `summary_gam.json`: full-panel average stays `null` for incomplete results.

Keep outputs under ignored `resources/` or `runs/`. Prepare a fresh output directory for each independent run; a launched directory cannot be reused.

## Checkpoint preparation

RoboTwin uses the dedicated resource configuration below rather than
`tools/prepare_eval.py`. The latter supports LIBERO-Plus, LIBERO-Pro and RoboCasa.

Retain each checkpoint's matching inference config, numerical normalization statistics and encoder weights. The direct preparer accepts either the source inference configs above or an existing saved training config, reads checkpoint progress, and writes a relocated runtime configuration without launching inference. Its output may still be named `train_config.yaml` for compatibility with the LIBERO loader; this does not require distributing the original training recipe. It accepts relative resource paths. RoboCasa inference does not read training dataset metadata or require `ROBOCASA_PATH_MAP`; those remain training-only resources. Public source statistics have sanitized path metadata; use the numerical statistics belonging to the selected checkpoint.

`tools/prepare_release_eval.py` prepares evaluation resources from existing packaged releases.

## RoboTwin C2C and C2R

Configure the policy/encoder weights, pinned simulator and frozen inputs using
[ROBOTWIN.md](ROBOTWIN.md) and its resource example. The default inference config
is `configs/inference/phoenix_robotwin.yaml`, and normalization is read directly
from `assets/stats/robotwin.json`. T5 encodes the exact frozen instruction online
and reuses it within the session. Training configs, datasets, training T5 caches,
initialization checkpoints and original Wan DiT shards are not inference inputs.

```bash
source configs/machines/robotwin.example.sh
# Independent full runs, each with a fresh automatically generated run ID.
bash scripts/eval_robotwin.sh --mode c2c --gpus 0,1,2,3,4,5,6,7
bash scripts/eval_robotwin.sh --mode c2r --gpus 0,1,2,3,4,5,6,7
```

Each mode contains 50 tasks × 100 frozen episodes (5,000 episodes). C2C uses
`demo_clean`, C2R `demo_randomized`; the frozen records specify scene seeds,
episode indices and unseen instructions. There is no expert replay or replacement
of failed episodes. Replan/chunk are 32/32, denoising steps 10, policy seed 0.
Native actions have 14 dimensions; the simulator keeps each task's original cap.

One policy server runs per selected GPU; tasks are distributed round-robin.
Scene numeric values and joints must match within 1e-6; RGB error is diagnostic
because driver/hardware changes can affect pixels. Technical errors are not
counted as policy failures. Rendering uses SAPIEN/Vulkan, not MuJoCo EGL.
Outputs default to ignored `runs/evaluation/robotwin/`; existing output directories
are refused. `--run-id` can specify a new name. See [environment setup](INSTALLATION.md#robotwin-environment).
`--inference-config` / `ROBOTWIN_INFERENCE_CONFIG` and `--dataset-stats` /
`ROBOTWIN_EVAL_STATS` allow explicit overrides. Old `ROBOTWIN_TRAIN_CONFIG` and
the shared server's `--training-config` remain compatibility aliases, not release
requirements. The new config takes precedence over the legacy environment name;
CLI paths take precedence over environment values.
