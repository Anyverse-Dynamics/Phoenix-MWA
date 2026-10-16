# Evaluation

All examples are relative to the repository root. Use a policy checkpoint with its saved configuration, normalization statistics and matching frozen encoders. No release manifest, checksum or prior acceptance report is required by the direct preparation entry. Training and evaluation do not verify file hashes. Prompt/task hashes used as cache keys are retained.

## LIBERO-Plus

```bash
python tools/prepare_eval.py --benchmark libero-plus \
  --checkpoint checkpoints/libero/policy.pt --config checkpoints/libero/config.yaml \
  --stats checkpoints/libero/dataset_stats.json \
  --dino checkpoints/dino.pt --stvae checkpoints/stvae.pt \
  --model-base checkpoints --output resources/libero
bash scripts/evaluate.sh libero-plus resources/libero/eval.env.sh preflight
bash scripts/evaluate.sh libero-plus resources/libero/eval.env.sh smoke
```

Create a different runtime directory for the subsequent full evaluation. The launcher refuses to mix new runs with existing launched outputs. `EVAL_GPUS` selects accelerators; `WORKERS_PER_GPU` defaults to 1. `EVAL_SHARDS` and `EVAL_SHARD_RANK` select disjoint partitions.

Full mode uses 10,030 registered tasks; smoke selects one task per suite. Both use one episode per task, the first benchmark-provided initial state, seed 42, 30 settling steps, 10 denoising steps and replan 10 from a 32-action prediction. The cap is 400 steps, or 700 for `libero_10`. BDDL language is authoritative, including language perturbations. Plans, errors and per-task outcomes are recorded.

The public LIBERO-Plus smoke entry audits and builds text caches only for the four selected tasks (the first task in each suite), not the full benchmark. Full evaluation still audits all 10,030 tasks.

## RoboCasa GR1

Install policy transport dependencies in the policy environment with `pip install -e '.[robocasa-policy]'`. Keep the simulator in its separate environment.

```bash
python tools/prepare_eval.py --benchmark robocasa-gr1 \
  --checkpoint checkpoints/robocasa/policy.pt --config checkpoints/robocasa/config.yaml \
  --dino checkpoints/dino.pt --stvae checkpoints/stvae.pt --model-base checkpoints \
  --package resources/robocasa-package --resources resources/robocasa \
  --metadata-root checkpoints/robocasa/metadata \
  --scene-seeds resources/robocasa/scene_seed_list.json --output resources/robocasa-eval
NUM_GPUS_TOTAL=8 bash scripts/evaluate.sh robocasa-gr1 resources/robocasa-eval/eval.env.sh
```

The package must include matching simulator dependencies, numeric/text resources and scene seeds. Prior acceptance reports and file digests are not startup requirements. `N_EPISODES=50`, `MAX_EPISODE_STEPS=720`, `EXECUTE_STEPS=12`, `DENOISING_STEPS=10` are defaults. Scene seeds are verified against the saved manifest during aggregation. The default policy evaluation seed is 7.

Run target-platform smoke before trusting full results. Structural validation does not certify numerical parity or detect same-size file corruption.

```bash
NUM_GPUS_TOTAL=1 bash scripts/evaluate.sh robocasa-gr1 resources/robocasa-eval/eval.env.sh smoke
```

RoboCasa smoke runs the first task (`PnPCupToDrawerClose`) for one episode, with the same 720-step cap, 12-step replan and 10 denoising steps as full evaluation. It checks simulator rendering before loading the policy and writes `merged_summary.json` after the rollout. Use a fresh preparation output directory for each run. A completed smoke validates execution, not benchmark success rate or cross-platform numerical parity.

## LIBERO-Pro

Use an external [LIBERO-PRO checkout](https://github.com/Zxy-MLlab/LIBERO-PRO) and the official [Pro resources](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro). Do not install Pro over the Plus package. Each entry selects its own benchmark in a fresh process.

Prepare `resources/libero-pro-benchmark/` with `bddl_files/`, `init_files/` from the official four-case data and `assets/` from LIBERO-PRO (symlinks are supported). The validated data release is `c86fc3b8293185a6f373677018ff3e37f8391602`; this documents provenance, not a runtime hash check. Keep the checkout and assets under ignored resource directories.

```bash
python tools/prepare_eval.py --benchmark libero-pro \
  --checkpoint checkpoints/libero/policy.pt --config checkpoints/libero/config.yaml \
  --stats checkpoints/libero/dataset_stats.json \
  --dino checkpoints/dino.pt --stvae checkpoints/stvae.pt --model-base checkpoints \
  --libero-repo-root resources/LIBERO-PRO \
  --benchmark-root resources/libero-pro-benchmark --output resources/libero-pro-run
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/evaluate.sh libero-pro resources/libero-pro-run/eval.env.sh smoke
```

Smoke runs the first task and trial in each of 16 settings (four suites × Object/Position/Semantic/Task), with full episode limits.

For full evaluation, prepare a separate runtime directory and launch all 8,000 episodes on eight GPUs:

```bash
python tools/prepare_eval.py --benchmark libero-pro \
  --checkpoint checkpoints/libero/policy.pt --config checkpoints/libero/config.yaml \
  --stats checkpoints/libero/dataset_stats.json \
  --dino checkpoints/dino.pt --stvae checkpoints/stvae.pt --model-base checkpoints \
  --libero-repo-root resources/LIBERO-PRO \
  --benchmark-root resources/libero-pro-benchmark --output resources/libero-pro-full
EVAL_GPUS=0,1,2,3,4,5,6,7 WORKERS_PER_GPU=1 EVAL_SHARDS=1 EVAL_SHARD_RANK=0 \
  bash scripts/evaluate.sh libero-pro resources/libero-pro-full/eval.env.sh full
```

Full mode runs 160 tasks × 50 official initial states = 8,000 episodes. The Environment category is not included in this four-case panel. Do not reuse the smoke output directory. If `resources/libero-pro-full` already exists, choose a fresh path and update both `--output` and the `eval.env.sh` argument. Full results are saved under `resources/libero-pro-full/evaluation/`.

The result index is `official_task_id * 50 + trial`; every trial uses its own official initial state. Instructions come from BDDL `:language`, not filenames. Seed 42, 30 settling steps, chunk 32, replan 10, 10 denoising steps, and 400/700-step limits match the reference protocol. Actions are decoded against the current observation at each simulator step. No videos are saved. `summary_gam.json` only reports `avg_16_settings` when the complete 8,000-episode panel has no technical errors. Smoke completion certifies execution, not a benchmark success rate.

`--libero-repo-root` selects external Pro code; `--benchmark-root` selects the directory containing `assets/`, `bddl_files/` and `init_files/`. Neither changes the bundled Plus package. Use the policy's saved configuration; `config.yaml` is an example filename. See [resource acquisition](RESOURCES.md#libero-pro-benchmark) and [installation](INSTALLATION.md#libero-pro).

Pass `preflight` for resource/contract checks without rollouts. `EVAL_GPUS` selects GPUs and `WORKERS_PER_GPU` defaults to 1. Full runs support `EVAL_SHARDS` and `EVAL_SHARD_RANK`; a shard summary is not a full-panel result. No checkpoint file hashes or historical transfer receipts are required.

### LIBERO-Pro smoke validation

The 2026-10-15 smoke used the migrated Phoenix-format LIBERO PT125k → FT25k policy, DINO110k and STVAE150k encoders, in the isolated Python 3.10 evaluation environment with NVIDIA EGL on eight RTX PRO 5000 GPUs. All 16 complete episodes finished with zero technical errors and eight zero worker exit codes; 10 episodes succeeded. This 10/16 outcome is not a full benchmark score. The combined protocol/regression suite passed 71 tests. PPU execution and full 8,000-episode evaluation were not certified by this smoke.

Outputs are saved in the prepared resource directory's `evaluation/`:

- `preflight.json`: checkpoint/configuration contract checks.
- `instruction_audit.json` and `plan.json`: tasks, instructions and trial indices.
- `worker_*/strict_load.json`, `first_action.json`, `egl_context.json`: model loading, finite `(32, 7)` predictions and rendering evidence.
- `results/<suite>/task_<index>.json`: episode outcome and any technical error.
- `COMPLETE.json`: `passed=true` only after all selected episodes and workers complete without technical errors.
- `summary_gam.json`: full-panel average stays `null` for smoke or incomplete results.

Keep outputs under ignored `resources/` or `validation_runs/`. Prepare a fresh output directory for each independent run; a launched directory cannot be reused.

## Checkpoint preparation

Retain each checkpoint's saved config, numerical normalization statistics and encoder weights. The direct preparer reads the checkpoint step and configuration, writes a portable runtime configuration, and launches no inference. It accepts relative resource paths. If an existing RoboCasa package references relocated dataset metadata, supply `ROBOCASA_PATH_MAP` for those paths. Public source statistics have sanitized path metadata; use the numerical statistics belonging to the selected checkpoint.

`tools/prepare_release_eval.py` remains available for existing packaged releases, without file hash verification. Export/packaging utilities are separate from runtime.
