<h1 align="center">
  <img src="docs/assets/phoenix-mwa-title.png" alt="Phoenix-MWA" width="800">
</h1>

<p align="center"><strong>Phoenix-MWA: Learning Generalizable World–Action Models with Structural and Semantic Representations</strong></p>

<p align="center">
  <a href="https://huggingface.co/collections/OpenPhoenix/phoenix-mwa"><img src="https://img.shields.io/badge/HuggingFace-Model%20%26%20Data-orange?style=flat&amp;logo=huggingface" alt="Model &amp; Data on Hugging Face"></a>
</p>

## Roadmap

- [ ] Data pipeline *(Coming soon)*
- [ ] Pre-training code and model weights *(Coming soon)*
- [x] Post-training code and model weights
- [x] Evaluation on LIBERO-Plus, LIBERO-Pro and RoboCasa GR1
- [ ] Evaluation on RoboTwin 2.0 (c2c & c2r) and RoboDojo (sim & real) *(Coming soon)*

## Overview

![Phoenix-MWA model architecture](docs/assets/main.png)

Phoenix-MWA combines 48-channel Wan video latents and 48-channel CS-VAE latents into a 96-channel visual representation, with canonical 80-dimensional action/state representations and embodiment-conditioned action prediction.

## Model Zoo

| Model | Weights | Benchmark Performance |
| --- | --- | --- |
| Encoder & VAE | [Hugging Face](https://huggingface.co/OpenPhoenix/Phoenix-Encoders) | — |
| LIBERO (Preview) | [Hugging Face](https://huggingface.co/OpenPhoenix/Phoenix-LIBERO) | 94.63% (LIBERO-Plus); 66.48% (LIBERO-Pro) |
| RoboCasa GR1 (Preview) | [Hugging Face](https://huggingface.co/OpenPhoenix/Phoenix-RoboCasa) | 82.67% |
| RoboTwin 2.0 (Preview) | Coming soon... 🏗️ 🚧 🔨 | — |
| RoboDojo-Sim (Preview) | Coming soon... 🏗️ 🚧 🔨 | — |
| RoboDojo-RealWorld (Preview) | Coming soon... 🏗️ 🚧 🔨 | — |

## Getting started

Use Python 3.10 for the validated environment and an accelerator-compatible PyTorch installation. Simulator dependencies should be installed in isolated environments; LIBERO and RoboCasa require different Robosuite versions.

```bash
conda create -n phoenix-mwa python=3.10 -y
conda activate phoenix-mwa
pip install -e '.[train,logging]'
export PYTHONPATH="$PWD/src:$PWD"
```

On PPU, keep the platform's compatible PyTorch build. If SDK activation is needed, set `PPU_SDK_ROOT`; the evaluation launcher sources its `envsetup.sh` in the current shell only. See [environment setup](docs/INSTALLATION.md).

| Directory | Contents |
| --- | --- |
| `src/wam/` | Model, codecs, datasets, trainer and inference |
| `configs/` | Training recipes and portable resource examples |
| `scripts/` | Training and evaluation launchers |
| `eval/` | LIBERO policy and simulator adapters |
| `third_party/` | Benchmark integration code and licenses |
| `assets/` | Normalization statistics and task selection metadata |
| `tools/` | Resource preparation, validation and source packaging |
| `tests/` | Contract and regression tests |

## Prepare data and weights

See [resource acquisition](docs/RESOURCES.md) for official benchmark sources,
directory layouts and the distinction between available resources and pending releases.

| Relative location | Contents |
| --- | --- |
| `checkpoints/` | Wan components, DINO/STVAE encoders and initialization weights |
| `data/libero/` | Four LIBERO LeRobot suites |
| `data/robocasa/` | RoboCasa GR1 LeRobot episodes |
| `data/cache/` | Text and paired visual latent caches |
| `resources/release/` | Policy/config/statistics and encoder bundle |
| `resources/robocasa/` | GR1 simulator sources, assets and environment |
| `third_party/LIBERO-plus/libero/libero/` | LIBERO-Plus assets, BDDL and initialization files |

See [data contracts](docs/DATA.md) and [evaluation preparation](docs/EVALUATION.md). Prepare a raw checkpoint with `tools/prepare_eval.py`: its configuration, normalization statistics and matching encoders are required.

## Training

Run from the repository root. Configure relative paths in `configs/machines/local.example.sh` or override its environment variables. Set `EXPECTED_PRETRAIN_STEP` to the actual step stored in your initialization checkpoint before sourcing the file; no checkpoint step is assumed by default. Public recipes are `libero_posttrain` and `robocasa_posttrain`.

```bash
source configs/machines/local.example.sh
RUN_ID=libero_run NNODES=1 NPROC_PER_NODE=8 EXPECTED_WORLD_SIZE=8 \
  bash scripts/train_libero.sh
```

The default LIBERO recipe mixes all four suites, uses cached visual features and predicts 32 actions per chunk.

For RoboCasa GR1, use the same initialization and encoder variables configured above, and supply the matching numeric/text package and visual cache:

```bash
PYTHON="$(command -v python)" \
ROBOCASA_PACKAGE=resources/robocasa-package \
ROBOCASA_CACHE=data/cache/robocasa \
RUN_ID=robocasa_run NNODES=1 NPROC_PER_NODE=8 EXPECTED_WORLD_SIZE=8 \
  bash scripts/train_robocasa.sh
```

This selects `configs/task/robocasa_posttrain.yaml`. Replace the example resource paths with your prepared resources, and set `WAM_PRETRAIN_CHECKPOINT` and `EXPECTED_PRETRAIN_STEP` to the initialization checkpoint used for this run. Changing topology changes the global batch size. Both launchers perform weights-only initialization and refuse existing output directories. See [training](docs/TRAINING.md) for distributed settings and hyperparameter overrides.

## Evaluation

Prepare resource environment files once using [evaluation preparation](docs/EVALUATION.md), then launch from the repository root.

### LIBERO-Plus

```bash
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/evaluate.sh libero-plus resources/libero/eval.env.sh
```

Full mode covers the 10,030 tasks registered by the bundled benchmark, with one episode per task. Instructions come from BDDL, initial states from the benchmark, and 10 actions are executed per 32-action prediction. Add `smoke` for one task per suite, or `preflight` for resource validation without policy rollouts.

### LIBERO-Pro

Prepare its own resource environment file as described in [LIBERO-Pro preparation](docs/EVALUATION.md#libero-pro), then run:

```bash
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/evaluate.sh libero-pro resources/libero-pro-run/eval.env.sh
```

Use a separate LIBERO-PRO checkout and official four-case resources; see [preparation](docs/EVALUATION.md#libero-pro). Full mode runs 160 tasks × 50 official initial states across Object, Position, Semantic and Task perturbations (8,000 episodes). Add `smoke` for 16 episodes covering all suite/perturbation combinations. The Environment category is not included.

Instructions come from BDDL `:language`; each trial uses its corresponding official initial state. The protocol uses seed 42, 30 settling steps, chunk 32 and replan 10. NVIDIA EGL smoke has passed with a Phoenix-format policy; this validates execution, not a full benchmark score or PPU compatibility. See [validation details](docs/EVALUATION.md#libero-pro-smoke-validation).

### RoboCasa GR1

```bash
NUM_GPUS_TOTAL=8 N_EPISODES=50 \
  bash scripts/evaluate.sh robocasa-gr1 resources/robocasa-eval/eval.env.sh
```

This evaluates 24 GR1 tasks with a 720-step episode cap and 12 executed actions per replan. A saved scene-seed manifest fixes the episode layout protocol. Add `smoke` for one episode of the first task only, or `preflight` for simulator/resource checks. File hashes are not checked. Tensor shapes, action conventions and finite values are checked at runtime.

Use a fresh runtime/output directory for each independent run. Runtime-generated configuration may contain resolved local paths and is excluded from source releases. Weight-dependent smoke validation on the target hardware remains required.

## Acknowledgements

We acknowledge FastWAM, Wan, DINOv3, StarVLA, LeRobot, LIBERO, LIBERO-Plus, LIBERO-Pro and RoboCasa. Original copyright notices are retained. Project code uses `LICENSE`; bundled components retain their own terms, including the DINOv3 License. This is not a blanket MIT license for all dependencies, weights, datasets or robot assets. See `NOTICE` and `LICENSES/`.
