<h1 align="center">
  <img src="docs/assets/phoenix-mwa-title.png" alt="Phoenix-MWA" width="800">
</h1>

<p align="center"><strong>Phoenix-MWA: Learning Generalizable World–Action Models with Structural and Semantic Representations</strong></p>

<p align="center">
  <a href="https://openphoenix-team.github.io/Phoenix-pages/"><img src="https://img.shields.io/badge/Project%20Page-Blog post-blue?style=flat&amp;logo=github" alt="Project Page"></a>
  <a href="https://huggingface.co/collections/OpenPhoenix/phoenix-mwa"><img src="https://img.shields.io/badge/HuggingFace-Model%20%26%20Data-orange?style=flat&amp;logo=huggingface" alt="Model &amp; Data on Hugging Face"></a>
</p>

<p align="center">
  <a href="#model-zoo">Model Zoo</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="docs/RESOURCES.md">Resources</a>
</p>

This is the **post-train branch of Phoenix-MWA**, providing **post-training code, model weights and benchmark evaluation code**. For world–action pretraining, see the [main branch](https://github.com/OpenPhoenix-Team/Phoenix-MWA/tree/main).

Available and planned releases include:

- [x] LIBERO-Plus, LIBERO-Pro and RoboCasa GR1
- [x] RoboTwin 2.0 C2C & C2R
- [ ] RoboDojo (sim & real) *(Coming soon)*
- [ ] Bench2Dex *(Coming soon)*

## Model Zoo

| Model | Weights | Benchmark Performance |
| --- | --- | --- |
| Encoder & VAE | [Hugging Face](https://huggingface.co/OpenPhoenix/Phoenix-Encoders) | — |
| LIBERO (Preview) | [Hugging Face](https://huggingface.co/OpenPhoenix/Phoenix-LIBERO) | 94.63% (LIBERO-Plus); 66.48% (LIBERO-Pro) |
| RoboCasa GR1 (Preview) | [Hugging Face](https://huggingface.co/OpenPhoenix/Phoenix-RoboCasa) | 82.67% |
| RoboTwin 2.0 (Preview) | [Hugging Face](https://huggingface.co/OpenPhoenix/Phoenix-RoboTwin) | 86.54% (Clean2Clean); 72.42% (Clean2Random) |
| RoboDojo-Sim (Preview) | Coming soon... 🏗️ 🚧 🔨 | — |
| RoboDojo-RealWorld (Preview) | Coming soon... 🏗️ 🚧 🔨 | — |
| Bench2Dex (Preview) | Coming soon... 🏗️ 🚧 🔨 | — |

## Getting started

Use Python 3.10 and an accelerator-compatible PyTorch installation. Simulator dependencies should be installed in isolated environments; LIBERO and RoboCasa require different Robosuite versions.

```bash
conda create -n phoenix-mwa python=3.10 -y
conda activate phoenix-mwa
pip install -e '.[train,logging]'
export PYTHONPATH="$PWD/src:$PWD"
```

On PPU, keep the platform's compatible PyTorch build. If SDK activation is needed, set `PPU_SDK_ROOT`; the evaluation launcher sources its `envsetup.sh` in the current shell only. See [environment setup](docs/INSTALLATION.md).

Use separate simulator environments for RoboCasa and RoboTwin:

```bash
bash scripts/install_simulator_env.sh robocasa resources/robocasa/environments/robocasa-sim
ROBOTWIN_ROOT="$PWD/resources/robotwin/RoboTwin" \
  bash scripts/install_simulator_env.sh robotwin resources/robotwin/sim-env
```

Prepare matching simulator sources/assets first. The RoboTwin installer requires
an NVIDIA CUDA development toolkit and builds CuRobo; it does not install the
host graphics driver. See [system prerequisites and pinned dependencies](docs/INSTALLATION.md).

| Directory | Contents |
| --- | --- |
| `src/wam/` | Model, codecs, datasets, trainer and inference |
| `configs/` | Training recipes, inference configurations and portable resource examples |
| `scripts/` | Training and evaluation launchers |
| `eval/` | Benchmark policy and simulator adapters |
| `third_party/` | Benchmark integration code and licenses |
| `assets/` | Normalization statistics and task selection metadata |
| `tools/` | Evaluation resource preparation |

## Prepare data and weights

See [resource acquisition](docs/RESOURCES.md) for official benchmark sources,
directory layouts and the distinction between available resources and pending releases.


Pretrain weight: [Phoenix-MWA Joint96 (step 256013)](https://www.modelscope.cn/api/v1/models/ssuuuuu/phoenix-wam/repo?Revision=master&FilePath=phoenix-mwa-pretrain-joint96-step_256013.pt)


| Relative location | Contents |
| --- | --- |
| `checkpoints/` | Wan components, policy and initialization weights |
| `resources/encoders/` | Shared `dino110k.pt` and `csvae.pt` encoder weights |
| `data/libero_mujoco3.3.2/` | FastWAM's four preprocessed LIBERO LeRobot v2.1 suites |
| `data/robocasa/` | RoboCasa GR1 LeRobot episodes |
| `data/cache/` | Text and paired visual latent caches |
| `resources/release/` | Policy/config/statistics and encoder bundle |
| `resources/robocasa/` | GR1 simulator sources, assets and environment |
| `data/robotwin/` | RoboTwin LeRobot episodes for online visual encoding |
| `resources/robotwin/` | Simulator, assets and frozen C2C/C2R episode inputs |
| `third_party/LIBERO-plus/libero/libero/` | LIBERO-Plus assets, BDDL and initialization files |

See [data contracts](docs/DATA.md) and [evaluation preparation](docs/EVALUATION.md). Prepare a raw checkpoint with `tools/prepare_eval.py`: its configuration, normalization statistics and matching encoders are required.

For the released LIBERO  and RoboCasa policies,
use [phoenix_libero.yaml](configs/inference/phoenix_libero.yaml) and
[phoenix_robocasa.yaml](configs/inference/phoenix_robocasa.yaml). Their normalization
statistics are included under [`assets/stats/`](assets/stats/); a full training
configuration does not need to be downloaded with these two policies.
RoboCasa evaluation reads its source statistics directly via
`inference.normalization_stats`; no `norm.json` is needed in the resource package.
Its small inference manifest, ordered 24-task list and original 50 scene seeds
are included in [`assets/robocasa/`](assets/robocasa/). Evaluation encodes the
environment's actual instructions with T5 on first use and caches the results
locally; no training package, episode metadata or prebuilt text cache is needed.
Download the common WanVAE/T5/tokenizer components below before evaluation.
RoboTwin also has a source-only [inference config](configs/inference/phoenix_robotwin.yaml)
and [normalization statistics](assets/stats/robotwin.json). Its T5 prompts are
encoded online; no training config, training dataset or training text cache is
required. It additionally needs the exact simulator snapshot and frozen inputs
described in [RoboTwin setup](docs/ROBOTWIN.md).
Simulator sources and assets are separate: see [LIBERO-Plus](docs/RESOURCES.md#libero-plus-benchmark),
[LIBERO-Pro](docs/RESOURCES.md#libero-pro-benchmark) and
[RoboCasa GR1](docs/RESOURCES.md#robocasa-gr1-simulator-and-data).

### WanVAE, T5 and tokenizer

Following [FastWAM's model-directory convention](https://github.com/yuantianyuan01/FastWAM#model-preparation),
use `DIFFSYNTH_MODEL_BASE_PATH` for the common model root. These commands match
Phoenix's released inference configs (`redirect_common_files: true`): download
the converted WanVAE/T5 from **ModelScope**, and the tokenizer from Hugging Face.
The same two weight files and tokenizer directory serve LIBERO, RoboCasa and RoboTwin.

Run in the policy environment, from the Phoenix-MWA repository root:

```bash
set -euo pipefail
python -m pip install 'modelscope>=1.34,<2' 'huggingface-hub>=0.34,<2'
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p "$DIFFSYNTH_MODEL_BASE_PATH"
(
  # Download now; evaluation launchers deliberately run offline.
  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE DIFFSYNTH_SKIP_DOWNLOAD
  modelscope download --model DiffSynth-Studio/Wan-Series-Converted-Safetensors \
    --revision master \
    --include Wan2.2_VAE.safetensors models_t5_umt5-xxl-enc-bf16.safetensors \
    --local_dir "$DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
  hf download Wan-AI/Wan2.1-T2V-1.3B \
    --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
    --include 'google/umt5-xxl/*' \
    --local-dir "$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.1-T2V-1.3B"
)
```

Then pass `--model-base "$DIFFSYNTH_MODEL_BASE_PATH"` to `tools/prepare_eval.py`.
Keep the repository-ID subdirectories; do not flatten the files or rename
official `.pth` files to `.safetensors`. The published LIBERO/RoboCasa/RoboTwin policy
checkpoints already restore both DiTs: their evaluation does **not** require
FastWAM's ActionDiT preprocessing step or a separate download of Wan DiT shards.
See [full directory layout and troubleshooting](docs/RESOURCES.md#wanvae-t5-and-tokenizer).

### LIBERO training data

Use the same preprocessed dataset as [FastWAM](https://github.com/yuantianyuan01/FastWAM):
[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam).
Download the four **LeRobot v2.1** archives generated with MuJoCo **3.3.2**;
the separate `lerobot_v30/` release is not used by this recipe.

Run from the repository root, extracting into a new dataset directory:

```bash
set -euo pipefail
mkdir -p data/libero_mujoco3.3.2
hf download yuanty/LIBERO-fastwam --repo-type dataset \
  libero_10_no_noops_lerobot.tar.gz \
  libero_goal_no_noops_lerobot.tar.gz \
  libero_object_no_noops_lerobot.tar.gz \
  libero_spatial_no_noops_lerobot.tar.gz \
  --local-dir data/libero_mujoco3.3.2

for suite in libero_10 libero_goal libero_object libero_spatial; do
  tar -xzf "data/libero_mujoco3.3.2/${suite}_no_noops_lerobot.tar.gz" \
    -C data/libero_mujoco3.3.2
done
```

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

This is the default `LIBERO_DATA_ROOT`. To reuse an existing copy, set
`LIBERO_DATA_ROOT` to its parent directory before sourcing
`configs/machines/local.example.sh`; no copy or re-download is needed.
These archives contain training data, not T5 or Wan/CS-VAE caches. Prepare the
matching caches separately for the cached training recipe. LIBERO-Plus/Pro
simulation assets and task/initialization files are separate
[evaluation resources](docs/RESOURCES.md#libero-plus-benchmark).

## Training

There are three training recipes for four evaluation benchmarks. LIBERO-Plus and
LIBERO-Pro evaluate the same LIBERO-trained policy; they do not require separate
perturbation-training recipes.

| Training recipe | Evaluation targets | Visual input |
| --- | --- | --- |
| `libero_posttrain` | LIBERO-Plus, LIBERO-Pro | Paired cached features |
| `robocasa_posttrain` | RoboCasa GR1 | Paired cached features |
| `robotwin_posttrain` | RoboTwin C2C and C2R | Online augmented RGB and frozen encoding |

Run from the repository root. Configure relative paths in `configs/machines/local.example.sh` or override its environment variables. Checkpoint steps are read automatically and recorded, never used as compatibility gates. Public recipes are `libero_posttrain`, `robocasa_posttrain`, and `robotwin_posttrain`; RoboTwin has its own [resource example and settings](docs/ROBOTWIN.md).

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

This selects `configs/task/robocasa_posttrain.yaml`. Replace the example resource paths with your prepared resources, and set `WAM_PRETRAIN_CHECKPOINT` to the initialization checkpoint used for this run. Changing topology changes the global batch size. Both launchers perform weights-only initialization and refuse existing output directories. See [training](docs/TRAINING.md) for distributed settings and hyperparameter overrides.

RoboTwin post-training can be launched locally after preparing its resources:

```bash
source configs/machines/robotwin.example.sh
RUN_ID=robotwin_run NNODES=1 NPROC_PER_NODE=8 EXPECTED_WORLD_SIZE=8 \
  bash scripts/train_robotwin.sh
```

See [training commands](docs/TRAINING.md) for all three recipes and
multi-node settings. Use a new run name for each independent training run.

## Evaluation

LIBERO-Plus, LIBERO-Pro and RoboCasa use EGL offscreen rendering. RoboTwin uses
SAPIEN/Vulkan, not EGL. See [renderer setup](docs/INSTALLATION.md#rendering-backends).
`tools/prepare_eval.py` supports the first three; RoboTwin uses its dedicated
resource environment and launcher.

Prepare resource environment files once using [evaluation preparation](docs/EVALUATION.md), then launch from the repository root.

### RoboTwin 2.0

Use the [pinned simulator and local patches](docs/ROBOTWIN.md#exact-simulator-and-assets),
not the latest upstream branch. Source inference settings live in
`configs/inference/phoenix_robotwin.yaml`; normalization is `assets/stats/robotwin.json`.
The policy file is `checkpoints/robotwin/policy.pt`. No separate Wan DiT or
pretraining initialization checkpoint is needed for this inference entry.

#### Required frozen snapshot download

**Download and extract the frozen snapshots before running RoboTwin evaluation.**
They are separate from policy weights and simulator
assets; the evaluation launcher does not download or generate them.
The policy repository and both frozen archives must be accessible before using
the download commands below.

| Mode | Required archive | Download |
| --- | --- | --- |
| C2C | `phoenix_robotwin_c2c_frozen_v1.tar.gz` | [Phoenix-RoboTwin](https://huggingface.co/OpenPhoenix/Phoenix-RoboTwin) |
| C2R | `phoenix_robotwin_c2r_frozen_v1.tar.gz` | [Phoenix-RoboTwin](https://huggingface.co/OpenPhoenix/Phoenix-RoboTwin) |

From the repository root, download both archives and extract their `c2c/` and
`c2r/` directories:

```bash
set -euo pipefail
mkdir -p resources/robotwin/frozen
for mode in c2c c2r; do
  archive="phoenix_robotwin_${mode}_frozen_v1.tar.gz"
  curl --fail --location --retry 3 \
    "https://huggingface.co/OpenPhoenix/Phoenix-RoboTwin/resolve/main/${archive}" \
    --output "resources/robotwin/frozen/${archive}"
  tar -xzf "resources/robotwin/frozen/${archive}" -C resources/robotwin/frozen
done
```

Each mode contains 50 tasks × 100 frozen episodes. See the
[snapshot layout and environment variables](docs/ROBOTWIN.md#required-frozen-snapshot-download)
for the complete package structure and configuration.

#### Launch evaluation

Prepare the pinned simulator/assets, policy/encoder weights and frozen episode
inputs following [RoboTwin setup](docs/ROBOTWIN.md). Then run:

```bash
source configs/machines/robotwin.example.sh
bash scripts/eval_robotwin.sh --mode c2c --gpus 0,1,2,3,4,5,6,7
bash scripts/eval_robotwin.sh --mode c2r --gpus 0,1,2,3,4,5,6,7
```

Each mode covers 50 tasks × 100 frozen episodes, using unseen instructions,
32-action chunks/replanning and 10 denoising steps.
Online-visual post-training is available through `scripts/train_robotwin.sh`;
see [training settings](docs/ROBOTWIN.md#training).

### LIBERO-Plus

```bash
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/evaluate.sh libero-plus resources/libero/eval.env.sh
```

Full mode covers the 10,030 tasks registered by the bundled benchmark, with one episode per task. Instructions come from BDDL, initial states from the benchmark, and 10 actions are executed per 32-action prediction.

### LIBERO-Pro

Prepare its own resource environment file as described in [LIBERO-Pro preparation](docs/EVALUATION.md#libero-pro), then run:

```bash
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/evaluate.sh libero-pro resources/libero-pro-full/eval.env.sh
```

Use a separate LIBERO-PRO checkout and official four-case resources; see [preparation](docs/EVALUATION.md#libero-pro). Full mode runs 160 tasks × 50 official initial states across Object, Position, Semantic and Task perturbations (8,000 episodes). The Environment category is not included.

Instructions come from BDDL `:language`; each trial uses its corresponding official initial state. The protocol uses seed 42, 30 settling steps, chunk 32 and replan 10. See [evaluation details](docs/EVALUATION.md#libero-pro).

### RoboCasa GR1

```bash
NUM_GPUS_TOTAL=8 N_EPISODES=50 \
  bash scripts/evaluate.sh robocasa-gr1 resources/robocasa-eval/eval.env.sh
```

This evaluates 24 GR1 tasks with a 720-step episode cap and 12 executed actions per replan. A saved scene-seed manifest fixes the episode layout protocol. Tensor shapes, action conventions and finite values are checked at runtime.

Use a fresh runtime/output directory for each independent run. Runtime-generated configuration may contain resolved local paths and is excluded from source releases.
