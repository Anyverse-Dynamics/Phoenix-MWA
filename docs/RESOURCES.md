# Resource acquisition

Run commands from the repository root. Keep downloaded resources under ignored
`resources/`, `data/` or `checkpoints/`, never add them with `git add -f`.
Official sources below describe their own data/model/asset terms. Source code,
weights, datasets and simulator assets are separate releases.

## Phoenix policies and encoders

Published model links are listed in the README Model Zoo; entries marked coming
soon are not yet supplied there. Evaluation
requires a matching set, not just a policy tensor file:

- LIBERO: `policy.pt`, `configs/inference/phoenix_libero.yaml`,
  `assets/stats/phoenix_libero.json`,
  DINO and CS-VAE weights, plus the Wan/text/tokenizer components required
  by that configuration.
- RoboCasa: the same model components, `configs/inference/phoenix_robocasa.yaml`,
  source statistics `assets/stats/phoenix_robocasa.json`, and the included
  `assets/robocasa/{package.json,tasks.json,scene_seed_list.json}`. T5 encodes
  actual simulator instructions online and caches them in the runtime directory.
  Evaluation does not need a training package, episode metadata, precomputed
  text embeddings/manifest or package `norm.json`. Matching simulator sources
  and assets are still external resources.
- Use the configuration, normalization statistics and encoders supplied for the
  selected policy.

The shared encoder filenames are `dino110k.pt` and `csvae.pt`. Download them to
the paths used by all three inference configurations:

```bash
hf download OpenPhoenix/Phoenix-Encoders dino110k.pt csvae.pt \
  --local-dir resources/encoders
```

The CSVAE payload uses `model_type: causal_semantic_vae` and `format_version: 2`.
The version describes checkpoint serialization; it does not identify a new model
architecture. Both Phoenix pretraining and Phoenix-MWA use the policy weight
schema `phoenix.local_joint96.weights.v1`, with `csvae` metadata and the bundled
source URI `bundled://dino-csvae-v2`. Use the matching released policy and encoder
files; changing a filename alone does not convert an earlier payload.
The inference launchers check model semantics and tensor structure, while runtime
file hashes and fixed encoder steps remain disabled.

Example layout:

```text
configs/inference/
  phoenix_libero.yaml
  phoenix_robocasa.yaml
assets/stats/
  phoenix_libero.json
  phoenix_robocasa.json
assets/robocasa/
  package.json
  tasks.json
  scene_seed_list.json
checkpoints/
  libero/policy.pt
  robocasa/policy.pt
  DiffSynth-Studio/Wan-Series-Converted-Safetensors/
    Wan2.2_VAE.safetensors
    models_t5_umt5-xxl-enc-bf16.safetensors
  Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
    special_tokens_map.json
    spiece.model
    tokenizer.json
    tokenizer_config.json
resources/
  encoders/{dino110k.pt,csvae.pt}
  robocasa/{simulator/,environments/}
```

Use `tools/prepare_eval.py` as documented in [EVALUATION.md](EVALUATION.md).
It prepares paths/configuration without downloading weights or checking file
hashes. Never replace normalization statistics with values from another model.
Check availability of every required package, not only the policy weights;
source code alone does not contain encoder weights, training caches or assets.

## WanVAE, T5 and tokenizer

Use the [download commands in the README](../README.md#wanvae-t5-and-tokenizer)
before running evaluation. They follow [FastWAM's model-root convention](https://github.com/yuantianyuan01/FastWAM#model-preparation)
but are adapted to Phoenix's full policy restore and offline launchers.
Do not copy FastWAM's ActionDiT-backbone preprocessing command into the released
Phoenix LIBERO/RoboCasa/RoboTwin evaluation workflow: both DiTs come from `policy.pt`.
This statement applies to those evaluation entries, not arbitrary from-scratch
training or other model configurations.

`configs/inference/phoenix_libero.yaml`, `phoenix_robocasa.yaml` and `phoenix_robotwin.yaml` use
`model_id: Wan-AI/Wan2.2-TI2V-5B`,
`tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B` and
`redirect_common_files: true`. The latter redirects only the VAE and T5 weights
to DiffSynth's converted files:

| Component | Download source | Exact selection | Approximate download |
| --- | --- | --- | --- |
| WanVAE (48-channel Wan2.2 VAE) | [ModelScope: DiffSynth-Studio/Wan-Series-Converted-Safetensors](https://modelscope.cn/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/files) | `Wan2.2_VAE.safetensors` | 1.41 GB |
| T5 text encoder | Same ModelScope repository | `models_t5_umt5-xxl-enc-bf16.safetensors` | 11.36 GB |
| Tokenizer | [Hugging Face: Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main/google/umt5-xxl) | Entire `google/umt5-xxl/` directory | Small relative to weights |

Only the tokenizer is taken from the Wan2.1 repository; do **not** download its
`Wan2.1_VAE.pth` as a replacement for the 48-channel Wan2.2 VAE. The converted
repository is available on ModelScope; an identically named Hugging Face
repository is not assumed to exist. The ModelScope CLI uses branch `master`;
the inspected snapshot was `150f75d811d51f6c7760154aa7fec371dccda529`. The
tokenizer command pins Hugging Face revision
`37ec512624d61f7aa208f7ea8140a131f93afc9a`. Keep download metadata when archiving
resources for reproducibility. These commands select files, not whole video
generation models, and download about 12.8 GB plus the tokenizer.

The directory tree above is the expected final layout. For example, with
`DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"`, the T5 path must be
`checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors`.
`--model-base` takes the **parent of the repository-ID directories**, not the
T5 file or the `DiffSynth-Studio/` directory. For an existing compatible model
root, reuse it without copying and pass its absolute path:

```bash
# Set this to your existing common model root if it is not ./checkpoints.
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
# Add this option to the benchmark-specific prepare_eval.py command:
# --model-base "$DIFFSYNTH_MODEL_BASE_PATH"
```

The preparer records that root in `eval.env.sh`. If you move resources, prepare
a new runtime directory instead of reusing an environment file containing the
old paths. Configs with `redirect_common_files: false` look for official `.pth`
files under `Wan-AI/Wan2.2-TI2V-5B` instead and are **not** the configuration
covered by these commands; do not change that flag just to suppress a missing
file error.

### Troubleshooting

- `No model files match ...`: check `--model-base`, the exact filenames and
  repository-ID subdirectories. Download in a network-enabled shell first;
  evaluation deliberately sets `DIFFSYNTH_SKIP_DOWNLOAD=true` and `HF_HUB_OFFLINE=1`.
- `modelscope` or `hf` not found: run the README's pip command inside the active
  policy environment and ensure its `bin` directory is on `PATH`.
- Tokenizer loading fails: download all four tokenizer files, preserving
  `google/umt5-xxl/`; do not point at an HF cache's parent instead of a local
  model directory.
- T5 weights versus text cache: the downloaded encoder weights are required to
  compute new prompts. RoboCasa builds its local text embeddings at inference
  time; those runtime caches need not be downloaded or committed to source.

## LIBERO-Plus benchmark

Official [code and installation](https://github.com/sylvestf/LIBERO-plus) and
[benchmark assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus) provide the
benchmark resources. Keep the bundled integration code unchanged: obtain a
separate upstream checkout for resources instead of cloning over it.

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git resources/LIBERO-plus
```

Download `assets.zip` from the official asset page and extract it so the directory
is `resources/LIBERO-plus/libero/libero/assets/`. BDDL, initialization files and
task classification come from the benchmark checkout. Prepare evaluation with:

```bash
python tools/prepare_eval.py --benchmark libero-plus \
  --checkpoint checkpoints/libero/policy.pt --config configs/inference/phoenix_libero.yaml \
  --stats assets/stats/phoenix_libero.json \
  --dino resources/encoders/dino110k.pt --csvae resources/encoders/csvae.pt --model-base checkpoints \
  --benchmark-root resources/LIBERO-plus/libero/libero --output resources/libero
EVAL_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/evaluate.sh libero-plus resources/libero/eval.env.sh full
```

Install the project `eval` extra, not a second competing `libero` package. The
launcher imports the bundled integration while resource paths point to the
downloaded benchmark. Follow [INSTALLATION.md](INSTALLATION.md) for system EGL
and MagickWand dependencies. Retain the upstream license notices
for the benchmark and downloaded assets.

## LIBERO-Pro benchmark

Obtain the external [LIBERO-PRO code](https://github.com/Zxy-MLlab/LIBERO-PRO) without replacing the bundled Plus integration:

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git resources/LIBERO-PRO
```

Download the official BDDL and initialization resources from [zhouxueyang/LIBERO-Pro](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro). The four-case data revision is `c86fc3b8293185a6f373677018ff3e37f8391602`. Follow the upstream asset instructions for the simulator meshes/textures. Retain original task names and all 50 initial states per task; do not substitute LIBERO-Plus initial states.

Arrange the resources as follows (directory symlinks are supported):

```text
resources/
  LIBERO-PRO/libero/libero/       # external benchmark code
  libero-pro-benchmark/
    assets/                     # Pro simulator assets
    bddl_files/                 # official four-case BDDL directories
    init_files/                 # corresponding official initial states
```

Use `--libero-repo-root resources/LIBERO-PRO` and `--benchmark-root resources/libero-pro-benchmark` with `tools/prepare_eval.py --benchmark libero-pro`. Supply the same matching Phoenix LIBERO policy/configuration/statistics/encoders as for Plus. See [complete preparation and launch commands](EVALUATION.md#libero-pro).

The implemented panel covers Object, Position, Semantic and Task perturbations, not Environment: four suites × four cases × ten tasks × fifty trials = 8,000 episodes. Runtime checks structure and numeric validity, not file hashes. Resources and generated environment files stay in ignored directories.

## RoboCasa GR1 simulator and data

Use the official [GR1 tabletop simulator](https://github.com/robocasa/robocasa-gr1-tabletop-tasks)
and its asset installer; generic RoboCasa is not a substitute for the GR1 fork.
For a fresh checkout, run from the Phoenix-MWA root. Reuse existing matching
source/assets instead of cloning over an existing directory.

```bash
set -euo pipefail
mkdir -p resources/robocasa/simulator
git clone --branch v1.5.1 --depth 1 https://github.com/ARISE-Initiative/robosuite.git \
  resources/robocasa/simulator/robosuite
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git resources/robocasa/simulator/robocasa-gr1-tabletop-tasks
bash scripts/install_simulator_env.sh robocasa resources/robocasa/environments/robocasa-sim
ROBOCASA_SIM_PYTHON="$PWD/resources/robocasa/environments/robocasa-sim/bin/python"
(
  cd resources/robocasa/simulator/robocasa-gr1-tabletop-tasks
  "$ROBOCASA_SIM_PYTHON" robocasa/scripts/download_tabletop_assets.py -y
)
```

Robosuite is pinned to tag `v1.5.1`, not an arbitrary newer checkout. The GR1
clone above uses its upstream default branch; for exact historical comparisons,
reuse the original GR1 revision and asset snapshot. A new clone plus identical
scene seeds does not certify the same benchmark build.
The launcher imports these source trees explicitly; do not install their full
training extras over the minimal simulator requirements. The official asset
installer places meshes/textures under the GR1 checkout's `robocasa/models/assets/`;
follow its download/license prompts. When preparing evaluation, use
`--resources resources/robocasa` and
`--sim-python resources/robocasa/environments/robocasa-sim/bin/python`.
Do not install its Robosuite over the LIBERO environment. Match source revisions
and assets to the selected benchmark release.

Original demonstrations are available at
[NVIDIA PhysicalAI Robotics GR00T Teleop Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim).
They are not the ready-to-use Phoenix Delta80 numeric/text cache package. For
training, obtain the matching package separately; data conversion is still
coming soon. Simulation evaluation needs neither those demonstrations nor the
training package.

### What the RoboCasa inference metadata contains

- `package.json` is a small inference manifest, not a model or a dataset. It
  declares the exact action/frame/controller/embodiment and text conventions.
  It is intentionally a different schema from the training package, which also
  indexes episode splits, cached features and dataset file locations.
- `tasks.json` is the ordered 24-task evaluation panel and simulator-name
  template. Original training `tasks.jsonl`, `modality.json` and
  `episodes.jsonl` are not required. The inference adapter supplies the numeric
  field mapping; instructions come from the live simulator.
- `scene_seed_list.json` preserves the original 50 seeds, reused per task and
  checkpoint. It fixes randomized environment resets, separate from policy
  diffusion seed 7. It is not a trajectory or an image dataset.

These small files ship with source; do not copy the historical training package
or its text tensors into Git. See [online text caching](EVALUATION.md#robocasa-gr1).

## Training data and caches

RoboTwin resources are separate from LIBERO and RoboCasa; see the dedicated
section below. Do not substitute a fresh random seed list for frozen evaluation
inputs when reproducing the C2C/C2R protocol.

### LIBERO training dataset

Use the same preprocessed data as FastWAM:
[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam).
Select the four LeRobot **v2.1** archives (`libero_10`, `libero_goal`,
`libero_object`, `libero_spatial`, each ending in `_no_noops_lerobot.tar.gz`).
This release was generated with MuJoCo **3.3.2**; the separate `lerobot_v30/`
release is not selected by the current recipe.

Follow the [download and extraction commands](../README.md#libero-training-data)
to populate `data/libero_mujoco3.3.2/`. Set `LIBERO_DATA_ROOT` to another existing
parent directory if needed. No raw-HDF5-to-LeRobot conversion is required when
using these preprocessed archives. Retain FastWAM and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) attribution and the
dataset's CC BY 4.0 terms.

The dataset is distinct from the T5 text cache, paired Wan/CS-VAE visual caches
and binding manifests required by Phoenix's cached recipe. These resources
must match the selected data and encoders; the dataset download does not create
them. The end-to-end cache-preparation workflow is not yet fully released.
See [DATA.md](DATA.md) and `configs/machines/local.example.sh` for the separate
resource variables. Plus/Pro task definitions, initialization states and
simulator assets are evaluation resources, not additional training datasets.

## RoboTwin training and evaluation resources

Configure `configs/machines/robotwin.example.sh` from the repository root. Example
locations below are placeholders for matching resources, not bundled downloads:

| Variable | Relative location / contents |
| --- | --- |
| `ROBOTWIN_DATA_ROOT` | Training only: `data/robotwin`, LeRobot metadata, numeric records and three-camera videos |
| `ROBOTWIN_TEXT_CACHE` | Training only: `data/cache/robotwin-text`, task-indexed T5 cache |
| `ROBOTWIN_ROOT` | `resources/robotwin/RoboTwin`: matching simulator source and licensed assets |
| `ROBOTWIN_SIM_PYTHON` | `resources/robotwin/sim-env/bin/python`: separate SAPIEN/CuRobo environment |
| `ROBOTWIN_FROZEN_C2C` | `resources/robotwin/frozen/c2c`: clean frozen episodes |
| `ROBOTWIN_FROZEN_C2R` | `resources/robotwin/frozen/c2r`: randomized frozen episodes |
| `ROBOTWIN_CHECKPOINT` | `checkpoints/robotwin/policy.pt` |
| `ROBOTWIN_INFERENCE_CONFIG` | Source: `configs/inference/phoenix_robotwin.yaml` |
| `ROBOTWIN_EVAL_STATS` | Source: `assets/stats/robotwin.json` |

Also supply the policy-side WanVAE/T5/tokenizer components and matching frozen
DINO/CSVAE weights. Evaluation strictly restores both DiTs from the complete
policy; it does not load a pretraining checkpoint or original Wan DiT shards.
Online T5 needs no prebuilt training cache. `ROBOTWIN_TRAIN_CONFIG` remains a
legacy fallback when no new inference config is specified.
Training uses online visual encoding, not a visual feature
cache. Evaluation needs the frozen `SPEC.json`, `CANDIDATES.json` and referenced
scene/observation records for both modes; a seed-only list is insufficient.
The public source does not automatically obtain these external resources. Model
Zoo availability does not imply simulator assets or frozen inputs are included.
See [ROBOTWIN.md](ROBOTWIN.md) for contracts and launch commands, and
[INSTALLATION.md](INSTALLATION.md#robotwin-environment) for rendering requirements.

Install the simulator with `scripts/install_simulator_env.sh robotwin` only after
the matching source release (including `envs/curobo`) and assets are present.
Use the same simulator release as the frozen input package; selecting a new
upstream branch can change scene construction even with identical episode seeds.
Extract the clean frozen archives into `resources/robotwin/frozen/` so each mode
contains its own `SPEC.json`. The simulator, assets, policy and frozen input
archives are distinct downloads with distinct redistribution terms.
See [exact simulator and assets](ROBOTWIN.md#exact-simulator-and-assets) for pinned
download commands and the local patches preserved from the evaluation snapshot.
Minimal project-hosted release files are `policy.pt`, both frozen archives and
their manifest; the inference config and normalization belong in source, and
the two encoders are reused from `OpenPhoenix/Phoenix-Encoders`. The frozen
archives can instead be hosted in a separate HF dataset, but cannot be omitted
from reproduction of this protocol. Obtain access to both archives before
launching evaluation.
