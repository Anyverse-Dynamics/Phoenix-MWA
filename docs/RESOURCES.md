# Resource acquisition

Run commands from the repository root. Keep downloaded resources under ignored
`resources/`, `data/` or `checkpoints/`, never add them with `git add -f`.
Official sources below describe their own data/model/asset terms. Source code,
weights, datasets and simulator assets are separate releases.

## Phoenix policies and encoders

Download links are intentionally blank in the README until publication. Evaluation
requires a matching set, not just a policy tensor file:

- LIBERO: Phoenix-format `policy.pt`, saved configuration, `dataset_stats.json`,
  DINO and CS-VAE/STVAE weights, plus the Wan/text/tokenizer components required
  by that configuration.
- RoboCasa: the same model components, saved configuration, numeric/text package
  (`package.json`, `norm.json` and its referenced files), task metadata and scene
  seed list. Use `--metadata-root` when metadata has been relocated.
- The policy metadata must use `phoenix.local_joint96.weights.v1`; do not rename
  parameter keys or substitute arbitrary encoder checkpoints.

Example layout:

```text
checkpoints/
  libero/{policy.pt,config.yaml,dataset_stats.json}
  robocasa/{policy.pt,config.yaml,metadata/}
  {dino.pt,stvae.pt}
  ... matching Wan/text/tokenizer files ...
resources/
  robocasa-package/
  robocasa/{simulator/,environments/,scene_seed_list.json}
```

Use `tools/prepare_eval.py` as documented in [EVALUATION.md](EVALUATION.md).
It prepares paths/configuration without downloading weights or checking file
hashes. Never replace normalization statistics with values from another model.
Until these model/package links are released, an external user cannot reproduce
the checkpoint evaluations from source alone.

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
  --checkpoint checkpoints/libero/policy.pt --config checkpoints/libero/config.yaml \
  --stats checkpoints/libero/dataset_stats.json \
  --dino checkpoints/dino.pt --stvae checkpoints/stvae.pt --model-base checkpoints \
  --benchmark-root resources/LIBERO-plus/libero/libero --output resources/libero
bash scripts/evaluate.sh libero-plus resources/libero/eval.env.sh smoke
```

Install the project `eval` extra, not a second competing `libero` package. The
launcher imports the bundled integration while resource paths point to the
downloaded benchmark. Follow [INSTALLATION.md](INSTALLATION.md) for system EGL
and MagickWand dependencies. Third-party license completion is deferred and is
not certified by the installation/smoke checks described here.

## LIBERO-Pro benchmark

Obtain the external [LIBERO-PRO code](https://github.com/Zxy-MLlab/LIBERO-PRO) without replacing the bundled Plus integration:

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git resources/LIBERO-PRO
```

Download the official BDDL and initialization resources from [zhouxueyang/LIBERO-Pro](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro). The validated four-case data release is `c86fc3b8293185a6f373677018ff3e37f8391602`. Follow the upstream asset instructions for the simulator meshes/textures. Retain original task names and all 50 initial states per task; do not substitute LIBERO-Plus initial states.

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

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git resources/robocasa/simulator/robosuite
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git resources/robocasa/simulator/robocasa-gr1-tabletop-tasks
```

Follow the GR1 repository's installation instructions in a separate Python 3.10
environment, install those two checkouts there, then run its
`robocasa/scripts/download_tabletop_assets.py -y` from the GR1 checkout. Supply
that environment's interpreter via `--sim-python`. Do not install its Robosuite
over the LIBERO environment. Upstream revisions can change; record your versions
and pass smoke before running the full benchmark.

Original demonstrations are available at
[NVIDIA PhysicalAI Robotics GR00T Teleop Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim).
They are not the ready-to-use Phoenix Delta80 numeric/text cache package. Obtain
the matching release package separately; data conversion is still coming soon.

## Training data and caches

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) provides the original
benchmark and demonstrations. The four LeRobot suite directories and cached
features expected by Phoenix are specified in [DATA.md](DATA.md). Downloading
raw demonstrations alone does not create canonical actions, normalization,
Wan/CS-VAE features, text caches or binding manifests. The full preparation
pipeline is not released yet; do not claim training-from-raw-data reproducibility.
