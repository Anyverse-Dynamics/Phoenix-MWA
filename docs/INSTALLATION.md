# Installation

Use an isolated Python 3.10 environment. Install the accelerator-compatible Torch 2.7.1 / torchvision 0.22.1 pair before resolving project dependencies. On PPU, keep the compatible vendor builds and verify dependency resolution before installation.

Install a C/C++ compiler and Make for native dependencies (Ubuntu:
`sudo apt-get install build-essential`). Robosuite's `egl_probe` dependency may
need a source build. Install CMake 3.x in the active environment before installing
the evaluation extra: `python -m pip install --upgrade pip wheel 'cmake<4'`.
Then build the legacy probe explicitly with
`python -m pip install --no-build-isolation egl_probe==1.0.2`; its isolated build
does not declare the Python CMake wrapper dependency. Only this dependency needs
the build-isolation override.

Install `pip install -e '.[train,logging]'` for training. LIBERO dependencies are available through `pip install -e '.[eval]'`; system EGL/OSMesa and ImageMagick libraries are also required. Do not install LIBERO and RoboCasa simulator dependencies into the same environment.

The LIBERO-Plus integration expects MuJoCo 3.3.2 and Robosuite 1.4.0. Place benchmark assets, BDDL and initialization files in the README's resource layout. No private checkout or machine environment is sourced automatically.

LIBERO-Plus image perturbations require both the Python `Wand` package (included in the `eval` extra) and the system MagickWand library. On Ubuntu 22.04, install `libmagickwand-6.q16-6` with your package manager. Verify in the evaluation environment:

```bash
python -c 'from wand.api import library; assert library.MagickMotionBlurImage is not None'
```

For RoboCasa, install the matching GR1 RoboCasa and Robosuite sources under `resources/robocasa/simulator/`, and prepare its separate interpreter at `resources/robocasa/environments/robocasa-sim/bin/python`, or pass `--sim-python` when preparing evaluation resources. The policy interpreter is independently configurable.

`PPU_SDK_ROOT` is optional and must point to an existing SDK if used. No SDK path is hardcoded. Run preflight and bounded smoke on the target machine before full evaluation.

LIBERO-Pro uses the same project `eval` extra, MuJoCo 3.3.2 and Robosuite 1.4.0 as the validated LIBERO environment. Obtain an external LIBERO-PRO checkout and its official resources following [RESOURCES.md](RESOURCES.md#libero-pro-benchmark). Do not run `pip install -e` on that checkout over LIBERO-Plus: the Pro launcher selects it through an explicit import path in a fresh process. Plus continues to use its bundled integration.

Prepare Pro with `--libero-repo-root` and `--benchmark-root`, then run the public `libero-pro` smoke entry. NVIDIA EGL was validated; no replacement NVIDIA driver libraries are required. PPU needs its own compatible SDK/Torch environment and target-platform smoke. Keep the separate RoboCasa simulator environment unchanged.

## Isolated installation acceptance

An existing environment passing smoke does not prove a fresh installation works.
For a clean policy/LIBERO test, use a Python 3.10 interpreter and do not pass
`--system-site-packages`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel 'cmake<4'
python -m pip install --no-build-isolation egl_probe==1.0.2
# NVIDIA Blackwell example; PPU users must retain the vendor-compatible build.
python -m pip install --no-deps torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# This next command resolves ALL dependencies, including CUDA runtime packages.
python -m pip install -e '.[eval,robocasa-policy]'
env -u PYTHONPATH python -m pip check
python -c 'import torch, torchvision, mujoco, robosuite; print(torch.__version__, torch.cuda.is_available())'
```

Run resource preparation with this environment's Python so `eval.env.sh` records
the new interpreter. Do not reuse an environment file pointing to an old Python.
Run the relevant public smoke commands in [EVALUATION.md](EVALUATION.md) with new output
directories. RoboCasa's simulator must remain in its own environment; reusing
external asset files and weights does not require copying Python packages.

Training dependencies (`.[train,logging]`) are separate from evaluation acceptance.
A policy smoke is not a distributed-training, PPU, or all-Python-version test.
The first Torch command only selects the accelerator wheels; do not stop there
or skip the subsequent complete dependency installation and `pip check`.
Do not install `decord` merely because it existed in an older environment; it is
not a declared dependency of this project.

For an evaluation-only RoboCasa simulator environment, the dependency set is
separate from the policy environment:

```bash
python3.10 -m venv resources/robocasa/environments/robocasa-sim
resources/robocasa/environments/robocasa-sim/bin/python -m pip install \
  -r configs/robocasa-sim-requirements.txt
env -u PYTHONPATH resources/robocasa/environments/robocasa-sim/bin/python -m pip check
```

The launcher imports the simulator checkouts from `ROBOCASA_RESOURCES/simulator`
through `PYTHONPATH`; this minimal evaluation environment does not install the
upstream training extras or a second Torch build. It targets Robosuite 1.5.1
source and the GR1 fork with MuJoCo 3.3.2. Do not apply it to LIBERO, which uses
Robosuite 1.4.0. Source revisions/assets remain separate prerequisites.
Clear `PYTHONPATH` for dependency checks: source-tree package metadata must not be
mistaken for installed policy packages inside the Torch-free simulator environment.
