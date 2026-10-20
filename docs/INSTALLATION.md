# Installation

Use an isolated Python 3.10 environment. Install the accelerator-compatible Torch 2.7.1 / torchvision 0.22.1 pair before resolving project dependencies. On PPU, keep the compatible vendor builds.

Install a C/C++ compiler and Make for native dependencies (Ubuntu:
`sudo apt-get install build-essential`). Robosuite's `egl_probe` dependency may
need a source build. Install CMake 3.x in the active environment before installing
the evaluation extra: `python -m pip install --upgrade pip wheel 'cmake<4'`.
Then build the legacy probe explicitly with
`python -m pip install --no-build-isolation egl_probe==1.0.2`; its isolated build
does not declare the Python CMake wrapper dependency. Only this dependency needs
the build-isolation override.

For the MuJoCo benchmarks on Ubuntu, install headless rendering/image libraries:

```bash
sudo apt-get update
sudo apt-get install build-essential libegl1 libgl1 libgles2 libosmesa6 \
  libglib2.0-0 libmagickwand-6.q16-6 ffmpeg
```

NVIDIA EGL also requires the host's driver-matched graphics libraries exposed to
the container. System packages and Python environments serve different purposes;
creating a venv does not install a driver.

Install `pip install -e '.[train,logging]'` for training. LIBERO dependencies are available through `pip install -e '.[eval]'`; system EGL/OSMesa and ImageMagick libraries are also required. Do not install LIBERO and RoboCasa simulator dependencies into the same environment.

The LIBERO-Plus integration expects MuJoCo 3.3.2 and Robosuite 1.4.0. Place benchmark assets, BDDL and initialization files in the README's resource layout. No private checkout or machine environment is sourced automatically.

LIBERO-Plus image perturbations require both the Python `Wand` package (included in the `eval` extra) and the system MagickWand library. On Ubuntu 22.04, install `libmagickwand-6.q16-6` with your package manager.

For RoboCasa, install the matching GR1 RoboCasa and Robosuite sources under `resources/robocasa/simulator/`, and prepare its separate interpreter at `resources/robocasa/environments/robocasa-sim/bin/python`, or pass `--sim-python` when preparing evaluation resources. The policy interpreter is independently configurable.

`PPU_SDK_ROOT` is optional and must point to an existing SDK if used. No SDK path is hardcoded.

## Common model resources

Installing Python packages does not download WanVAE, T5 or tokenizer files.
Before starting evaluation, run the [common-component download commands](../README.md#wanvae-t5-and-tokenizer)
in the policy environment and follow the [directory layout](RESOURCES.md#wanvae-t5-and-tokenizer).
They use ModelScope for the two converted `.safetensors` files and Hugging Face
for the tokenizer, matching the released Phoenix inference configs. Evaluation
launchers are offline and cannot fill in missing downloads automatically.
Simulator sources/assets must also be prepared separately for the selected benchmark.

## LIBERO-Pro

LIBERO-Pro uses the project `eval` extra, MuJoCo 3.3.2 and Robosuite 1.4.0. Obtain an external LIBERO-PRO checkout and its official resources following [RESOURCES.md](RESOURCES.md#libero-pro-benchmark). Do not run `pip install -e` on that checkout over LIBERO-Plus: the Pro launcher selects it through an explicit import path in a fresh process. Plus continues to use its bundled integration.

Prepare Pro with `--libero-repo-root` and `--benchmark-root`, then launch [LIBERO-Pro evaluation](EVALUATION.md#libero-pro). Use driver-compatible EGL libraries. PPU needs its own compatible SDK/Torch environment. Keep the separate RoboCasa simulator environment unchanged.

## Isolated installation

For an isolated policy/LIBERO installation, use a Python 3.10 interpreter and do not pass
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
```

Run resource preparation with this environment's Python so `eval.env.sh` records
the new interpreter. Do not reuse an environment file pointing to an old Python.
Use the benchmark commands in [EVALUATION.md](EVALUATION.md) with new output
directories. RoboCasa's simulator must remain in its own environment; reusing
external asset files and weights does not require copying Python packages.

Install training dependencies separately with `.[train,logging]`.
The first Torch command only selects the accelerator wheels; do not stop there
or skip the subsequent complete dependency installation.
Do not install `decord` merely because it existed in an older environment; it is
not a declared dependency of this project.

Use a virtual environment without `--system-site-packages`. An offline install
may use original wheel/source distributions via `--no-index --find-links`, but
must still resolve dependencies; do not copy installed
packages from an existing environment or bypass dependencies with `--no-deps`.

For an evaluation-only RoboCasa simulator environment, the dependency set is
separate from the policy environment:

```bash
bash scripts/install_simulator_env.sh robocasa resources/robocasa/environments/robocasa-sim
```

The launcher imports the simulator checkouts from `ROBOCASA_RESOURCES/simulator`
through `PYTHONPATH`; this minimal evaluation environment does not install the
upstream training extras or a second Torch build. It targets Robosuite 1.5.1
source and the GR1 fork with MuJoCo 3.3.2. Do not apply it to LIBERO, which uses
Robosuite 1.4.0. Source revisions/assets remain separate prerequisites.
Keep policy packages out of the separate simulator environment.

## Rendering backends

| Benchmark | Renderer | Environment boundary |
| --- | --- | --- |
| LIBERO-Plus | MuJoCo/Robosuite, NVIDIA EGL offscreen | Policy/LIBERO environment, Robosuite 1.4.0 |
| LIBERO-Pro | MuJoCo/Robosuite, NVIDIA EGL offscreen | Same dependencies; separate benchmark import process |
| RoboCasa GR1 | MuJoCo/Robosuite, NVIDIA EGL offscreen | Separate GR1 simulator, Robosuite 1.5.1 |
| RoboTwin C2C/C2R | SAPIEN, NVIDIA Vulkan offscreen | Separate SAPIEN/CuRobo simulator environment |

No desktop display is required. EGL and Vulkan are different backends: setting
`MUJOCO_GL=egl` does not configure RoboTwin rendering. Install driver-compatible
NVIDIA userspace libraries and loader manifests on the target machine.

On a container where EGL cannot discover NVIDIA despite the driver being present,
set `__EGL_VENDOR_LIBRARY_FILENAMES` to the installed NVIDIA EGL vendor manifest.
If you stage that manifest in ignored resources, a portable setting is:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/configs/render/nvidia_egl.json"
```

This file must actually exist and reference the installed driver library; do not
copy arbitrary driver binaries from another machine.
Generated environment files record the selected interpreter: regenerate them if
an environment is moved or removed, rather than reusing an obsolete temporary one.

## RoboTwin environment

Prepare the matching RoboTwin 2.0 simulator and its licensed assets under
`resources/robotwin/RoboTwin`. The release must include `envs/curobo` (CuRobo
0.7.8), robot descriptions and task configuration, not only the frozen episode
archive. Frozen C2C/C2R inputs do not contain simulator meshes or Python code.
Keep this environment separate from the policy/LIBERO environment.
Use the [exact revisions, asset downloads and local patches](ROBOTWIN.md#exact-simulator-and-assets).
The upstream `main` branch has a different layout and is not a substitute for
the simulator used by the frozen evaluation protocol.

On Ubuntu, install system build and rendering dependencies first:

```bash
sudo apt-get update
sudo apt-get install build-essential libvulkan1 vulkan-tools libgl1 libegl1 \
  libglib2.0-0 ffmpeg
```

An NVIDIA driver with graphics capability and a CUDA development toolkit (`nvcc`,
headers and libraries, not just Torch runtime wheels) are required. Use a CUDA
12.x toolkit compatible with the Torch CUDA 12.1 build. Set `CUDA_HOME` to the
installed toolkit and add its `bin` directory to `PATH` when it is not discovered
automatically. In a container, expose NVIDIA `compute,utility,graphics` driver
capabilities. The host driver and toolkit are system prerequisites, not Python
packages installed by the script.

```bash
export ROBOTWIN_ROOT="$PWD/resources/robotwin/RoboTwin"
bash scripts/install_simulator_env.sh robotwin resources/robotwin/sim-env
export ROBOTWIN_SIM_PYTHON="$PWD/resources/robotwin/sim-env/bin/python"
export PYTHON="$PWD/.venv/bin/python"
```

The installer creates a Python 3.10 venv without inherited site-packages, installs
Torch 2.4.1 / torchvision 0.19.1 (CUDA 12.1), installs the pinned simulator
requirements, builds CuRobo from the matching source, and runs `pip check`.
Core simulator versions are SAPIEN 3.0.0b1, MPLib 0.2.1, Open3D 0.18.0,
NumPy 1.26.4 and warp-lang 0.10.1. `BASE_PYTHON` selects the Python 3.10 executable;
`MAX_JOBS` controls native compilation parallelism. Do not reuse compiled CUDA
extensions from another Torch version. No old environment's installed packages
are copied. Downloaded wheels can be cached for offline installation.

CuRobo's target architecture is detected from the visible GPUs using Torch, not
inferred from a marketing model name. To compile for additional deployment GPUs,
set `TORCH_CUDA_ARCH_LIST` explicitly to their actual compute capabilities before
running the installer. Rebuild when changing GPU architecture; a wheel can import
successfully yet fail at the first CUDA kernel if it lacks the required device code.

Configure the remaining variables in `configs/machines/robotwin.example.sh`,
then keep `PYTHON` and `ROBOTWIN_SIM_PYTHON` pointing to these separate environments.
Use the venv's `bin/python` path, not its resolved base-interpreter symlink target.
The project does not automatically download licensed assets or frozen episodes.

The evaluation adapter discovers SAPIEN's bundled NVIDIA Vulkan/EGL manifest files
from the simulator interpreter when available. Ensure the matching Vulkan loader
and NVIDIA ICD are installed; a stale or incompatible driver manifest is not
fixed by changing `CUDA_VISIBLE_DEVICES`. Driver differences can produce small
pixel differences. Numeric frozen-scene checks remain required.

SAPIEN can fall back to its bundled Vulkan loader when the system loader is
absent; this does not supply the NVIDIA driver itself.

When the container is missing loader manifests but has the matching NVIDIA
libraries, the repository supplies manifests using library names (not private
machine paths):

```bash
export VK_ICD_FILENAMES="$PWD/configs/render/nvidia_vulkan.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/configs/render/nvidia_egl.json"
```

Explicit manifests are respected by the launcher. These JSON files do not install
drivers; `libGLX_nvidia.so.0` / `libEGL_nvidia.so.0` must be discoverable. A CPU
software renderer is not a substitute for the intended NVIDIA backend.

After configuring [RoboTwin resources](ROBOTWIN.md), launch both evaluation modes
using [the full benchmark commands](EVALUATION.md#robotwin-c2c-and-c2r). Training
uses the policy environment and online encoder weights; it does not require
SAPIEN rendering.
