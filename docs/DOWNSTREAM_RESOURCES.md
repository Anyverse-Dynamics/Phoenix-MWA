# Phoenix downstream model resources

[Encoder overview](../README.md)

These resource commands support the world–action model integration. Encoder pretraining uses the teacher sources and weights described in the [encoder guide](encoder_pretrain/README.md). Run commands from the repository root.

## WanVAE, T5 and tokenizer

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
See [full directory layout, offline checks and troubleshooting](RESOURCES.md#wanvae-t5-and-tokenizer).

## LIBERO training data

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
[evaluation resources](RESOURCES.md#libero-plus-benchmark).
