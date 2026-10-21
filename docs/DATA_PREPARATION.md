# Phoenix example data preparation

The data pipeline has five explicit stages:

```text
Prometheus recordings
  → native LeRobot v3 episode shards
  → frozen H32 manifest and normalization selection
  → Wan UMT5 instruction contexts
  → frozen Wan VAE + DINO/CS-VAE visual windows
  → Phoenix Training RTC
```

`bash run.sh data` runs one stage at a time. It never starts training or robot services. Inputs remain read-only; generated data and weights are external to the source release.

## Environments

Use a dedicated conversion environment, separate from the accelerator environment:

```bash
python3 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r requirements-data.txt
python -m pip install -e . --no-deps
export PYTHON="$(command -v python)"
```

Conversion uses the official `lerobot==0.4.3` writer and its supported Torch/torchvision dependencies. A compatible FFmpeg installation is required for H.264 (`libx264`) encoding. The converter calls `save_episode()` and `finalize()` before publishing a completed native-v3 shard. See the [LeRobot v3 documentation](https://huggingface.co/docs/lerobot/v0.4.3/en/lerobot-dataset-v3).

Preparation, text caching, visual caching, and training use the model environment described in the [Phoenix example guide](PHOENIX_RTC.md#getting-started). Text caching loads only Wan UMT5 and its tokenizer. Visual caching loads only the frozen Wan VAE and DINO/CS-VAE branches; policy weights and the action expert are not needed. Real encoder execution requires a supported CUDA-compatible accelerator and sufficient memory for the selected frozen components.

## 1. Convert Prometheus recordings

Create an external JSON array containing explicit episode paths and instructions. Paths in this array resolve from the command's working directory; absolute paths are recommended:

```json
[
  {"path": "recordings/episode_000000", "instruction": "Pick up the object."},
  {"path": "recordings/episode_000001", "instruction": "Pick up the object."}
]
```

Instruction text is never inferred from directory names. Only open trusted Prometheus recordings: their acquisition metadata uses Python pickle.

Supported layouts:

```text
episode/                         Flat recording layout
  robot_state_dict.pkl
  image_timestamps.pkl
  base_0_rgb.mp4
  left_wrist_0_rgb.mp4
  right_wrist_0_rgb.mp4

episode/                         Nested recording layout
  robot/robot_state_dict.pkl
  camera/timestamp.pkl
  camera/base_0_rgb.mp4
  camera/left_wrist_0_rgb.mp4
  camera/right_wrist_0_rgb.mp4
```

Robot streams default to `left` and `right`, each containing `joint[N,7]` and `timestamps[N]`. Camera timestamp keys default to `base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb`; keys without `_rgb` are accepted for nested recordings. Timestamps are milliseconds in a common clock domain. Override `conversion.robot_parts` and `conversion.camera_streams` if acquisition uses other logical names, preserving left/right and main/left-wrist/right-wrist order.

```bash
export PHOENIX_RECORDING_LIST="$PWD/data/recording_episodes.json"
bash run.sh data convert
```

The converter finds the common sensor/robot interval and samples a regular 30 Hz clock using nearest timestamps. It cleans invalid robot samples, sorts timestamps, removes duplicates, applies alignment tolerances, optionally trims static boundaries, and rejects short, discontinuous, blank, or saturated episodes. Quality thresholds are configurable in `configs/phoenix_data.yaml`; gripper thresholds use the declared source unit.

The output uses physical absolute joint14 labels:

- State at step `t` is the aligned observed state `q[t]`.
- `action_abs_joint[t]` is `q[t+1]`, a **next-observation proxy action**. It is not a recorded command target.
- The final aligned observation has no next-state label and is dropped.
- No joint/gripper unit conversion is performed. Joints must already use radians; declare the actual gripper unit consistently through conversion, preparation, training, and deployment.
- Images are RGB, resized to 256×256 and encoded at 30 Hz.

This path includes the three RGB streams and state/action joint14 fields used by Phoenix. Tactile, EEF, velocity, and effort signals are not converted into Phoenix training features. To train on actual command targets, supply an existing aligned v3 dataset and select its field and `commanded_target` semantics at preparation.

Each source episode becomes an independent native-v3 dataset under `data/lerobot/episode_XXXXXX/`. `dataset_dirs.json` lists the datasets. `CONVERTED.json` records source hashes, settings, output hashes, action semantics, and units. Repeating the same command reuses completed, unchanged episode shards. Existing outputs with different settings or source content are rejected. Inspect an interrupted `.partial` directory before removing that directory and retrying; completed shards remain reusable.

## 2. Freeze membership, split, and normalization

Use an existing normalization file that matches the intended initialization/deployment coordinates:

```bash
source .venv/bin/activate
export PYTHON="$(command -v python)"
export PHOENIX_NORMALIZATION="$PWD/assets/phoenix_normalization.json"
bash run.sh data prepare
```

The preparer accepts `dataset_dirs.json` from conversion or a JSON array of existing native-v3 dataset roots. It requires at least two episodes, 30 Hz, three ordered RGB streams, and physical 14-dimensional state/action fields. It fixes an episode-level train/held-out split with `split_seed=42` and `holdout=0.05` by default. Parquet rows, task identities, video intervals, source identities, and metadata hashes are recorded or checked before publication.

```text
data/prepared/
  manifest.json          Frozen membership, windows, source identities, episode split
  normalization.json     Selected normalization, copied without numeric/unit changes
  train_statistics.json  Statistics fitted from training episodes only
  PREPARED.json          Manifest and normalization fingerprints
```

For a deliberate new normalization, explicitly enable fitting:

```bash
bash run.sh data prepare \
  preparation.normalization=null preparation.fit_normalization=true
```

New statistics change normalized coordinates. Use them consistently with the intended training initialization and deployment; do not silently substitute them for an existing model's statistics. Preparation always writes to a new directory and never changes existing prepared versions.

For existing datasets, override `preparation.dataset_list`, `state_key`, `action_key`, `camera_keys`, `action_semantics`, and `gripper_unit` to match their actual fields. A converted shard's declared semantics and units must agree with preparation.

## 3. Cache language contexts

Switch to the model environment. Supply the original Wan UMT5 weights and tokenizer directory:

```bash
export PHOENIX_T5_CHECKPOINT="$PWD/checkpoints/models_t5_umt5-xxl-enc-bf16.pth"
export PHOENIX_T5_TOKENIZER="$PWD/checkpoints/google/umt5-xxl"
bash run.sh data cache-text
```

Each distinct instruction is encoded with the shared Phoenix prompt template. The cache contains BF16 `[128,4096]` contexts and boolean masks. Padding embeddings are zeroed, and all context tokens remain visible, matching `WorldActionModel.encode_prompt`. The index binds the exact manifest, prompt text, encoder weights, tokenizer files, and encoding implementation. Completed entries are reusable only under the same contract and their individual hash receipts. `text_index.json` is published after all required prompts are present; `READY.json` records its fingerprint. If interrupted between saving a tensor and its receipt, inspect and remove that incomplete tensor before retrying, or choose a new output directory.

## 4. Cache visual windows

Use the same model definition, Wan VAE, DINO weights, and CS-VAE weights as training:

```bash
export PHOENIX_ENCODER_CHECKPOINT="$PWD/checkpoints/phoenix_encoder.pt"
export PHOENIX_STVAE_CHECKPOINT="$PWD/checkpoints/phoenix_stvae.pt"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
bash run.sh data cache-visual
```

Every valid window has 32 actions and spans 33 raw frames. Video offsets `[0,4,...,32]` produce nine frames in main/left-wrist/right-wrist order. Episode tails repeat the last image; training masks padded supervision. Posterior means are encoded into `[96,3,16,48]` and stored as BF16 bit patterns in `uint16` memmaps. Each window uses numerical batch size one. The cache preserves the model definition's `current_conditioning` mode, including independent single-frame conditioning when configured, and never reuses H64/17-frame cache contents.

The default shard contains at most 256 windows. Storage is approximately **432 KiB per window** before filesystem/receipt overhead; 100,000 windows require about 41.2 GiB. Completed shards have whole-file and per-row SHA256 values. The cache contract binds the manifest, actual frozen weight hashes, encoder implementation, RGB decoding, shapes, camera order, temporal offsets, conditioning mode, and dtype. The reader validates each row when consumed.

For multiple devices or machines, use a shared cache directory and identical settings. Initialize once before starting workers:

```bash
bash run.sh data cache-visual visual_cache.initialize_only=true

# Run one command per device, assigning unique ranks 0 through world_size-1.
CUDA_VISIBLE_DEVICES=0 bash run.sh data cache-visual \
  visual_cache.rank=0 visual_cache.world_size=2
CUDA_VISIBLE_DEVICES=1 bash run.sh data cache-visual \
  visual_cache.rank=1 visual_cache.world_size=2

# Run after every worker completes.
bash run.sh data finalize-visual
```

Workers own disjoint shard ordinals. Repeating an interrupted worker reuses unchanged completed shards. A single worker finalizes automatically; multiple workers require the explicit finalization command. Finalization checks all expected shards and their hashes before publishing `index.json` and `READY.json`. An incomplete cache is never accepted by training.

`*.owner` directories prevent concurrent writers using atomic directory creation. After a killed process, confirm the writer has stopped before removing its stale owner directory. Do not remove ownership while another worker is active. Use a new cache directory if data, weights, preprocessing, or shard size changes.

## 5. Train from caches

```bash
export PHOENIX_MANIFEST="$PWD/data/prepared/manifest.json"
export PHOENIX_NORMALIZATION="$PWD/data/prepared/normalization.json"
export PHOENIX_TEXT_INDEX="$PWD/data/text_cache/text_index.json"
export PHOENIX_VISUAL_CACHE_INDEX="$PWD/data/visual_cache/index.json"

# Also set initialization/encoder paths and SHA256 values from the Phoenix example guide.
bash run.sh train output_dir=runs/phoenix_training_rtc
```

With `data.visual_cache_index` configured, the dataset reads cached latents alongside parquet state/action labels and cached language contexts. It avoids RGB decoding and visual encoding during each training step. Frozen encoder resources are still required by model construction, checkpoint identity, and inference; do not omit them. BF16 visual caching requires `training.mixed_precision=bf16`.

Set `data.visual_cache_index=null` to encode the same windows online. Both paths retain the H32 objective and the default **100% uniform inclusive P0–P10** RTC sampler. The cache fingerprint participates in the saved training contract and full-state resume checks.
