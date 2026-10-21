# Phoenix example operation guide

For Prometheus framework setup and policy integration, see the [repository README](../README.md). This guide covers the bundled Phoenix model service and client.

Run commands from the repository root. `run.sh` sets the source import paths; `PYTHON` selects the interpreter for each process.

## Model assets

| Asset | Purpose |
| --- | --- |
| `checkpoints/phoenix_pretrained.pt` | Weights-only training initialization |
| `checkpoints/phoenix_rtc.pt` | Completed H32 Training RTC inference checkpoint |
| `configs/phoenix_model.yaml` | Model architecture and visual codec configuration |
| `assets/phoenix_normalization.json` | State/action mean and standard deviation with gripper units |
| `assets/phoenix_text.pt` | Inference instruction and language context |

Supply Wan components, the Phoenix encoder, and CS-VAE separately. Set `PHOENIX_ENCODER_CHECKPOINT`, `PHOENIX_STVAE_CHECKPOINT`, and `DIFFSYNTH_MODEL_BASE_PATH` for the chosen resources.

To export a model definition:

```bash
PYTHON="$MODEL_PYTHON" bash run.sh prepare \
  --source-config configs/saved_model.yaml --output configs/phoenix_model.local.yaml
```

Set training's `model_definition` and inference's `model_config` accordingly. Exporting a configuration does not modify a checkpoint or train RTC behavior.

## Model service

Edit `configs/phoenix.yaml` to select the RTC checkpoint, normalization, instruction context, their SHA256 values, hardware embodiment, and endpoints.

The language file is a Torch dictionary with `instruction`, `context[1,L,4096]`, and boolean `mask[1,L]`; unbatched `[L,4096]` and `[L]` are also accepted. The text must match the configured instruction.

```bash
PYTHON="$MODEL_PYTHON" bash run.sh server --config configs/phoenix.yaml --check-config
PYTHON="$MODEL_PYTHON" bash run.sh server --config configs/phoenix.yaml
```

The configuration command reads assets and metadata. Starting the service constructs and restores the full model, then listens for clients. It does not connect to the robot environment.

## Recorded observations

The NPZ input contains float32 `state[14]` and three uint8 HWC images named `base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb`. RTC mode may include float32 `action_prefix[d,14]`, where `0≤d≤rtc.max_prefix_steps`. Sync mode requires an empty prefix.

```bash
PYTHON="$CLIENT_PYTHON" bash run.sh client --mode rtc \
  --input assets/observation.npz --output outputs/phoenix_actions.npz
```

The output includes all 32 physical actions and inference information. Existing output files are not overwritten.

## Robot execution

Configure hardware in `prometheus/configs/run_env/phoenix.yaml` and addresses in `configs/phoenix.yaml`. Start the model service on the model device and the following processes on the robot workstation:

```bash
PYTHON="$ROBOT_PYTHON" bash run.sh env rtc
PYTHON="$CLIENT_PYTHON" bash run.sh client --mode rtc --execute --chunks 3
```

The environment defaults to `env.dry_run=true`. Append `env.dry_run=false` to its command to enable physical action publication. The client prompts before resetting the environment.

For synchronous execution, select matching modes:

```bash
PYTHON="$ROBOT_PYTHON" bash run.sh env sync
PYTHON="$CLIENT_PYTHON" bash run.sh client --mode sync --execute --chunks 3
```

On client exit, the model generation is invalidated and the environment session is closed. RTC plans exceeding the queue's available time budget are rejected rather than submitted late.
