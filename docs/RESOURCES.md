# Resources

Run commands from the repository root. Weights, data, generated environment
files and outputs belong in Git-ignored directories. A source checkout does not
include model weights, prepared indexes or raw data.

## Layout

| Resource | Repository-relative location |
| --- | --- |
| Wan2.2-TI2V-5B DiT shards | `resources/weights/Wan-AI/Wan2.2-TI2V-5B/` |
| WanVAE | `resources/weights/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors` |
| Phoenix Encoder | `resources/weights/phoenix_encoder/step_00110000.pt` |
| CS-VAE | `resources/weights/csvae/csvae.pt` |
| Data indexes, normalization and view plans | `resources/data_manifests/` |
| Precomputed UMT5 contexts | `resources/text_cache/` |
| Optional resource inventory | `resources/MANIFEST.json` |
| Local deployment bindings | External JSON selected by `PHOENIX_RESOURCE_BINDINGS` |

Use the dataset package, view plan and RGB selections specified in
`configs/phoenix_pretrain.yaml`. Every worker needs access to the raw videos and
numeric trajectories referenced by these indexes. The resource package does not
duplicate raw data. Text caches contain 128 × 4,096 BF16 contexts.
Fresh training does not require a policy checkpoint or optimizer state.

The full raw-data preparation pipeline is not supplied by this source release.
Obtain the matching prepared resources before training. Encoder downloads are
listed at [Phoenix Encoders](https://huggingface.co/OpenPhoenix/Phoenix-Encoders);
weights and data remain subject to their own terms.

## Portable paths

Configure resources through YAML or command-line overrides. Paths resolved from
the checkout root work independently of the installation directory. Keep
deployment-specific paths outside the checkout and set
`PHOENIX_RESOURCE_BINDINGS` to that local JSON file. Its `mappings` relocate
indexed prefixes without changing sample IDs; `aliases` resolve portable
`local-resource://` references used by prepared metadata:

```json
{
  "version": 1,
  "mappings": {
    "dataset-resources/text": "resources/text_cache"
  },
  "aliases": {}
}
```

Mapping destinations must stay inside `resources/`. Deployment bindings, raw
dataset indexes, text-cache indexes, and complete training states are local
inputs and are excluded from the public source and model-weight releases.
These indexes may retain private filesystem locations and stable sample IDs;
they must not be included in a source package. Raw datasets must be mounted at their
indexed locations or prepared with appropriate paths. For ARX media, set
`PHOENIX_ARX_ORIGINAL_ROOT` if the original videos are not in the expected sibling
dataset directory; frame timing and media-link consistency remain checked.

## Compatibility

The bundled visual encoder uses CS-VAE model type `causal_semantic_vae`, format
version 2, raw 1,024-dimensional DINO features and 48-dimensional causal latents.
Wan and CS-VAE latents form 96 channels. Cameras are ordered main/left/right;
actions and states use the canonical 80-dimensional convention.

Loading validates model type, tensor keys/shapes, supported dtypes, finite
values, normalization, camera/coordinate conventions and data schema. Dataset
completion markers, index bounds, array sizes and text routing are checked.
Published dataset IDs are retained as opaque associations, not recomputed from
file bytes. Random sampling and prompt/cache keys still use deterministic hashes.

Runtime loading does not check source or checkpoint file digests, Git commit
fingerprints, or development reports. Checkpoint steps are read for logging,
not matched to a fixed release step. Full-state resume validates actual training
counters, topology, complete optimizer/scheduler/RNG files and the saved recipe.
It is not interchangeable with weights-only initialization.

Without file-digest verification, same-size corruption cannot always be detected.
Use trusted resource downloads and keep backups of full training states.
