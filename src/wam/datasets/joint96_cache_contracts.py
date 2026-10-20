"""Paired WanVAE and CSVAE cache contracts, without dataset imports."""
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(f"VAE+CS-VAE cache: {message}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def pair_contracts(vae_root, csvae_roots):
    vae_root = Path(vae_root).resolve()
    csvae_roots = [Path(p).resolve() for p in csvae_roots]
    require(len(csvae_roots) == 2 and [p.name for p in csvae_roots] == ["head", "wrist"]
            and csvae_roots[0].parent == csvae_roots[1].parent, "CS-VAE head/wrist routing")
    roots = [vae_root / "head", vae_root / "wrist", *csvae_roots]
    contracts = [read_json(p / "contract.json") for p in roots]
    vr = read_json(vae_root / "contract.json")
    for view, contract in zip(("head", "wrist"), contracts[:2]):
        require(contract == {**vr, "view": view}, "VAE view contract identity")
    require(vr["scope"] == "full" and vr["selection"] is None
            and vr["feature_shape_per_view"] == [48, 3, 16, 16]
            and vr["encoder_type"] == "wan22_vae_per_view", "VAE256 native full cache required")
    sr = contracts[2]
    require({k:v for k,v in sr.items() if k != "view"} ==
            {k:v for k,v in contracts[3].items() if k != "view"}, "CS-VAE view contract identity")
    require(all(c["dataset_length"] == 277713 for c in contracts), "full LIBERO length")
    require(len({c["dataset_snapshot_sha256"] for c in contracts}) == 1, "dataset snapshot mismatch")
    require(vr["reference_csvae_root"] == str(csvae_roots[0].parent), "VAE paired with another CS-VAE cache")
    for key in ("state_frames", "action_horizon", "rgb_offsets", "latent_frames", "pad_policy"):
        require(vr["window"][key] == sr["window"][key], f"window mismatch: {key}")
    require(sr["selection"] == "full_dataset" and sr["posterior"] == "mean"
            and sr["extra_latent_scaling"] == "none", "CS-VAE deterministic full mean cache required")
    return roots, contracts
