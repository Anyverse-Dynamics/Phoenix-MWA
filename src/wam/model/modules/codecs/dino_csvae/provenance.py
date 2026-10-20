"""Bundled encoder interface, independent of source file fingerprints."""

import copy
from pathlib import Path

BUNDLED_SOURCE = "bundled://dino-csvae-v2"
PACKAGE = Path(__file__).resolve().parent
WAM_ROOT = PACKAGE.parents[3]
RUNTIME = WAM_ROOT / "third_party/dino_runtime"


def verify_bundle():
    """Compatibility API: check required modules, not source bytes."""
    for path in (PACKAGE / "csvae.py", PACKAGE / "online_dinov3_csvae_encoder.py",
                 RUNTIME / "dinov3", RUNTIME / "starVLA"):
        if not path.exists():
            raise FileNotFoundError(f"Missing bundled encoder module: {path.name}")
    return {"schema": "phoenix.encoder.source.v2", "source_sha256": None}


def bundled_config(config):
    result = copy.deepcopy(dict(config))
    verify_bundle()
    result.update(source_root=BUNDLED_SOURCE, source_sha256=None)
    result["dino_kwargs"] = dict(result["dino_kwargs"])
    result["dino_kwargs"].update(starvla_source_path=str(RUNTIME), dinov3_source_path=str(RUNTIME / "dinov3"))
    return result


def canonical_source_identity(encoder):
    """Validate the supported encoder interface; digests are informational."""
    if encoder.get("source_root") != BUNDLED_SOURCE:
        raise ValueError("Unsupported CSVAE source identity")
    verify_bundle()
    return BUNDLED_SOURCE
