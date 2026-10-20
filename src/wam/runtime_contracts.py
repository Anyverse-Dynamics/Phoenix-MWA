"""Structural resource checks; file digests are not runtime admission gates."""
import json
from pathlib import Path


def recorded_manifest_id(root, marker="_INDEX_COMPLETE"):
    """Read an opaque dataset ID without hashing files or claiming verification.

    Existing sample/RNG identities retain their published IDs. A completion
    marker is still required, but its digest is not compared to file bytes.
    """
    root = Path(root)
    payload = json.loads((root / marker).read_text())
    identity = payload.get("manifest_sha256")
    if not isinstance(identity, str) or not identity:
        raise ValueError(f"Missing dataset identity in {root / marker}")
    return identity


def semantic_contract(value):
    """Compare actual interfaces/configuration, not release fingerprints."""
    if isinstance(value, dict):
        return {
            key: semantic_contract(item)
            for key, item in value.items()
            if not key.endswith(("_sha256", "_fingerprint"))
            and key not in ("source_sha256", "dinov3_source_commit", "verify_source_commit",
                            "expected_csvae_step", "expected_pretrain_step", "expected_init_step",
                            "teacher_identity_verified", "teacher_structure_verified")
        }
    if isinstance(value, (list, tuple)):
        return [semantic_contract(item) for item in value]
    return value


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty resource: {path}")
    return path
