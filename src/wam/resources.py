"""Relocate immutable resource IO without changing sample identities or RNG keys."""
from functools import lru_cache
import json
import os
from pathlib import Path


def repository_root():
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _bindings():
    """Load explicit, local-only deployment bindings kept outside the checkout."""
    filename = os.environ.get("PHOENIX_RESOURCE_BINDINGS")
    if not filename:
        return {"version": 1, "mappings": {}, "aliases": {}}
    payload = json.loads(Path(filename).expanduser().read_text())
    if payload.get("version") != 1:
        raise ValueError("Unsupported resource bindings")
    return payload


@lru_cache(maxsize=1)
def _mappings():
    root = repository_root()
    payload = _bindings()
    result = []
    for old, relative in payload.get('mappings', {}).items():
        target = (root / relative).resolve()
        if not target.is_relative_to((root / 'resources').resolve()):
            raise ValueError('Resource mapping escapes Phoenix: ' + relative)
        result.append((Path(old), target))
    return sorted(result, key=lambda item: len(str(item[0])), reverse=True)


def resolve_resource_references(value):
    """Restore local index identities from portable metadata references.

    Published metadata contains logical references. Existing private indexes
    may use deployment-specific identities as database keys; resolving them
    here preserves their sample ordering and augmentation seeds.
    """
    if isinstance(value, str) and value.startswith("local-resource://"):
        aliases = _bindings().get("aliases", {})
        if value not in aliases:
            raise ValueError("Missing local resource binding: " + value)
        return aliases[value]
    if isinstance(value, dict):
        return {resolve_resource_references(key): resolve_resource_references(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_resource_references(item) for item in value]
    return value


def resource_path(path):
    """For package/cache IO only; never rewrite the stored metadata or identities."""
    path = Path(resolve_resource_references(str(path)))
    if not path.is_absolute() and path.parts and path.parts[0] == "resources":
        path = repository_root() / path
    for old, target in _mappings():
        if path == old or path.is_relative_to(old):
            result = target / path.relative_to(old)
            if not result.exists():
                raise FileNotFoundError('Missing relocated resource: ' + str(result))
            return result
    return path
