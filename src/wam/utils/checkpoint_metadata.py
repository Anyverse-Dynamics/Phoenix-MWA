"""Portable metadata for public weights; runtime and resume paths stay local."""
from __future__ import annotations

import os
import re
from collections.abc import Mapping


# Match paths as whole values and inside explanatory strings. A remote or bundled
# URI (https://..., bundled://...) is not a local filesystem location.
_LOCAL_PATH = re.compile(
    r"(?:^|[\s\"'=([{<>:,;])(?:/(?!/)|~[/\\]|[A-Za-z]:[/\\]|\\\\)"
    r"|(?:^|[\s\"'=([{<>,;])//"
    r"|(?:^|[\s\"'=([{<>:,;])file:",
    re.IGNORECASE,
)
_OMIT = object()


def _text(value):
    return os.fspath(value) if isinstance(value, os.PathLike) else value


def _local_path(value):
    value = _text(value)
    return isinstance(value, str) and bool(_LOCAL_PATH.search(value))


def private_checkpoint_metadata_fields(value):
    """Locate local paths in keys and values.

    Locations contain dictionary positions rather than private key contents.
    This keeps error messages from repeating the metadata being removed.
    """
    found = []

    def visit(item, location):
        if _local_path(item):
            found.append(location)
        elif isinstance(item, Mapping):
            for index, (key, child) in enumerate(item.items()):
                visit(key, f"{location}.key[{index}]")
                visit(child, f"{location}.value[{index}]")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")

    visit(value, "$")
    return found


def sanitize_checkpoint_metadata(value):
    """Return a copy without machine-local provenance paths.

    Keep steps, digests, model semantics and portable relative paths/URIs.
    Remove complete path-bearing fields instead of inventing a new historical
    source location. Never apply this to a training-state resume contract.
    """
    def clean(item):
        item = _text(item)
        if _local_path(item):
            return _OMIT
        if isinstance(item, Mapping):
            result = {}
            for key, child in item.items():
                if _local_path(key):
                    continue
                if not isinstance(key, str):
                    raise TypeError("Checkpoint metadata dictionary keys must be strings")
                copied = clean(child)
                if copied is not _OMIT:
                    result[key] = copied
            return result
        if isinstance(item, (list, tuple)):
            result = [copied for child in item if (copied := clean(child)) is not _OMIT]
            return tuple(result) if isinstance(item, tuple) else result
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise TypeError(f"Unsupported checkpoint metadata value: {type(item).__name__}")

    result = clean(value)
    if result is _OMIT:
        raise ValueError("Checkpoint metadata root must not be a local path")
    if private_checkpoint_metadata_fields(result):
        raise ValueError("Checkpoint metadata is not portable after sanitization")
    return result
