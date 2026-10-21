#!/usr/bin/env python3
"""Verify the shipped file set and SHA256 values using only the standard library."""
from pathlib import Path
import hashlib
import sys


def main():
    root = Path(__file__).resolve().parent
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in expected:
            raise ValueError(f"Invalid manifest entry: {relative}")
        expected[relative] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    actual.discard("SHA256SUMS")
    errors = []
    for relative in sorted(actual - set(expected)):
        errors.append(f"unexpected: {relative}")
    for relative, digest in expected.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            errors.append(f"missing or symlink: {relative}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != digest:
            errors.append(f"changed: {relative}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"BUNDLE_SHA256_VERIFIED: {len(expected)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
