"""File identities and atomic publication shared by data preparation stages."""
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temp.open("w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def directory_guard(path):
    """Atomic mkdir also works on shared filesystems without relying on flock.

    After a killed writer, remove its owner directory manually only after
    checking that the process has stopped. Completed shards remain reusable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    try:
        atomic_json(path / "owner.json", {"pid": os.getpid()})
        yield
    finally:
        (path / "owner.json").unlink(missing_ok=True)
        path.rmdir()


def inventory(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "CONVERTED.json"}


def check_sources(manifest):
    for name, expected in manifest["metadata_sha256"].items():
        if sha256(name) != expected:
            raise ValueError("Frozen dataset metadata changed")
    files = dict(manifest["files"])
    for episode in manifest["episodes"]:
        for view in episode["views"]:
            files[view["path"]] = view
    for name, identity in files.items():
        stat = Path(name).stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError("Frozen dataset file changed")
