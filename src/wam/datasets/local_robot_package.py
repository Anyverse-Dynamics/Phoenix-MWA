"""Immutable native-window packages derived from the existing physical holdout."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import zlib
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from wam.resources import resource_path

from wam.datasets.local_robot_contracts import (
    CONFIG_ROOT, SCHEMA, digest, eligible_group, native_ranges,
    normalization_policy, resolve_profile, sha256,
)

DEFAULT_SPLIT = str(Path(__file__).resolve().parents[3] / "resources/data_manifests/base_split")
UR_SINGLETON_EXCLUSIONS = {
    "robomind2_8e7feb922f407865": "92df74b4fdc6ce13696679acd9b40ebf0e725296ed641064917e345360ba36e5",
    "robomind2_bb0184581d19d068": "e4cb27c6d58e6d66d229abd2c44a82db47469c2c4be594e6caa3e985c2e3b3bf",
}


def read_json(path):
    from wam.resources import resolve_resource_references
    return resolve_resource_references(json.loads(resource_path(path).read_text()))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def readonly_db(path):
    db = sqlite3.connect(resource_path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    db.execute("PRAGMA query_only=ON")
    return db


def pack(value):
    return zlib.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())


def unpack(value):
    return json.loads(zlib.decompress(value))


def prepare_package(output, *, split_package=DEFAULT_SPLIT, limit_per_group=0, exclude_ur_singletons=False):
    """Never regenerate holdouts. A limited audit package cannot be published for training."""
    output, old = Path(output).resolve(), Path(split_package).resolve()
    if output.exists():
        raise FileExistsError(output)
    if limit_per_group < 0:
        raise ValueError("limit_per_group must be nonnegative")
    m = read_json(old / "manifest.json")
    marker = read_json(old / "_SPLIT_COMPLETE")
    inputs = {}
    for name in ("manifest.json", "membership.npz"):
        actual = sha256(old / name)
        if marker["files"][name] != actual:
            raise ValueError("old split identity mismatch: " + name)
        inputs[str(old / name)] = actual
    with np.load(old / "membership.npz", allow_pickle=False) as archive:
        members = {k: archive[k] for k in archive.files}
    output.mkdir(parents=True)
    db = sqlite3.connect(output / "episodes.sqlite")
    db.executescript("""
        CREATE TABLE episodes (id INTEGER PRIMARY KEY, profile TEXT NOT NULL,
            split TEXT NOT NULL, unit TEXT NOT NULL, nstarts INTEGER NOT NULL,
            ranges BLOB NOT NULL, record BLOB NOT NULL, root_id TEXT NOT NULL,
            old_group TEXT NOT NULL, old_ordinal INTEGER NOT NULL,
            UNIQUE(old_group,old_ordinal));
        CREATE TABLE roots (id TEXT PRIMARY KEY, payload BLOB NOT NULL);
        CREATE TABLE units (id TEXT PRIMARY KEY, split TEXT NOT NULL);
        CREATE INDEX episodes_profile_split ON episodes(profile,split,id);
    """)
    profiles, totals, catalogs, connections = {}, Counter(), {}, {}
    exclusions, profile_cache = [], {}
    expected_exclusions = UR_SINGLETON_EXCLUSIONS if exclude_ur_singletons else {}
    if expected_exclusions and not set(expected_exclusions).issubset({g["id"] for g in m["groups"]}):
        raise ValueError("approved singleton groups are missing from old membership")
    started, seen = time.monotonic(), 0
    expected_records = sum(sum(g["splits"].values()) if not limit_per_group else
                   sum(min(limit_per_group, n) for n in g["splits"].values())
                   for g in m["groups"] if eligible_group(g))
    input_catalogs = {}
    try:
        for g in m["groups"]:
            if not eligible_group(g):
                continue
            arrays = [members[s + "__" + g["id"]] for s in ("train", "validation")]
            for split, values in zip(("train", "validation"), arrays):
                if values.dtype != np.int64 or values.ndim != 1 or len(values) != g["splits"][split]:
                    raise ValueError("membership shape/count mismatch")
                if len(values) and (values[0] < 0 or values[-1] >= g["parent_episodes"] or np.any(np.diff(values) <= 0)):
                    raise ValueError("membership ordinal mismatch")
            if np.intersect1d(*arrays).size:
                raise ValueError("train/validation ordinal leakage")
            if g["id"] in expected_exclusions and (arrays[0].tolist() != [0] or arrays[1].tolist() != []):
                raise ValueError("approved UR singleton membership changed")
            catalog_root = Path(m["catalogs"][g["catalog"]])
            if str(catalog_root) not in catalogs:
                c = read_json(catalog_root / "catalog.json")
                catalog_hash = sha256(catalog_root / "catalog.json")
                if read_json(catalog_root / "_CATALOG_COMPLETE")["sha256"] != catalog_hash:
                    raise ValueError("old catalog marker mismatch")
                expected = m["inputs"].get(str(catalog_root / "catalog.json"))
                if expected != catalog_hash:
                    raise ValueError("old split/catalog release mismatch")
                inputs[str(catalog_root / "catalog.json")] = catalog_hash
                catalogs[str(catalog_root)] = c
            c = catalogs[str(catalog_root)]
            desc = c["sources"][g["source"]]
            catalog_db = catalog_root / desc["file"]
            if catalog_db.stat().st_size != desc["bytes"]:
                raise ValueError("old catalog size drift")
            if "mtime_ns" in desc and catalog_db.stat().st_mtime_ns != desc["mtime_ns"]:
                raise ValueError("old catalog modification time drift")
            input_catalogs[str(catalog_db)] = dict(bytes=desc["bytes"], sha256=desc["sha256"])
            if str(catalog_db) not in connections:
                connections[str(catalog_db)] = readonly_db(catalog_db)
            old_db = connections[str(catalog_db)]
            for split, ordinals in zip(("train", "validation"), arrays):
                selected = ordinals[:limit_per_group] if limit_per_group else ordinals
                for ordinal in selected:
                    result = old_db.execute("SELECT payload FROM episodes WHERE group_id=? AND ordinal=?",
                                            (g["id"], int(ordinal))).fetchone()
                    if result is None:
                        raise ValueError(f"old catalog membership missing: {g['id']}/{ordinal}")
                    row = unpack(result[0])
                    seen += 1
                    if g["id"] in expected_exclusions:
                        if (row["physical_episode_uid"] != expected_exclusions[g["id"]]
                                or split != "train" or int(ordinal) != 0):
                            raise ValueError("approved UR singleton physical identity changed")
                        exclusions.append(dict(old_group=g["id"], old_ordinal=int(ordinal), split=split,
                            physical_episode_uid=row["physical_episode_uid"], sample_id=row["sample_id"],
                            reason="user_approved_exclusion_of_two_train_only_UR_singletons"))
                        totals[split + "_excluded_episodes"] += 1
                        continue
                    if row["physical_episode_uid"] in expected_exclusions.values():
                        raise ValueError("excluded physical trajectory unexpectedly appears in another group")
                    if float(row["fps"]) != float(g["signature"][5]):
                        raise ValueError("native FPS/group mismatch")
                    if row["source_id"] != g["source"] or row["robot_type"] != g["signature"][1]:
                        raise ValueError("record/group embodiment mismatch")
                    cache_key = g["id"], row["source_root"] if g["source"] == "robocoin" else ""
                    if cache_key not in profile_cache:
                        profile_cache[cache_key] = resolve_profile(g, row)
                    profile = profile_cache[cache_key]
                    profiles[profile["id"]] = profile
                    unit = "physical:" + row["physical_episode_uid"]
                    db.execute("INSERT OR IGNORE INTO units VALUES (?,?)", (unit, split))
                    if db.execute("SELECT split FROM units WHERE id=?", (unit,)).fetchone()[0] != split:
                        raise ValueError("physical trajectory crosses train/validation")
                    ranges = native_ranges(row)
                    nstarts = sum(b - a for a, b, _ in ranges)
                    if not nstarts:
                        totals[split + "_no_native_window"] += 1
                        continue
                    if set(v["key"] for v in row["selected_views"]) != set(profile["cameras"]) - {""}:
                        raise ValueError("episode camera selection differs from group contract")
                    rid = digest([str(catalog_db), row["root_id"]])
                    if not db.execute("SELECT 1 FROM roots WHERE id=?", (rid,)).fetchone():
                        root = old_db.execute("SELECT payload FROM roots WHERE root_id=?", (row["root_id"],)).fetchone()
                        if root is None:
                            raise ValueError("catalog root missing")
                        db.execute("INSERT INTO roots VALUES (?,?)", (rid, root[0]))
                    db.execute("INSERT INTO episodes VALUES (NULL,?,?,?,?,?,?,?,?,?)",
                               (profile["id"], split, unit, nstarts, pack(ranges), pack(row), rid, g["id"], int(ordinal)))
                    totals[split + "_episodes"] += 1
                    totals[split + "_legal_starts"] += nstarts
                    if seen % 10000 == 0:
                        elapsed = time.monotonic() - started
                        progress = dict(stage="index", records=seen, expected_records=expected_records,
                            elapsed_seconds=elapsed, eta_seconds=elapsed / seen * (expected_records - seen), totals=dict(totals))
                        write_json(output / "index_progress.json", progress)
                        print(f"[local_robot] progress {json.dumps(progress)}", flush=True)
            db.commit()
            print(f"[local_robot] indexed {g['id']} {dict(totals)}", flush=True)
        if not totals["train_episodes"] or not totals["validation_episodes"]:
            raise ValueError("package needs nonempty held-out and training membership")
        if len(exclusions) != len(expected_exclusions):
            raise ValueError("approved exclusion scope was not matched exactly")
        summary = [dict(profile=p, split=s, episodes=n, legal_starts=k)
                   for p, s, n, k in db.execute("SELECT profile,split,count(*),sum(nstarts) FROM episodes GROUP BY profile,split")]
    finally:
        db.close()
        for c in connections.values():
            c.close()
    for path in [CONFIG_ROOT / "dataset.yaml", *(CONFIG_ROOT / "embodiments").glob("*.yaml"),
                 *(CONFIG_ROOT / "sources").glob("*.yaml")]:
        inputs[str(path)] = sha256(path)
    manifest = dict(schema=SCHEMA, purpose="audit" if limit_per_group else "training_candidate",
                    split_package=str(old), inputs=inputs, input_catalogs=input_catalogs,
                    profiles=profiles, groups=summary, totals=dict(totals),
                    exclusions=exclusions,
                    normalization=normalization_policy(),
                    temporal=dict(stride=1, states=33, actions=32, video_indices=list(range(0, 33, 4))),
                    sampler=dict(training_block_size=4, quota="native_legal_starts", family="real"),
                    gates={"abc130k": "command/EEF timing mapping not validated"},
                    index_sha256=sha256(output / "episodes.sqlite"))
    write_json(output / "manifest.json", manifest)
    write_json(output / "_INDEX_COMPLETE", dict(manifest_sha256=sha256(output / "manifest.json")))
    return manifest


class LocalRobotPackage:
    def __init__(self, root, *, audit=False, verify_index=True):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        from wam.runtime_contracts import recorded_manifest_id, require_file
        self.fingerprint = recorded_manifest_id(self.root)
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("unsupported local package")
        if not audit and self.manifest["purpose"] != "training_candidate":
            raise ValueError("audit-only package cannot train")
        require_file(self.root / "episodes.sqlite")
        self._db, self._pid = None, None

    @property
    def db(self):
        if self._pid != os.getpid():
            if self._db is not None:
                self._db.close()
            self._db = readonly_db(self.root / "episodes.sqlite")
            self._pid = os.getpid()
        return self._db

    def _root(self, root_id):
        return unpack(self.db.execute("SELECT payload FROM roots WHERE id=?", (root_id,)).fetchone()[0])

    @lru_cache(maxsize=32)
    def episode(self, episode_id):
        row = self.db.execute("SELECT profile,split,unit,ranges,record,root_id FROM episodes WHERE id=?", (int(episode_id),)).fetchone()
        if row is None:
            raise KeyError(episode_id)
        profile, split, unit, ranges, record, rid = row
        root = self._root(rid)
        return dict(id=int(episode_id), profile=profile, split=split, unit=unit,
                    ranges=unpack(ranges), record=unpack(record), root=root)

    @lru_cache(maxsize=128)
    def ranges(self, episode_id):
        row = self.db.execute("SELECT ranges FROM episodes WHERE id=?", (int(episode_id),)).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return unpack(row[0])

    def __getstate__(self):
        return dict(self.__dict__, _db=None, _pid=None)

    def close(self):
        if self._db is not None:
            self._db.close()
        self._db, self._pid = None, None
