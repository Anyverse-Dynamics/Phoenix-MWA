"""Additive raw-state index. Completion is not media or training admission."""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from wam.resources import resource_path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wam.datasets.local_robot_contracts import digest, eligible_group, native_ranges, normalization_policy, sha256
from wam.datasets.local_robot_package import (
    DEFAULT_SPLIT, UR_SINGLETON_EXCLUSIONS, LocalRobotPackage,
    pack, read_json, readonly_db, unpack, write_json,
)

SCHEMA = "phoenix.local_robot80.raw_bi.v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "artifacts/local_robot_raw_bi/full_20261002_v1"
DEFAULT_FR3_SPLIT = str(Path(__file__).resolve().parents[3] / "resources/data_manifests/franka_split")
HISTORICAL_ADDITIONS = frozenset({
    "cosmos3_droid", "galaxea_upperbody", "abc130k", "openneo_aloha",
    "openneo_arx5", "openneo_arx5_single", "openneo_flexiv", "openneo_ur",
    "arx456_xt_motion_dedup_20260916", "arx_machine_named_20260916",
    "arx_incremental920_20260917",
})
EXCLUDED_SOURCES = frozenset({
    "agibot_dex6", "bridge_data2", "robocasa365", "robodojo_real",
    "robodojo_sim", "robotwin", "generic_pick",
})
_BASE = str(Path(__file__).resolve().parents[3] / "data/grasp") + "/"
FRESH_ROOTS = {
    "consolidated_pickup": _BASE + "consolidated_pickup_20260926/lerobot",
    "random_pick_and_place": _BASE + "foundation_tasks_20260926/shards/random_pick_and_place/*",
    "object_to_container": _BASE + "foundation_tasks_20260926/shards/object_to_container/*",
    "relational_pick_and_place": _BASE + "foundation_tasks_20260926/shards/relational_pick_and_place/*",
    "stack_blocks": _BASE + "stack_blocks_collect1000_20260927/workers/*/shards/episode_*",
    "cover_blocks": _BASE + "cover_blocks_1000_1020_20260927/workers/*/shards/episode_*",
    "arrange_largest_number": _BASE + "arrange_largest_number_1000_20260927/workers/*/shards/episode_*",
    "task_generalization": str(Path(__file__).resolve().parents[3] / "data/task_generalization"),
}
IDENTITY_COLUMNS = ("index", "frame_index", "episode_index", "task_index", "timestamp")
REQUIRED_SCOPE = {
    f"D{i:02d}": tuple("raw_bi/" + group for group in groups)
    for i, groups in enumerate([
        ("a1/franka",), ("a1/lift2",), ("a1/split_aloha",), ("a1/genie1",),
        ("agibot/a2d",), ("RoboCOIN/Cobot_Magic",), ("RoboCOIN/Split_aloha",),
        ("RoboCOIN/R1_Lite",), ("RoboCOIN/RMC-AIDA-L",),
        ("robomind2/franka",), ("robomind2/ur5",), ("cosmos3_droid/panda",),
        ("galaxea_upperbody/r1lite", "galaxea_upperbody/r1pro"), ("abc130k/yam_bimanual",),
        ("openneo_aloha/aloha",), ("openneo_arx5/arx5",), ("openneo_arx5_single/arx5_single",),
        ("openneo_flexiv/flexiv",), ("openneo_ur/ur",),
        tuple("arx456_xt_motion_dedup_20260916/" + b for b in ("arx4", "arx5", "arx6")),
        tuple("arx_machine_named_20260916/" + b for b in ("arx4", "arx5", "arx6")),
        tuple("arx_incremental920_20260917/" + b for b in ("arx4", "arx5", "arx6")),
        ("consolidated_pickup/arx_x5",), ("random_pick_and_place/arx_x5",),
        ("object_to_container/arx_x5",), ("relational_pick_and_place/arx_x5",),
        ("stack_blocks/arx_x5",), ("cover_blocks/arx_x5",), ("arrange_largest_number/arx_x5",),
        ("task_generalization/franka_fr3",),
    ], 1)
}


def required_scope_report(profiles, groups, reviewed_exclusions=None):
    reviewed_exclusions = reviewed_exclusions or {}
    required = {group for values in REQUIRED_SCOPE.values() for group in values}
    if set(reviewed_exclusions) - required or any(not r for r in reviewed_exclusions.values()):
        raise ValueError("invalid reviewed scope exclusions")
    counts, profile_counts = {}, Counter()
    for row in groups:
        group = profiles[row["profile"]]["stats_group"]
        count = counts.setdefault(group, Counter())
        count[row["split"] + "_episodes"] += row["episodes"]
        count[row["split"] + "_legal_starts"] += row["legal_starts"]
        profile_counts[row["profile"]] += row["episodes"]
    coverage = []
    for entry, expected in REQUIRED_SCOPE.items():
        for group in expected:
            count = counts.get(group, {})
            status = "present" if count.get("train_episodes", 0) > 0 and count.get("train_legal_starts", 0) > 0 else (
                "reviewed_exclusion" if group in reviewed_exclusions else "missing_or_no_train_windows")
            coverage.append(dict(entry=entry, stats_group=group, status=status,
                                 counts=dict(count), reason=reviewed_exclusions.get(group)))
    empty = sorted(p for p in profiles if not profile_counts[p])
    unexpected = sorted(set(counts) - required)
    return dict(scope_passed=all(r["status"] != "missing_or_no_train_windows" for r in coverage)
                and not empty and not unexpected, required_entries=30, required_stats_groups=len(required),
                coverage=coverage, empty_profiles=empty, unexpected_stats_groups=unexpected)


def resolve_fresh_identity(db, detail):
    """Check both identities before merging a content-equivalent variant."""
    original_uid = detail["physical_episode_uid"]
    original_unit = "physical:" + original_uid
    alias = "observation:" + detail["observation_sha256"]
    previous = db.execute("SELECT unit FROM identity_aliases WHERE alias=?", (alias,)).fetchone()
    unit = previous[0] if previous else original_unit
    known = {}
    for candidate in sorted({original_unit, unit}):
        found = db.execute("SELECT split FROM units WHERE id=?", (candidate,)).fetchone()
        if found:
            known[candidate] = found[0]
    splits = set(known.values())
    if detail.get("existing_split"):
        splits.add(detail["existing_split"])
    if len(splits) > 1:
        raise ValueError("fresh_identity_split_conflict: " + json.dumps(dict(
            original_unit=original_unit, alias_unit=unit, known_splits=known,
            existing_split=detail.get("existing_split")), sort_keys=True))
    uid = unit.removeprefix("physical:")
    split = next(iter(splits)) if splits else physical_split(uid)
    return dict(physical_episode_uid=uid, original_physical_episode_uid=original_uid,
                unit=unit, original_unit=original_unit, alias=alias, split=split,
                split_policy="existing" if splits else "hash1pct",
                merge_provenance=dict(original_unit=original_unit, canonical_unit=unit,
                                      known_splits=known, observation_sha256=detail["observation_sha256"],
                                      merged=original_unit != unit))


def selected_group(group):
    return (group["source"] not in EXCLUDED_SOURCES and
            (eligible_group(group) or group["source"] in HISTORICAL_ADDITIONS))


def physical_split(uid):
    """Stable exact 1/100 hash buckets, independent of traversal/root names."""
    return "validation" if int(digest(["raw_bi_physical_split_v1", uid]), 16) % 100 == 0 else "train"


def explicit_split(info, episode):
    value = episode.get("split")
    if value is not None:
        if value not in ("train", "val", "validation"):
            raise ValueError("unsupported explicit episode split: " + str(value))
        return "validation" if value == "val" else value
    splits = info.get("splits", {})
    # A standard LeRobot all-train export is not a physical holdout release.
    heldout = [v for k, v in splits.items() if k in ("val", "validation")]
    if not any(isinstance(v, str) and v.count(":") == 1 and
               int(v.split(":")[1]) > int(v.split(":")[0]) for v in heldout):
        return None
    found = []
    for name, interval in splits.items():
        if name not in ("train", "val", "validation"):
            continue
        if not isinstance(interval, str) or interval.count(":") != 1:
            raise ValueError("unsupported metadata split interval")
        a, b = map(int, interval.split(":"))
        if a <= int(episode["episode_index"]) < b:
            found.append("validation" if name == "val" else name)
    if len(found) != 1:
        raise ValueError("metadata split does not assign episode exactly once")
    return found[0]


def trajectory_fingerprint(values):
    """All observed joint/gripper rows, no camera, instruction or path identity."""
    values = np.asarray(values, dtype="<f8").copy()
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("non-finite or non-matrix observation identity")
    values[values == 0] = 0  # Normalize signed zero across equivalent exports.
    h = hashlib.sha256(json.dumps(list(values.shape)).encode())
    h.update(values.tobytes(order="C"))
    return h.hexdigest()


def normalize_spans(record):
    """Supply missing structural aliases without broadening historical spans."""
    row = dict(record)
    spans = row.get("sampling_spans")
    if not spans:
        raise ValueError("missing retained sampling_spans")
    if "allowed_frame_ranges" not in row:
        row["allowed_frame_ranges"] = [dict(start=s["start"], end_exclusive=s["end_exclusive"]) for s in spans]
    if "task_spans" not in row:
        row["task_spans"] = [dict(s) for s in spans]
    row.setdefault("dataset_to_index", int(row["dataset_from_index"]) + int(row["length"]))
    return row


def raw_normalization_policy(profiles):
    policy = copy.deepcopy(normalization_policy())
    groups = {p["stats_group"].removeprefix("raw_bi/") for p in profiles.values()}
    mappings = []
    for kind, rules in policy["stats"].get("minimum_widths", {}).items():
        mappings.append(("minimum_widths/" + kind, rules))
    for kind, config in policy.get("post_normalize_transforms", {}).items():
        mappings.append(("post_normalize_transforms/" + kind, config.get("by_stats_group", {})))
    audit = []
    for name, rules in mappings:
        before = copy.deepcopy(rules)
        rules.clear()
        for group, value in before.items():
            if group in groups:
                target = "raw_bi/" + group
                rules[target] = value
                audit.append(dict(policy=name, legacy_group=group, raw_group=target))
        expected = {"raw_bi/" + g: v for g, v in before.items() if g in groups}
        if rules != expected:
            raise AssertionError("legacy normalization rule not mapped exactly: " + name)
    return policy, audit


def _metadata(root):
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    episodes = []
    if files:
        for path in files:
            with pq.ParquetFile(path) as file:
                columns = [c for c in file.schema_arrow.names if not c.startswith("stats/")]
                episodes.extend(file.read(columns=columns, use_threads=False).to_pylist())
    elif (root / "meta/episodes.jsonl").is_file():
        with (root / "meta/episodes.jsonl").open() as file:
            episodes = [json.loads(line) for line in file if line.strip()]
    else:
        raise ValueError("missing episode metadata")
    return episodes


def _array(column):
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    n = len(array)
    if pa.types.is_fixed_size_list(array.type):
        return np.asarray(array.flatten()).reshape(n, -1)
    return np.asarray(array.to_pylist())


def _fresh_details(root, info, episodes, tables):
    table = pa.concat_tables(tables)
    arrays = {k: _array(table[k]) for k in IDENTITY_COLUMNS}
    states = _array(table["observation.state"])
    provenance_path = root / "meta/source_manifest.json"
    provenance = {}
    if provenance_path.is_file():
        provenance = {int(e["episode_index"]): e for e in read_json(provenance_path).get("entries", [])}
    details = {}
    for episode in episodes:
        eid = int(episode["episode_index"])
        a, b = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        n = int(episode["length"])
        try:
            if (a < 0 or b > len(table) or b - a != n or
                    not np.array_equal(arrays["index"][a:b], np.arange(a, b)) or
                    not np.array_equal(arrays["frame_index"][a:b], np.arange(n)) or
                    not np.all(arrays["episode_index"][a:b] == eid)):
                raise ValueError("raw episode/frame/global-index mismatch")
            times = arrays["timestamp"][a:b]
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("timestamps not finite strictly increasing")
            tasks = arrays["task_index"][a:b]
            edges = np.r_[0, np.flatnonzero(np.diff(tasks)) + 1, n]
            spans = [dict(start=int(x), end_exclusive=int(y), task_index=int(tasks[x]))
                     for x, y in zip(edges[:-1], edges[1:])]
            content = trajectory_fingerprint(states[a:b])
            split = explicit_split(info, episode)
            if "dataset_split" in table.column_names:
                values = set(table["dataset_split"].slice(a, n).to_pylist())
                if len(values) != 1 or next(iter(values)) not in {"train", "val", "validation"}:
                    raise ValueError("unsupported or mixed per-frame dataset_split: " + repr(values))
                frame_split = next(iter(values))
                frame_split = "validation" if frame_split == "val" else frame_split
                if split and split != frame_split:
                    raise ValueError("metadata/per-frame split conflict")
                split = frame_split
            source = provenance.get(eid, {})
            if source and (int(source["length"]) != n or int(source["dataset_from_index"]) != a):
                raise ValueError("source provenance episode mapping mismatch")
            uid = episode.get("physical_episode_uid") or source.get("physical_episode_uid")
            if not uid and source.get("trajectory_sha256"):
                uid = digest(["source_trajectory_sha256", source["trajectory_sha256"]])
            details[str(eid)] = dict(sampling_spans=spans, observation_sha256=content,
                physical_episode_uid=uid or digest(["raw_observation_trajectory_v1", content]),
                identity_basis="source_provenance" if uid else "full_observed_trajectory_sha256",
                source_provenance=source, existing_split=split)
        except (ValueError, KeyError, TypeError) as exc:
            details[str(eid)] = dict(error=str(exc))
    return details


def inspect_raw_root(spec, resolver, projector_class):
    """Check every Parquet footer and actual dimensions per schema per root."""
    root = Path(spec["path"])
    result = dict(path=str(root), source=spec["source"], status="quarantined")
    try:
        info = read_json(root / "meta/info.json")
        result.update(info_sha256=sha256(root / "meta/info.json"),
                      raw_total_episodes=info.get("total_episodes"), raw_total_frames=info.get("total_frames"))
        group, record = spec["group"], dict(spec["record"])
        record["raw_info"] = info
        profile = resolver(group, record)
        columns = sorted(set(projector_class(profile).columns))
        if any("eef_xforward" in c or ".x_forward" in c for c in columns):
            raise ValueError("derived xforward field in raw contract")
        missing = set(columns) - set(info["features"])
        if missing:
            raise ValueError("missing raw metadata fields: " + repr(sorted(missing)))
        if float(info["fps"]) <= 0 or float(info["fps"]) != float(record["fps"]):
            raise ValueError("raw metadata FPS/profile mismatch")
        episodes = _metadata(root)
        if len(episodes) != int(info["total_episodes"]):
            raise ValueError("episode metadata count/info mismatch")
        metadata = {}
        for e in episodes:
            key = str(int(e["episode_index"]))
            if key in metadata:
                raise ValueError("duplicate metadata episode index")
            metadata[key] = {k: e[k] for k in ("episode_index", "length", "dataset_from_index", "dataset_to_index")}
        parts, files, schemas, fresh_tables = [], {}, {}, []
        offset = 0
        paths = sorted((root / "data").rglob("*.parquet"))
        if not paths:
            raise ValueError("no raw Parquet files")
        for path in paths:
            stat = path.stat()
            files[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            with pq.ParquetFile(path) as file:
                missing = set(columns + list(IDENTITY_COLUMNS)) - set(file.schema_arrow.names)
                if missing:
                    raise ValueError(f"missing Parquet fields {path}: {sorted(missing)}")
                signature = digest([(k, str(file.schema_arrow.field(k).type)) for k in columns])
                if signature not in schemas:
                    batch = next(file.iter_batches(batch_size=1, columns=columns, use_threads=False), None)
                    if batch is None:
                        raise ValueError("empty raw Parquet file")
                    dimensions = {}
                    for key in columns:
                        values = np.asarray(batch.column(key).to_pylist(), dtype=np.float64)
                        width = int(values.size)
                        expected = int(np.prod(info["features"][key].get("shape", [1])))
                        if width != expected:
                            raise ValueError(f"raw shape mismatch {key}: {width} != {expected}")
                        dimensions[key] = width
                    schemas[signature] = dict(dimensions=dimensions, sample_file=str(path))
                for i in range(file.metadata.num_row_groups):
                    n = file.metadata.row_group(i).num_rows
                    parts.append(dict(path=str(path), row_group=i, start=offset, end=offset + n))
                    offset += n
                if spec["fresh"]:
                    identity_columns = list(IDENTITY_COLUMNS) + ["observation.state"]
                    if "dataset_split" in file.schema_arrow.names:
                        identity_columns.append("dataset_split")
                    fresh_tables.append(file.read(columns=identity_columns, use_threads=False))
        if offset != int(info["total_frames"]):
            raise ValueError(f"raw total frame mismatch: {offset} != {info['total_frames']}")
        cursor = 0
        for e in sorted(metadata.values(), key=lambda x: int(x["dataset_from_index"])):
            a, b = int(e["dataset_from_index"]), int(e["dataset_to_index"])
            if a != cursor or b - a != int(e["length"]) or b <= a:
                raise ValueError("metadata ranges overlap, gap, or length mismatch")
            cursor = b
        if cursor != offset:
            raise ValueError("metadata/raw frame coverage mismatch")
        payload = dict(root=str(root), row_groups=parts, files=files, rows=offset,
                       info_sha256=result["info_sha256"], raw_schema=schemas)
        result.update(status="checked", profile=profile, payload=payload, metadata=metadata,
                      schema_variants=len(schemas), parquet_files=len(paths), required_columns=columns)
        if spec["fresh"]:
            result.update(episodes=episodes, info=info,
                          details=_fresh_details(root, info, episodes, fresh_tables))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, pa.ArrowException) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _fresh_views(root, info, episode):
    views = []
    for key, feature in info["features"].items():
        if feature.get("dtype") != "video":
            continue
        prefix = "videos/" + key + "/"
        try:
            path = root / info["video_path"].format(video_key=key,
                chunk_index=episode[prefix + "chunk_index"], file_index=episode[prefix + "file_index"])
            views.append(dict(key=key, path=str(path), source_root=str(root),
                start_s=episode[prefix + "from_timestamp"], end_s=episode[prefix + "to_timestamp"],
                fps=info["fps"], available=path.is_file()))
        except KeyError:
            views.append(dict(key=key, available=False, reason="missing_video_metadata"))
    return views


def _fresh_record(source, root, info, episode, detail):
    row = dict(episode)
    spans = detail["sampling_spans"]
    row.update(source_id=source, source_root=str(root), robot_type=info["robot_type"],
        root_id=digest(["raw_root_v1", str(root)]),
        sample_id=digest(["raw_sample_v1", str(root), int(episode["episode_index"])]),
        fps=float(info["fps"]), physical_episode_uid=detail["physical_episode_uid"],
        observation_sha256=detail["observation_sha256"], identity_basis=detail["identity_basis"],
        source_provenance=detail["source_provenance"], has_action=True, eligibility="eligible",
        allowed_frame_ranges=[dict(start=0, end_exclusive=int(episode["length"]))],
        task_spans=spans, sampling_spans=spans, t5_task_bindings=[],
        selected_views=_fresh_views(root, info, episode), cleaning_policy="source_export_retained_rows")
    row["original_physical_episode_uid"] = detail.get("original_physical_episode_uid", detail["physical_episode_uid"])
    row["identity_merge_provenance"] = detail.get("merge_provenance", {})
    return row


def load_fr3_split(package, inputs):
    """Verify the existing task-generalization physical-path holdout release."""
    package = Path(package)
    marker = read_json(package / "_FRANKA_DATA_COMPLETE.json")
    for name in ("manifest.json", "episodes.sqlite"):
        actual = sha256(package / name)
        if marker["files"][name] != actual:
            raise ValueError("FR3 split release hash mismatch: " + name)
        inputs[str(package / name)] = actual
    db = readonly_db(package / "episodes.sqlite")
    result = {}
    try:
        for episode, split, unit, blob in db.execute(
                "SELECT episode,split,unit,payload FROM episodes WHERE source='task_generalization'"):
            if not unit.startswith("trajectory:") or split not in {"train", "validation"}:
                raise ValueError("invalid FR3 physical split identity")
            row = unpack(blob)
            if row["episode_index"] != episode or row["split_unit_id"] != unit or row["split"] != split:
                raise ValueError("FR3 split row identity mismatch")
            path = unit.removeprefix("trajectory:")
            if path in result and result[path]["split"] != split:
                raise ValueError("FR3 physical split leakage")
            result[path] = dict(split=split, unit=unit, record=row)
    finally:
        db.close()
    if not result:
        raise ValueError("FR3 historical split empty")
    return result


def apply_fr3_split(episode, detail, crosswalk):
    source = detail.get("source_provenance", {}).get("source")
    if source not in crosswalk:
        raise ValueError("FR3 source physical path absent from historical split")
    old = crosswalk[source]
    if any(int(old["record"][k]) != int(episode[k]) for k in
           ("episode_index", "length", "dataset_from_index")):
        raise ValueError("FR3 source path/episode crosswalk mismatch")
    detail.update(existing_split=old["split"], physical_episode_uid=old["unit"],
                  identity_basis="inherited_franka5_physical_path",
                  sampling_spans=old["record"]["sampling_spans"])


def _create_db(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript("""
        CREATE TABLE episodes(id INTEGER PRIMARY KEY, profile TEXT NOT NULL,
          split TEXT NOT NULL, unit TEXT NOT NULL, nstarts INTEGER NOT NULL,
          ranges BLOB NOT NULL, record BLOB NOT NULL, root_id TEXT NOT NULL,
          old_group TEXT NOT NULL, old_ordinal INTEGER NOT NULL,
          record_fingerprint TEXT NOT NULL UNIQUE, UNIQUE(old_group,old_ordinal));
        CREATE TABLE roots(id TEXT PRIMARY KEY,payload BLOB NOT NULL);
        CREATE TABLE units(id TEXT PRIMARY KEY,split TEXT NOT NULL);
        CREATE TABLE quarantine(id INTEGER PRIMARY KEY,source TEXT,root TEXT,
          old_group TEXT,old_ordinal INTEGER,split TEXT,reason TEXT NOT NULL,record BLOB);
        CREATE TABLE root_audits(id TEXT PRIMARY KEY,source TEXT,path TEXT,status TEXT,payload BLOB);
        CREATE TABLE incoming(id INTEGER PRIMARY KEY,source TEXT,old_group TEXT,
          old_ordinal INTEGER,split TEXT,root_id TEXT,record BLOB);
        CREATE TABLE root_checks(id TEXT PRIMARY KEY,payload BLOB NOT NULL);
        CREATE TABLE identity_aliases(alias TEXT PRIMARY KEY,unit TEXT NOT NULL);
        CREATE INDEX episodes_profile_split ON episodes(profile,split,id);
        CREATE INDEX episodes_unit ON episodes(unit);
    """)
    return db


def prepare_package(output=DEFAULT_OUTPUT, *, split_package=DEFAULT_SPLIT,
                    fresh_roots=None, workers=16, resolver=None, projector_class=None,
                    fr3_split_package=DEFAULT_FR3_SPLIT, resume=False):
    """Full historical membership and fresh roots, with auditable quarantines."""
    if resolver is None or projector_class is None:
        from wam.datasets.local_robot_raw_bi_contracts import resolve_profile, RawBIProjector
        resolver = resolver or resolve_profile
        projector_class = projector_class or RawBIProjector
    output, old = Path(output).resolve(), Path(split_package).resolve()
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume and ((output / "_INDEX_COMPLETE").exists() or not (output / "_INDEX_FAILED").is_file()):
        raise ValueError("resume requires a failed, unpublished index")
    if workers < 1:
        raise ValueError("workers must be positive")
    fresh_roots = FRESH_ROOTS if fresh_roots is None else fresh_roots
    if set(fresh_roots) - set(FRESH_ROOTS):
        raise ValueError("unselected fresh source requested")
    manifest = read_json(old / "manifest.json")
    marker = read_json(old / "_SPLIT_COMPLETE")
    inputs = {}
    for name in ("manifest.json", "membership.npz"):
        inputs[str(old / name)] = sha256(old / name)
        if inputs[str(old / name)] != marker["files"][name]:
            raise ValueError("historical split identity mismatch: " + name)
    fr3_split = load_fr3_split(fr3_split_package, inputs) if "task_generalization" in fresh_roots else {}
    output.mkdir(parents=True, exist_ok=resume)
    db = sqlite3.connect(output / "episodes.sqlite") if resume else _create_db(output / "episodes.sqlite")
    if resume and (db.execute("SELECT count(*) FROM root_checks").fetchone()[0] or
                   db.execute("SELECT count(*) FROM episodes").fetchone()[0]):
        db.close()
        raise ValueError("resume currently requires import-stage failure; preserve later-stage artifact")
    started = time.monotonic()
    totals, profiles, root_specs, groups, excluded, catalogs = Counter(), {}, {}, {}, [], {}
    source_coverage = {}
    fresh_split_observations = {}
    resumed = {}
    if resume:
        old_groups = {g["id"]: g for g in manifest["groups"]}
        for source, gid, ordinal, split, rid, blob in db.execute(
                "SELECT source,old_group,old_ordinal,split,root_id,record FROM incoming ORDER BY id"):
            resumed.setdefault(gid, {})[ordinal] = split
            totals["historical_imported"] += 1
            source_coverage.setdefault(source, Counter())[split + "_historical_episodes"] += 1
            if rid not in root_specs:
                row = unpack(blob)
                row.setdefault("robot_type", row.get("machine", old_groups[gid]["signature"][1]))
                root_specs[rid] = dict(path=str(Path(row["source_root"]).resolve()), source=source,
                                       group=old_groups[gid], record=row, fresh=False)
        failure = read_json(output / "_INDEX_FAILED")
        write_json(output / "resumed_import.json", dict(previous_failure=failure, inputs=inputs,
                                                       preserved_records=totals["historical_imported"]))
        (output / "_INDEX_FAILED").unlink()

    def progress(stage, **extra):
        report = dict(stage=stage, elapsed_seconds=time.monotonic() - started,
                      totals=dict(totals), **extra)
        write_json(output / "index_progress.json", report)
        print("[raw_bi] " + json.dumps(report, sort_keys=True), flush=True)

    def quarantine(source, path, group, ordinal, split, reason, record=None):
        db.execute("INSERT INTO quarantine VALUES(NULL,?,?,?,?,?,?,?)",
                   (source, path, group, ordinal, split, reason, pack(record) if record else None))
        totals["quarantined_episodes" if record else "quarantined_roots"] += 1

    def bind_unit(uid, split):
        unit = "physical:" + uid
        found = db.execute("SELECT split FROM units WHERE id=?", (unit,)).fetchone()
        if found and found[0] != split:
            raise ValueError("physical trajectory crosses historical train/validation: " + uid)
        db.execute("INSERT OR IGNORE INTO units VALUES(?,?)", (unit, split))
        return unit

    try:
        with np.load(old / "membership.npz", allow_pickle=False) as members:
            for group in manifest["groups"]:
                if not selected_group(group):
                    excluded.append(dict(group=group["id"], source=group["source"],
                        reason="not_selected_source_or_embodiment", splits=group["splits"]))
                    continue
                source, gid = group["source"], group["id"]
                groups[gid] = group
                arrays = [members[s + "__" + gid] for s in ("train", "validation")]
                for split, a in zip(("train", "validation"), arrays):
                    if (a.dtype != np.int64 or a.ndim != 1 or len(a) != group["splits"][split] or
                        (len(a) and (a[0] < 0 or a[-1] >= group["parent_episodes"] or np.any(np.diff(a) <= 0)))):
                        raise ValueError("historical membership shape/count/order mismatch: " + gid)
                if np.intersect1d(*arrays).size:
                    raise ValueError("historical split ordinal leakage")
                assignments = {int(i): split for split, a in zip(("train", "validation"), arrays) for i in a}
                if any(assignments.get(i) != s for i, s in resumed.get(gid, {}).items()):
                    raise ValueError("resumed historical membership drift")
                catroot = Path(manifest["catalogs"][group["catalog"]])
                if str(catroot) not in catalogs:
                    cat = read_json(catroot / "catalog.json")
                    h = sha256(catroot / "catalog.json")
                    if (read_json(catroot / "_CATALOG_COMPLETE")["sha256"] != h or
                            manifest["inputs"].get(str(catroot / "catalog.json")) != h):
                        raise ValueError("historical catalog release mismatch")
                    inputs[str(catroot / "catalog.json")] = h
                    catalogs[str(catroot)] = cat
                desc = catalogs[str(catroot)]["sources"][source]
                catalog_path = catroot / desc["file"]
                stat = catalog_path.stat()
                if stat.st_size != desc["bytes"] or ("mtime_ns" in desc and stat.st_mtime_ns != desc["mtime_ns"]):
                    raise ValueError("historical catalog identity drift")
                coverage = source_coverage.setdefault(source, Counter())
                found = set()
                c = readonly_db(catalog_path)
                try:
                    for ordinal, blob in c.execute("SELECT ordinal,payload FROM episodes WHERE group_id=? ORDER BY ordinal", (gid,)):
                        if ordinal not in assignments:
                            coverage["catalog_records_outside_historical_membership"] += 1
                            continue
                        found.add(ordinal)
                        split = assignments[ordinal]
                        if ordinal in resumed.get(gid, {}):
                            continue
                        row = unpack(blob)
                        row.setdefault("robot_type", row.get("machine", group["signature"][1]))
                        uid = row.get("physical_episode_uid") or row.get("source_trajectory_uid")
                        if not uid:
                            quarantine(source, row.get("source_root", ""), gid, ordinal, split,
                                       "missing_historical_physical_identity", row)
                            continue
                        if gid in UR_SINGLETON_EXCLUSIONS:
                            if uid != UR_SINGLETON_EXCLUSIONS[gid] or ordinal != 0 or split != "train":
                                raise ValueError("approved excluded UR identity changed")
                            excluded.append(dict(group=gid, ordinal=ordinal, physical_episode_uid=uid,
                                reason="preserved_two_UR_exclusion"))
                            totals["excluded_ur_episodes"] += 1
                            continue
                        if uid in UR_SINGLETON_EXCLUSIONS.values():
                            raise ValueError("excluded UR appears in another group")
                        if row["source_id"] != source or row["robot_type"] != group["signature"][1]:
                            raise ValueError("historical record/group identity mismatch")
                        row["physical_episode_uid"] = uid
                        bind_unit(uid, split)
                        path = str(Path(row["source_root"]).resolve())
                        rid = digest(["raw_root_v1", path, source])
                        if rid not in root_specs:
                            root_specs[rid] = dict(path=path, source=source, group=group, record=row, fresh=False)
                        db.execute("INSERT INTO incoming VALUES(NULL,?,?,?,?,?,?)",
                                   (source, gid, ordinal, split, rid, pack(row)))
                        coverage[split + "_historical_episodes"] += 1
                        totals["historical_imported"] += 1
                        if totals["historical_imported"] % 50000 == 0:
                            db.commit()
                            progress("import_historical", roots=len(root_specs))
                    if found != set(assignments):
                        raise ValueError("historical catalog membership missing: " + gid)
                finally:
                    c.close()
                db.commit()
        for source, pattern in sorted(fresh_roots.items()):
            paths = sorted(Path(p).resolve() for p in glob.glob(str(pattern)) if Path(p).is_dir())
            coverage = source_coverage.setdefault(source, Counter())
            coverage["discovered_roots"] = len(paths)
            if not paths:
                quarantine(source, str(pattern), "fresh:" + source, -1, None, "selected_root_pattern_no_matches")
            for root in paths:
                try:
                    info = read_json(root / "meta/info.json")
                    keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
                    if source == "task_generalization":
                        keys = ["observation.images.franka_head", "observation.images.franka_hand"]
                    group = dict(id="fresh:" + source, source=source,
                        signature=[source, info["robot_type"], "raw_metadata", keys, [], float(info["fps"])])
                    row = dict(source_id=source, source_root=str(root), robot_type=info["robot_type"],
                               fps=float(info["fps"]), selected_views=[dict(key=k) for k in keys])
                    rid = digest(["raw_root_v1", str(root), source])
                    root_specs[rid] = dict(path=str(root), source=source, group=group, record=row, fresh=True)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    quarantine(source, str(root), "fresh:" + source, -1, None, "root_metadata: " + str(exc))
        progress("inspect_raw_roots", roots=len(root_specs))
        items = sorted(root_specs.items())
        def inspect(item):
            rid, spec = item
            return rid, inspect_raw_root(spec, resolver, projector_class)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (rid, check) in enumerate(pool.map(inspect, items), 1):
                summary = {k: v for k, v in check.items() if k not in
                           ("payload", "metadata", "episodes", "info", "details", "profile")}
                db.execute("INSERT INTO root_audits VALUES(?,?,?,?,?)",
                           (rid, check["source"], check["path"], check["status"], pack(summary)))
                db.execute("INSERT INTO root_checks VALUES(?,?)", (rid, pack(check)))
                if check["status"] == "checked":
                    db.execute("INSERT INTO roots VALUES(?,?)", (rid, pack(check["payload"])))
                    profiles[check["profile"]["id"]] = check["profile"]
                    totals["checked_roots"] += 1
                    if root_specs[rid]["fresh"]:
                        labels = fresh_split_observations.setdefault(check["source"], Counter())
                        for detail in check["details"].values():
                            if "error" not in detail:
                                labels[str(detail.get("existing_split"))] += 1
                else:
                    quarantine(check["source"], check["path"], "root", -1, None, check["reason"])
                if i % 200 == 0:
                    db.commit()
                    progress("inspect_raw_roots", roots_done=i, roots=len(items))
        db.commit()

        @lru_cache(maxsize=8)
        def get_check(rid):
            return unpack(db.execute("SELECT payload FROM root_checks WHERE id=?", (rid,)).fetchone()[0])

        def admit(source, gid, ordinal, split, rid, original, provenance):
            totals["candidate_episodes"] += 1
            check = get_check(rid)
            path = check["path"]
            try:
                if check["status"] != "checked":
                    raise ValueError("root_quarantine: " + check["reason"])
                row = normalize_spans(original)
                raw = check["metadata"].get(str(int(row["episode_index"])))
                if raw is None or any(int(raw[k]) != int(row[k]) for k in
                                      ("length", "dataset_from_index", "dataset_to_index")):
                    raise ValueError("historical/raw metadata episode offset mismatch")
                row.setdefault("physical_episode_uid", row.get("source_trajectory_uid"))
                unit = bind_unit(row["physical_episode_uid"], split)
                ranges = native_ranges(row)
                nstarts = sum(b - a for a, b, _ in ranges)
                if not nstarts:
                    raise ValueError("no_33_row_native_window_within_retained_spans")
                profile = check["profile"]
                row["raw_bi_provenance"] = provenance
                row["raw_state_root"] = path
                row["downstream_readiness"] = dict(video="not_audited", text="not_audited", training=False)
                fingerprint = digest(dict(source=source, sample_id=row["sample_id"], unit=unit,
                    profile=profile["id"], split=split, spans=row["sampling_spans"],
                    raw_info_sha256=check["info_sha256"], from_index=row["dataset_from_index"], length=row["length"]))
                row["record_fingerprint"] = fingerprint
                if db.execute("SELECT 1 FROM episodes WHERE record_fingerprint=?", (fingerprint,)).fetchone():
                    raise ValueError("duplicate_record_fingerprint")
                db.execute("INSERT INTO episodes VALUES(NULL,?,?,?,?,?,?,?,?,?,?)",
                    (profile["id"], split, unit, nstarts, pack(ranges), pack(row), rid, gid, ordinal, fingerprint))
                totals[split + "_episodes"] += 1
                totals[split + "_legal_starts"] += nstarts
                source_coverage[source][split + "_admitted_episodes"] += 1
            except (ValueError, KeyError, TypeError) as exc:
                if "crosses historical train/validation" in str(exc):
                    raise
                quarantine(source, path, gid, ordinal, split, str(exc), original)
                source_coverage[source]["quarantined_episodes"] += 1

        progress("admit_historical")
        for _, source, gid, ordinal, split, rid, blob in db.execute("SELECT * FROM incoming ORDER BY id"):
            admit(source, gid, ordinal, split, rid, unpack(blob), dict(kind="historical", group=gid, ordinal=ordinal))
            if totals["candidate_episodes"] % 50000 == 0:
                db.commit()
                progress("admit_historical")
        db.commit()
        progress("admit_fresh")
        # Register known fresh splits before hashing any duplicate variant.
        for rid, spec in items:
            if not spec["fresh"]:
                continue
            check = get_check(rid)
            if check["status"] != "checked":
                continue
            for episode in check["episodes"]:
                detail = check["details"][str(int(episode["episode_index"]))]
                if "error" in detail:
                    continue
                if spec["source"] == "task_generalization":
                    try:
                        apply_fr3_split(episode, detail, fr3_split)
                    except ValueError as exc:
                        detail["error"] = str(exc)
                        continue
                if detail["existing_split"]:
                    unit = bind_unit(detail["physical_episode_uid"], detail["existing_split"])
                    alias = "observation:" + detail["observation_sha256"]
                    previous = db.execute("SELECT unit FROM identity_aliases WHERE alias=?", (alias,)).fetchone()
                    if previous and previous[0] != unit:
                        prior = db.execute("SELECT split FROM units WHERE id=?", previous).fetchone()[0]
                        if prior != detail["existing_split"]:
                            raise ValueError("duplicate fresh observation crosses existing splits")
                    db.execute("INSERT OR IGNORE INTO identity_aliases VALUES(?,?)", (alias, unit))
            db.execute("UPDATE root_checks SET payload=? WHERE id=?", (pack(check), rid))
        get_check.cache_clear()
        for rid, spec in items:
            if not spec["fresh"]:
                continue
            check = get_check(rid)
            source, root = spec["source"], Path(spec["path"])
            if check["status"] != "checked":
                source_coverage[source]["quarantined_root_metadata_episodes"] += int(check.get("raw_total_episodes") or 0)
                continue
            for episode in check["episodes"]:
                eid = int(episode["episode_index"])
                detail = check["details"][str(eid)]
                source_coverage[source]["fresh_metadata_episodes"] += 1
                if "error" in detail:
                    quarantine(source, str(root), "fresh:" + rid, eid, None, detail["error"], episode)
                    continue
                try:
                    identity = resolve_fresh_identity(db, detail)
                except ValueError as exc:
                    quarantine(source, str(root), "fresh:" + rid, eid, detail.get("existing_split"),
                               str(exc), dict(episode, original_identity=detail))
                    source_coverage[source]["quarantined_episodes"] += 1
                    continue
                uid, split = identity["physical_episode_uid"], identity["split"]
                duplicate = bool(db.execute("SELECT 1 FROM episodes WHERE unit=? LIMIT 1", (identity["unit"],)).fetchone())
                if duplicate:
                    totals["fresh_duplicate_trajectory_records"] += 1
                detail.update(identity)
                bind_unit(identity["original_physical_episode_uid"], split)
                unit = bind_unit(uid, split)
                db.execute("INSERT OR IGNORE INTO identity_aliases VALUES(?,?)", (identity["alias"], unit))
                row = _fresh_record(source, root, check["info"], episode, detail)
                admit(source, "fresh:" + rid, eid, split, rid, row,
                      dict(kind="fresh", split_policy=identity["split_policy"], duplicate_observation=duplicate))
            db.commit()
        progress("finalize")
        summaries = [dict(profile=p, split=s, episodes=n, legal_starts=k) for p, s, n, k in db.execute(
            "SELECT profile,split,count(*),sum(nstarts) FROM episodes GROUP BY profile,split ORDER BY profile,split")]
        reasons = [dict(reason=r, entries=n) for r, n in db.execute(
            "SELECT reason,count(*) FROM quarantine GROUP BY reason ORDER BY count(*) DESC,reason")]
        totals["physical_units"] = db.execute("SELECT count(DISTINCT unit) FROM episodes").fetchone()[0]
        if db.execute("SELECT unit FROM episodes GROUP BY unit HAVING count(DISTINCT split)>1").fetchone():
            raise ValueError("physical split leakage")
        get_check.cache_clear()
        db.executescript("DROP TABLE incoming; DROP TABLE root_checks;")
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.execute("PRAGMA journal_mode=DELETE")
        db.close()
        normalization, policy_mapping = raw_normalization_policy(profiles)
        scope = required_scope_report(profiles, summaries)
        result = dict(schema=SCHEMA, purpose="raw_state_index", split_package=str(old), inputs=inputs,
            profiles=profiles, groups=summaries, totals=dict(totals), exclusions=excluded,
            normalization=normalization, normalization_policy_mapping=policy_mapping,
            required_scope=scope, scope_passed=scope["scope_passed"],
            fresh_split_observations={s: dict(c) for s, c in fresh_split_observations.items()},
            source_coverage={s: dict(c) for s, c in sorted(source_coverage.items())},
            quarantine_reasons=reasons, index_sha256=sha256(output / "episodes.sqlite"),
            temporal=dict(stride=1, states=33, actions=32, video_indices=list(range(0, 33, 4))),
            split_policy=dict(historical="unchanged physicalUID train/validation", fresh="existing else SHA256 modulo100 == 0",
                fresh_identity="source provenance plus complete observed joint/gripper trajectory SHA256",
                all_train_export="unpartitioned info.splits is not a holdout; explicit per-frame dataset_split remains authoritative",
                foundation_partition="scene-generated train/test_object/test_scene/test_combination; current selected shards are train-only; unsupported test labels are quarantined"),
            coverage=dict(scope="all selected historical membership plus all discovered fresh metadata",
                raw_schema="all Parquet footers and one shape sample per schema per root",
                numerical_finiteness="full fresh observation identity only; remaining numeric rows delegated to full stats",
                historical_cleaning="original sampling_spans unchanged; absent boundary aliases conservatively derived"),
            readiness=dict(raw_index_complete=True, full_statistics=False, sample_audit=False,
                           video=False, text=False, gpu=False, training=False),
            elapsed_seconds=time.monotonic() - started)
        write_json(output / "manifest.json", result)
        write_json(output / "_INDEX_COMPLETE", dict(manifest_sha256=sha256(output / "manifest.json"), scope="raw_state_index"))
        progress("complete")
        return result
    except BaseException as exc:
        try:
            db.commit()
            db.close()
        except sqlite3.Error:
            pass
        write_json(output / "_INDEX_FAILED", dict(error=f"{type(exc).__name__}: {exc}", totals=dict(totals)))
        raise


class LocalRobotRawBIPackage(LocalRobotPackage):
    """Read-only LocalRobotPackage-compatible access; no old schema acceptance."""
    def __init__(self, root, *, audit=True, verify_index=True):
        self.root = resource_path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        from wam.runtime_contracts import recorded_manifest_id, require_file
        self.fingerprint = recorded_manifest_id(self.root)
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("unsupported raw BI package")
        if not audit:
            raise ValueError("raw-state index alone does not authorize training")
        require_file(self.root / "episodes.sqlite")
        self._db, self._pid = None, None


def audit_completed_index(root):
    """Independently bind scope, original identities and splits to a frozen index."""
    package = LocalRobotRawBIPackage(root)
    destination = package.root / "raw_index_admission_audit.json"
    if destination.exists():
        package.close()
        raise FileExistsError(destination)
    try:
        profiles = package.manifest["profiles"]
        groups = [dict(profile=p, split=s, episodes=n, legal_starts=k) for p, s, n, k in package.db.execute(
            "SELECT profile,split,count(*),sum(nstarts) FROM episodes GROUP BY profile,split ORDER BY profile,split")]
        scope = required_scope_report(profiles, groups)
        if groups != package.manifest["groups"]:
            raise ValueError("manifest/index group mismatch")
        historical = {}
        for gid, ordinal, split in package.db.execute(
                "SELECT old_group,old_ordinal,split FROM episodes WHERE old_group NOT LIKE 'fresh:%' ORDER BY old_group,old_ordinal"):
            historical.setdefault((gid, split), []).append(ordinal)
        old = Path(package.manifest["split_package"])
        old_manifest = read_json(old / "manifest.json")
        split_errors = []
        expected_keys = set()
        with np.load(old / "membership.npz", allow_pickle=False) as members:
            for group in old_manifest["groups"]:
                if not selected_group(group) or group["id"] in UR_SINGLETON_EXCLUSIONS:
                    continue
                for split in ("train", "validation"):
                    key = group["id"], split
                    expected_keys.add(key)
                    actual = np.asarray(historical.get(key, []), dtype=np.int64)
                    expected = members[split + "__" + group["id"]]
                    if not np.array_equal(actual, expected):
                        split_errors.append(dict(group=group["id"], split=split,
                            actual=len(actual), expected=len(expected)))
        unexpected = set(historical) - expected_keys
        if unexpected:
            split_errors.append(dict(unexpected_groups=sorted(unexpected)))
        fr3 = load_fr3_split(DEFAULT_FR3_SPLIT, {})
        fresh_counts, fr3_counts, conflicts, merges = Counter(), Counter(), [], []
        original_uid_metadata_roots = {}
        fresh_roots = {}
        for _, profile, split, unit, blob in package.db.execute(
                "SELECT id,profile,split,unit,record FROM episodes WHERE old_group LIKE 'fresh:%' ORDER BY id"):
            row = unpack(blob)
            source = profiles[profile]["source"]
            fresh_counts[source] += 1
            fresh_roots[row["source_root"]] = source
            original = row.get("original_physical_episode_uid")
            if source == "task_generalization":
                source_path = row["source_provenance"]["source"]
                old_row = fr3.get(source_path)
                if not old_row or split != old_row["split"]:
                    conflicts.append(dict(source=source, episode=row["episode_index"], reason="FR3 old split mismatch"))
                else:
                    original = old_row["unit"]
                fr3_counts[split] += 1
            if original is None:
                original = digest(["raw_observation_trajectory_v1", row["observation_sha256"]])
            origin_unit = "physical:" + original
            alias = package.db.execute("SELECT unit FROM identity_aliases WHERE alias=?",
                ("observation:" + row["observation_sha256"],)).fetchone()
            origin_split = package.db.execute("SELECT split FROM units WHERE id=?", (origin_unit,)).fetchone()
            if alias != (unit,) or (origin_split and origin_split[0] != split):
                conflicts.append(dict(source=source, episode=row["episode_index"],
                    original_unit=origin_unit, canonical_unit=unit, alias=alias,
                    original_split=origin_split, index_split=split))
            if origin_unit != unit:
                merges.append(dict(source=source, episode=row["episode_index"],
                    original_unit=origin_unit, canonical_unit=unit, original_split=origin_split, split=split))
        # For indexes built before original-UID provenance was added, independently
        # verify that no UID supplied in fresh metadata was overwritten unnoticed.
        def metadata_uid_columns(path):
            files = sorted((Path(path) / "meta/episodes").rglob("*.parquet"))
            fields = set()
            for file in files:
                with pq.ParquetFile(file) as parquet:
                    fields.update(k for k in parquet.schema_arrow.names if k == "physical_episode_uid")
            return path, sorted(fields)
        with ThreadPoolExecutor(max_workers=16) as pool:
            for path, fields in pool.map(metadata_uid_columns, sorted(fresh_roots)):
                if fields:
                    original_uid_metadata_roots[path] = fields
        leakage = package.db.execute(
            "SELECT count(*) FROM (SELECT unit FROM episodes GROUP BY unit HAVING count(DISTINCT split)>1)").fetchone()[0]
        passed = (scope["scope_passed"] and not split_errors and not conflicts and not leakage
                  and not original_uid_metadata_roots and not merges and
                  dict(fr3_counts) == {"train": 1421, "validation": 20})
        report = dict(schema="phoenix.raw_bi_index_admission_audit.v1", passed=passed,
            manifest_sha256=package.fingerprint, index_sha256=package.manifest["index_sha256"],
            scope_passed=scope["scope_passed"], required_scope=scope,
            historical_split_errors=split_errors, historical_records=sum(map(len, historical.values())),
            fresh_counts=dict(fresh_counts), fresh_records=sum(fresh_counts.values()),
            fresh_uid_conflicts=conflicts, fresh_distinct_uid_merges=merges,
            original_uid_metadata_roots=original_uid_metadata_roots,
            fr3_splits=dict(fr3_counts), physical_split_leakage_units=leakage,
            verification_scope="full index identity/scope checks; not statistics or sample/video/GPU acceptance")
        from wam.datasets.local_robot_contracts import metadata_content_sha256
        if sha256(package.root / "manifest.json") != metadata_content_sha256(package.fingerprint):
            raise ValueError("manifest changed during admission audit")
        write_json(destination, report)
        if passed:
            write_json(package.root / "_ADMISSION_AUDIT_COMPLETE", dict(
                report_sha256=sha256(destination), manifest_sha256=package.fingerprint,
                index_sha256=package.manifest["index_sha256"], scope_passed=True))
        return report
    finally:
        package.close()


prepare_raw_bi_package = prepare_package
