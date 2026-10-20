"""Freeze complete Phoenix view/window coverage into compact block-four runs.

No training package, normalization, raw data or historical split is changed.
Every non-padding sample occurs once. RH20T alignment uses the complete raw
source.timestamp_ms grid, never guessed metadata offsets or equal local starts.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "phoenix.robot37.multiview_plan.v1"
RH_SOURCE = "rh20t_v6_20261003"
EXPECTED_UNIQUE_SAMPLES = 1_133_257_771


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ViewRange:
    episode_id: int
    variant_id: int
    first: int
    stop: int
    clock_offset: int = 0

    @property
    def view(self):
        return self.episode_id, self.variant_id


@dataclass(frozen=True)
class TailSample:
    episode_id: int
    start: int
    variant_id: int
    group_id: str
    clock_tick: int
    source: str


@dataclass
class Run:
    specs: np.ndarray
    count: int
    pair_ids: np.ndarray
    padding: np.ndarray
    sources: tuple[str, ...]


def _merged_intervals(intervals):
    result = []
    for first, stop in sorted(intervals):
        if first < 0 or first >= stop:
            raise ValueError("Invalid nonempty local-start interval")
        if result and first < result[-1][1]:
            raise ValueError("Coverage contains a duplicate view/window")
        if result and first == result[-1][1]:
            result[-1] = result[-1][0], stop
        else:
            result.append((first, stop))
    return result


def _pair_ids(keys):
    counts = Counter(key for key in keys if key is not None)
    assigned = {}
    result = np.full(4, -1, dtype=np.int8)
    for lane, key in enumerate(keys):
        if key is not None and counts[key] > 1:
            assigned.setdefault(key, len(assigned))
            result[lane] = assigned[key]
    return result


def plan_group(ranges, *, group_id, source):
    """Partition a union of legal view ranges without an intersection-only loss.

    The sweep keeps the active-view set constant in each interval. For m views,
    a gcd(m, 4) pattern covers every view at every start exactly once. Time strata
    are separated inside each run; each stratum advances by one native start.
    """
    ranges = list(ranges)
    expected = defaultdict(list)
    events = defaultdict(list)
    for item in ranges:
        if item.episode_id < 0 or item.variant_id < -1:
            raise ValueError("Invalid episode or explicit view variant")
        expected[item.view].append((item.first, item.stop))
        events[item.clock_offset + item.first].append((1, item))
        events[item.clock_offset + item.stop].append((-1, item))
    expected = {view: _merged_intervals(parts) for view, parts in expected.items()}
    active = {}
    runs, tails = [], []
    points = sorted(events)
    for position, first in enumerate(points):
        for direction, item in sorted(events[first], key=lambda event: event[0]):
            if direction == -1:
                if active.pop(item.view, None) != item:
                    raise ValueError("Unbalanced view interval endpoint")
            elif item.view in active:
                raise ValueError("Overlapping ranges for one view")
            else:
                active[item.view] = item
        if position + 1 == len(points) or not active:
            continue
        stop = points[position + 1]
        views = [active[key] for key in sorted(active)]
        view_count = len(views)
        divisor = math.gcd(view_count, 4)
        time_strata = 4 // divisor
        patterns = view_count // divisor
        count, remaining = divmod(stop - first, time_strata)
        if count:
            for pattern in range(patterns):
                specs, times = [], []
                for lane in range(4):
                    flat = pattern * 4 + lane
                    time_slot, view_index = divmod(flat, view_count)
                    view = views[view_index]
                    tick = first + time_slot * count
                    specs.append(
                        [view.episode_id, tick - view.clock_offset, view.variant_id, 1]
                    )
                    times.append(tick)
                runs.append(
                    Run(
                        specs=np.asarray(specs, dtype=np.int64),
                        count=count,
                        pair_ids=_pair_ids(times),
                        padding=np.zeros(4, dtype=np.uint8),
                        sources=(source,) * 4,
                    )
                )
        for tick in range(first + time_strata * count, stop):
            tails.append(
                [
                    TailSample(
                        view.episode_id,
                        tick - view.clock_offset,
                        view.variant_id,
                        group_id,
                        tick,
                        source,
                    )
                    for view in views
                ]
            )
        if remaining != len(range(first + time_strata * count, stop)):
            raise AssertionError("Pattern remainder arithmetic mismatch")
    if active:
        raise ValueError("Unclosed view interval")
    _verify_group_coverage(expected, runs, tails)
    return runs, tails


def _verify_group_coverage(expected, runs, tails):
    """Independent interval oracle: no window enumeration is required."""
    actual = defaultdict(list)
    for run in runs:
        for episode, first, variant, step in run.specs.tolist():
            if step != 1 or run.count < 1:
                raise ValueError("Main coverage runs must have positive stride one")
            actual[episode, variant].append((first, first + run.count))
    for group in tails:
        for sample in group:
            actual[sample.episode_id, sample.variant_id].append(
                (sample.start, sample.start + 1)
            )
    actual = {view: _merged_intervals(parts) for view, parts in actual.items()}
    if actual != expected:
        raise ValueError("Run plan did not preserve every original view/window")


class TailPacker:
    """Keep tail timestamp groups together where possible; count every padding."""

    def __init__(self, emit):
        self.emit = emit
        self.pending = []

    def add(self, samples):
        samples = list(samples)
        while len(samples) > 4:
            if self.pending:
                self.flush()
            # Avoid a final singleton from a five/nine/...-view timestamp.
            take = 3 if len(samples) % 4 == 1 else 4
            self.pending = samples[:take]
            self.flush()
            samples = samples[take:]
        if not samples:
            return
        if len(self.pending) + len(samples) > 4:
            self.flush()
        self.pending.extend(samples)
        if len(self.pending) == 4:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        real_count = len(self.pending)
        samples = self.pending + [self.pending[-1]] * (4 - real_count)
        padding = np.asarray([0] * real_count + [1] * (4 - real_count), dtype=np.uint8)
        keys = [
            (sample.group_id, sample.clock_tick) if lane < real_count else None
            for lane, sample in enumerate(samples)
        ]
        self.emit(
            Run(
                specs=np.asarray(
                    [
                        [sample.episode_id, sample.start, sample.variant_id, 0]
                        for sample in samples
                    ],
                    dtype=np.int64,
                ),
                count=1,
                pair_ids=_pair_ids(keys),
                padding=padding,
                sources=tuple(sample.source for sample in samples),
            )
        )
        self.pending = []


class ArrayStream:
    """Append bounded chunks, then publish a standard memory-mappable NPY."""

    def __init__(self, directory, name, dtype, row_shape):
        self.path = Path(directory) / name
        self.raw_path = self.path.with_suffix(".building")
        self.dtype = np.dtype(dtype)
        self.row_shape = tuple(row_shape)
        self.count = 0
        self.handle = self.raw_path.open("xb")

    def append(self, value):
        value = np.asarray(value, dtype=self.dtype)
        if value.shape != self.row_shape:
            raise ValueError(f"Unexpected {self.path.name} row shape: {value.shape}")
        self.handle.write(value.tobytes(order="C"))
        self.count += 1

    def finish(self):
        self.handle.close()
        shape = (self.count, *self.row_shape)
        with self.path.open("xb") as destination, self.raw_path.open("rb") as source:
            np.lib.format.write_array_header_2_0(
                destination,
                {"descr": self.dtype.str, "fortran_order": False, "shape": shape},
            )
            for chunk in iter(lambda: source.read(8 << 20), b""):
                destination.write(chunk)
        self.raw_path.unlink()
        return {
            "sha256": sha256(self.path),
            "bytes": self.path.stat().st_size,
            "dtype": str(self.dtype),
            "shape": list(shape),
        }


class PlanWriter:
    def __init__(self, directory):
        self.streams = {
            "run_specs.npy": ArrayStream(directory, "run_specs.npy", "int64", (4, 4)),
            "run_counts.npy": ArrayStream(directory, "run_counts.npy", "int64", ()),
            "run_pair_ids.npy": ArrayStream(directory, "run_pair_ids.npy", "int8", (4,)),
            "run_padding.npy": ArrayStream(directory, "run_padding.npy", "uint8", (4,)),
        }
        self.unique_samples = 0
        self.padded_samples = 0
        self.num_blocks = 0
        self.num_runs = 0
        self.paired_samples = 0
        self.source_samples = Counter()
        self.source_paired_samples = Counter()
        self.tail = TailPacker(self.emit)

    def emit(self, run):
        if run.count < 1 or run.specs.shape != (4, 4):
            raise ValueError("Invalid block run")
        if np.any((run.padding != 0) & (run.padding != 1)):
            raise ValueError("Padding must be a binary per-lane mask")
        self.streams["run_specs.npy"].append(run.specs)
        self.streams["run_counts.npy"].append(run.count)
        self.streams["run_pair_ids.npy"].append(run.pair_ids)
        self.streams["run_padding.npy"].append(run.padding)
        self.num_runs += 1
        self.num_blocks += run.count
        self.unique_samples += run.count * int((run.padding == 0).sum())
        self.padded_samples += run.count * int(run.padding.sum())
        self.paired_samples += run.count * int(((run.pair_ids >= 0) & (run.padding == 0)).sum())
        for source, is_pad, pair_id in zip(
            run.sources, run.padding.tolist(), run.pair_ids.tolist()
        ):
            if not is_pad:
                self.source_samples[source] += run.count
                if pair_id >= 0:
                    self.source_paired_samples[source] += run.count

    def add_group(self, ranges, *, group_id, source):
        runs, tails = plan_group(ranges, group_id=group_id, source=source)
        for run in runs:
            self.emit(run)
        for samples in tails:
            self.tail.add(samples)

    def finish(self):
        self.tail.flush()
        return {name: stream.finish() for name, stream in self.streams.items()}


def integer_timestamps(values):
    """Retain exact millisecond identity and reject silently rounded clocks."""
    values = np.asarray(values)
    if values.ndim != 1 or not len(values):
        raise ValueError("RH20T source clock must be a nonempty scalar column")
    if values.dtype.kind in "iu":
        if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
            raise ValueError("RH20T clock exceeds int64")
        result = values.astype(np.int64, copy=False)
    elif values.dtype.kind == "f":
        if (
            not np.isfinite(values).all()
            or np.any(np.abs(values) > 2**53)
            or not np.array_equal(values, np.rint(values))
        ):
            raise ValueError("RH20T clock is nonfinite, fractional or not exactly representable")
        result = values.astype(np.int64)
    else:
        raise ValueError("RH20T clock must be numeric")
    if np.any(result <= 0):
        raise ValueError("RH20T requires positive absolute source timestamps")
    return result


def validate_clock_piece(state, times, *, expected_rows):
    times = integer_timestamps(times)
    if np.any(np.diff(times) != 100):
        raise ValueError("RH20T source.timestamp_ms is not the complete 100ms grid")
    if state["rows"] and int(times[0]) - state["last_ms"] != 100:
        raise ValueError("RH20T grid is discontinuous across parquet row groups")
    if not state["rows"]:
        state["first_ms"] = int(times[0])
    state["last_ms"] = int(times[-1])
    state["rows"] += len(times)
    if state["rows"] > expected_rows:
        raise ValueError("RH20T source clock has duplicated rows")


def verify_root_clocks(payload, entries):
    """Read every selected episode row, with bounded parquet-row-group memory."""
    import pyarrow.parquet as pq

    entries = sorted(entries, key=lambda entry: entry["record"]["dataset_from_index"])
    starts = [int(entry["record"]["dataset_from_index"]) for entry in entries]
    states = {entry["id"]: {"rows": 0} for entry in entries}
    for previous, following in zip(entries, entries[1:]):
        a, b = previous["record"], following["record"]
        if int(a["dataset_to_index"]) > int(b["dataset_from_index"]):
            raise ValueError("RH20T episode global row ranges overlap")
    columns = ["index", "episode_index", "frame_index", "source.timestamp_ms"]
    for part in sorted(payload["row_groups"], key=lambda value: value["start"]):
        first, stop = int(part["start"]), int(part["end"])
        position = max(0, bisect_right(starts, first) - 1)
        relevant = []
        while position < len(entries) and starts[position] < stop:
            entry = entries[position]
            if int(entry["record"]["dataset_to_index"]) > first:
                relevant.append(entry)
            position += 1
        if not relevant:
            continue
        path = Path(part["path"])
        identity = payload["files"][str(path)]
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
            raise ValueError(f"RH20T numeric file identity changed: {path}")
        with pq.ParquetFile(path) as file:
            table = file.read_row_group(int(part["row_group"]), columns=columns, use_threads=False)
        if len(table) != stop - first or not set(columns).issubset(table.column_names):
            raise ValueError("RH20T row-group coverage/columns changed")
        arrays = {column: table[column].to_numpy() for column in columns}
        if not np.array_equal(arrays["index"], np.arange(first, stop)):
            raise ValueError("RH20T raw global index is not contiguous")
        for entry in relevant:
            record = entry["record"]
            begin, end = int(record["dataset_from_index"]), int(record["dataset_to_index"])
            left, right = max(begin, first), min(end, stop)
            selected = slice(left - first, right - first)
            if (
                end - begin != int(record["length"])
                or not np.all(arrays["episode_index"][selected] == int(record["episode_index"]))
                or not np.array_equal(
                    arrays["frame_index"][selected],
                    np.arange(left - begin, right - begin),
                )
                or states[entry["id"]]["rows"] != left - begin
            ):
                raise ValueError("RH20T raw episode/frame identity or row coverage mismatch")
            validate_clock_piece(
                states[entry["id"]],
                arrays["source.timestamp_ms"][selected],
                expected_rows=int(record["length"]),
            )
    for entry in entries:
        if states[entry["id"]]["rows"] != int(entry["record"]["length"]):
            raise ValueError("RH20T source clock scan did not cover the whole episode")
    return states


def verify_habit_numeric_identity(manifest):
    receipt = manifest.get("habit_resize256")
    aliases = receipt.get("aliases", {}) if receipt else {}
    if len(aliases) != 3:
        raise ValueError("Exactly three frozen HABIT media variants are required")
    for filename, fingerprint in receipt["inputs"].items():
        if sha256(filename) != fingerprint:
            raise ValueError(f"HABIT resize evidence changed: {filename}")
    numeric_roots = []
    for original, resized in sorted(aliases.items()):
        old_data, new_data = Path(original) / "data", Path(resized) / "data"
        if not os.path.samefile(old_data, new_data):
            raise ValueError("HABIT resized variant no longer shares its numeric data")
        numeric_roots.append(new_data.resolve())
    if len(set(numeric_roots)) == 1:
        return {"variants": 3, "mode": "all_variants_share_numeric_directory"}
    # Byte identity is sufficient and avoids interpreting any numerical fields.
    reference = None
    for directory in numeric_roots:
        hashes = {
            str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*.parquet"))
        }
        if not hashes or (reference is not None and hashes != reference):
            raise ValueError("HABIT variants do not have identical complete numeric parquet data")
        reference = hashes
    return {"variants": 3, "mode": "all_numeric_parquet_bytes_equal", "files": len(reference)}


def prepare(
    package_root,
    output,
    *,
    expected_unique_samples=None,
    supervision_review=None,
    include_interndata=False,
):
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    from wam.datasets.phoenix_robot import PhoenixRobotPackage
    from wam.datasets.local_robot_contracts import metadata_content_sha256
    from wam.datasets.local_robot_package import unpack
    from wam.datasets.interndata_multiview import (
        INTERNDATA_MULTIVIEW_POLICY,
        INTERNDATA_SOURCES,
        interndata_camera_variants,
    )

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    package = PhoenixRobotPackage(package_root)
    try:
        manifest = package.manifest
        norm_path = package.root / "norm.json"
        norm, marker = read_json(norm_path), read_json(package.root / "_NORM_COMPLETE")
        norm_hash = marker["norm_sha256"]
        manifest_content_hash = sha256(package.root / "manifest.json")
        norm_content_hash = sha256(norm_path)
        rejected_groups = set()
        supervision_evidence = None
        if supervision_review is not None:
            review = read_json(supervision_review)
            identity = review.get("sampling_contract", {})
            if (
                review.get("classification_complete") is not True
                or review.get("strict_source_equality") is not True
                or identity.get("package_sha256") != package.fingerprint
                or identity.get("norm_sha256") != norm_hash
            ):
                raise ValueError("Incomplete or unrelated paired-supervision review")
            rejected_groups = {tuple(key) for key in review["rejected_phase_groups"]}
            supervision_evidence = {
                "sha256": sha256(supervision_review),
                "reviewed_plan_sha256": review["plan_sha256"],
                "policy": "preserve_all_samples_but_do_not_pair_rejected_phase_groups",
                "rejected_phase_groups": len(rejected_groups),
            }
        required_groups = {profile["stats_group"] for profile in manifest["profiles"].values()}
        if (
            norm["purpose"] != "training"
            or norm["split"] != "train"
            or norm["package_sha256"] != package.fingerprint
            or marker["package_sha256"] != package.fingerprint
            or norm_content_hash != metadata_content_sha256(norm_hash)
            or manifest_content_hash != metadata_content_sha256(package.fingerprint)
            or set(norm["groups"]) != required_groups
            or len(required_groups) != 51
            or (norm["action_dim"], norm["state_dim"]) != (80, 80)
        ):
            raise ValueError("Multiview requires the unchanged full Phoenix training normalization")
        habit_identity = verify_habit_numeric_identity(manifest)
        profiles = manifest["profiles"]
        expected_sources = Counter()
        additional_interndata_samples = 0
        for group in manifest["groups"]:
            if group["split"] == "train":
                source = profiles[group["profile"]]["source"]
                multiplier = 3 if source == "habit" else 1
                if include_interndata and source in INTERNDATA_SOURCES:
                    multiplier = 6
                    additional_interndata_samples += 5 * int(group["legal_starts"])
                expected_sources[source] += multiplier * int(group["legal_starts"])
        if expected_unique_samples is None:
            expected_unique_samples = EXPECTED_UNIQUE_SAMPLES + additional_interndata_samples
        if sum(expected_sources.values()) != expected_unique_samples:
            raise ValueError("Unexpected complete logical view/window population")
        output.mkdir(parents=True)
        write_json(output / "_BUILDING.json", {"schema": SCHEMA, "package_sha256": package.fingerprint})
        writer = PlanWriter(output)
        clock_stream = ArrayStream(output, "episode_clocks.npy", "int64", (2,))
        rh_roots = set()
        ordinary_episodes = 0
        query = (
            "SELECT id,profile,unit,nstarts,ranges,root_id "
            "FROM episodes WHERE split='train' ORDER BY id"
        )
        for episode_id, profile_id, unit, count, blob, root_id in package.db.execute(query):
            source = profiles[profile_id]["source"]
            if source == RH_SOURCE:
                rh_roots.add(root_id)
                continue
            ranges = unpack(blob)
            if sum(stop - first for first, stop, _ in ranges) != count:
                raise ValueError("Frozen episode start count differs from its ranges")
            variants = [-1]
            if source == "habit":
                episode = package.episode(episode_id)
                camera_variants = episode["record"].get("camera_variants", [])
                if len(camera_variants) != 3:
                    raise ValueError("HABIT must explicitly cover exactly three variants")
                variant_roots = []
                for variant in camera_variants:
                    roots = {view["source_root"] for view in variant["selected_views"]}
                    if len(roots) != 1:
                        raise ValueError("HABIT variant combines unrelated numeric releases")
                    variant_roots.append(next(iter(roots)))
                if set(variant_roots) != set(manifest["habit_resize256"]["aliases"].values()):
                    raise ValueError("HABIT variants do not cover the three verified releases")
                variants = list(range(3))
            elif include_interndata and source in INTERNDATA_SOURCES:
                episode = package.episode(episode_id)
                camera_variants = interndata_camera_variants(episode["record"])
                variants = list(range(len(camera_variants)))
            by_task = defaultdict(list)
            for first, stop, task in ranges:
                for variant in variants:
                    by_task[int(task)].append(ViewRange(int(episode_id), variant, first, stop))
            for task, view_ranges in sorted(by_task.items()):
                # Explicit camera variants share one numeric episode and clock.
                # Other records may share a physical unit after segmentation;
                # equal local starts are not evidence of equal physical times.
                group_id = f"{unit}:{profile_id}:{episode_id}:{task}"
                writer.add_group(view_ranges, group_id=group_id, source=source)
            ordinary_episodes += 1
            if ordinary_episodes % 100_000 == 0:
                print(
                    f"[multiview] ordinary episodes={ordinary_episodes} "
                    f"runs={writer.num_runs}",
                    flush=True,
                )
        rh_episodes, rh_rows = 0, 0
        clock_digest = hashlib.sha256()
        clock_rows = []
        rh_groups = defaultdict(list)
        seen_views = set()
        for root_number, root_id in enumerate(sorted(rh_roots), 1):
            entries = []
            rows = package.db.execute(
                "SELECT id,profile,unit,nstarts,ranges,record FROM episodes "
                "WHERE root_id=? AND split='train' AND base_episode_id IS NULL ORDER BY id",
                (root_id,),
            )
            for episode_id, profile_id, unit, count, ranges, record in rows:
                entries.append(
                    {
                        "id": int(episode_id),
                        "profile": profile_id,
                        "unit": unit,
                        "nstarts": int(count),
                        "ranges": unpack(ranges),
                        "record": unpack(record),
                    }
                )
            clocks = verify_root_clocks(package._root(root_id), entries)
            for entry in entries:
                record, state = entry["record"], clocks[entry["id"]]
                if (
                    profiles[entry["profile"]]["source"] != RH_SOURCE
                    or record["source_id"] != RH_SOURCE
                    or entry["unit"] != record["physical_episode_uid"]
                    or entry["unit"] != "rh20t_v6_scene:" + record["source_scene"]
                ):
                    raise ValueError("RH20T physical scene identity differs from the frozen index")
                first_ms = state["first_ms"]
                clock_rows.append((entry["id"], first_ms, state["last_ms"], state["rows"]))
                rh_episodes += 1
                rh_rows += state["rows"]
                view_key = entry["unit"], record["pair_id"]
                if view_key in seen_views:
                    raise ValueError("Duplicate RH20T physical scene/view identity")
                seen_views.add(view_key)
                if sum(stop - first for first, stop, _ in entry["ranges"]) != entry["nstarts"]:
                    raise ValueError("RH20T start count differs from its frozen ranges")
                for first, stop, task in entry["ranges"]:
                    if not 0 <= first < stop <= int(record["length"]) - 32:
                        raise ValueError("RH20T legal start does not contain all 33 rows")
                    key = entry["unit"], entry["profile"], int(task), first_ms % 100
                    rh_groups[key].append(ViewRange(entry["id"], -1, first, stop, first_ms // 100))
            if root_number % 20 == 0 or root_number == len(rh_roots):
                print(
                    f"[multiview] RH roots={root_number}/{len(rh_roots)} "
                    f"episodes={rh_episodes}",
                    flush=True,
                )
        for key, view_ranges in sorted(rh_groups.items()):
            if key in rejected_groups:
                # Different saved robot states cannot supervise one identical
                # chunk. Keep every view/window in ordinary time blocks.
                per_view = defaultdict(list)
                for item in view_ranges:
                    per_view[item.view].append(item)
                for view, spans in sorted(per_view.items()):
                    writer.add_group(spans, group_id=json.dumps([*key, *view]), source=RH_SOURCE)
            else:
                writer.add_group(view_ranges, group_id=json.dumps(key), source=RH_SOURCE)
        for row in sorted(clock_rows):
            clock_stream.append(row[:2])
            clock_digest.update(np.asarray(row, dtype="<i8").tobytes())
        files = writer.finish()
        files["episode_clocks.npy"] = clock_stream.finish()
        if writer.unique_samples != expected_unique_samples or writer.source_samples != expected_sources:
            raise ValueError("Final plan lost or duplicated a required source view/window")
        if writer.unique_samples + writer.padded_samples != 4 * writer.num_blocks:
            raise AssertionError("Block and unique/padding counts disagree")
        if (
            sha256(package.root / "manifest.json") != manifest_content_hash
            or sha256(norm_path) != norm_content_hash
        ):
            raise ValueError("Training package or normalization changed during planning")
        result = {
            "schema": SCHEMA,
            "package_root": str(package.root),
            "package_sha256": package.fingerprint,
            "norm_sha256": norm_hash,
            "files": files,
            "anchor_stride": 1,
            "block_size": 4,
            "unique_samples": writer.unique_samples,
            "padded_samples": writer.padded_samples,
            "num_blocks": writer.num_blocks,
            "num_runs": writer.num_runs,
            "paired_samples": writer.paired_samples,
            "source_samples": dict(sorted(writer.source_samples.items())),
            "source_paired_samples": {
                source: writer.source_paired_samples[source]
                for source in sorted(writer.source_samples)
            },
            "source_solo_samples": {
                source: count - writer.source_paired_samples[source]
                for source, count in sorted(writer.source_samples.items())
            },
            "coverage": "exact_disjoint_interval_oracle_per_physical_task_group",
            "pairing": "gcd_view_four_time_strata; complete_union_and_all_tails",
            "supervision_review": supervision_evidence,
            "habit_identity": habit_identity,
            "rh_clock_validation": {
                "episodes": rh_episodes,
                "rows": rh_rows,
                "roots": len(rh_roots),
                "step_ms": 100,
                "complete_rows_checked": True,
                "phase_groups": len(rh_groups),
                "physical_task_groups": len({key[:3] for key in rh_groups}),
                "phase_groups_with_multiple_views": sum(
                    len({item.view for item in view_ranges}) > 1
                    for view_ranges in rh_groups.values()
                ),
                "paired_samples": writer.source_paired_samples[RH_SOURCE],
                "solo_samples": (
                    writer.source_samples[RH_SOURCE]
                    - writer.source_paired_samples[RH_SOURCE]
                ),
                "episode_clock_summary_sha256": clock_digest.hexdigest(),
                "identity": "raw_source.timestamp_ms; no_metadata_offset_guess",
            },
            "builder_sha256": sha256(Path(__file__)),
        }
        if include_interndata:
            result["interndata_multiview"] = INTERNDATA_MULTIVIEW_POLICY
        write_json(output / "manifest.json", result)
        write_json(output / "_PLAN_COMPLETE", {"manifest_sha256": sha256(output / "manifest.json")})
        (output / "_BUILDING.json").unlink()
        summary_keys = ("unique_samples", "padded_samples", "num_blocks", "num_runs")
        print(json.dumps({key: result[key] for key in summary_keys}), flush=True)
        return result
    finally:
        package.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervision-review", type=Path)
    parser.add_argument("--include-interndata", action="store_true")
    args = parser.parse_args()
    prepare(
        args.package,
        args.output,
        supervision_review=args.supervision_review,
        include_interndata=args.include_interndata,
    )


if __name__ == "__main__":
    main()
