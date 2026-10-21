"""Freeze native v3 membership and episode splits; select normalization explicitly."""
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .common import atomic_json, check_sources, read_json, sha256


def prepare(config):
    roots = read_json(config["dataset_list"])
    if not isinstance(roots, list) or not roots or len(set(roots)) != len(roots):
        raise ValueError("Expected a unique nonempty array of native LeRobot v3 roots")
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError("Use a new directory for each frozen manifest")
    cameras, state_key, action_key = config["camera_keys"], config["state_key"], config["action_key"]
    manifest = dict(schema="phoenix.windows.v1", fps=30, dataset_list_sha256=sha256(config["dataset_list"]),
                    state_key=state_key, action_key=action_key, cameras=cameras, joint_unit="radian",
                    gripper_unit=config["gripper_unit"], action_semantics=config["action_semantics"],
                    files={}, metadata_sha256={}, tasks={}, episodes=[], split_seed=config["split_seed"])
    if config["action_semantics"] not in ("materialized_next_observation", "commanded_target"):
        raise ValueError("Specify action semantics explicitly")
    for rid, name in enumerate(roots):
        root = Path(name).resolve()
        info_path, task_path = root / "meta/info.json", root / "meta/tasks.parquet"
        info = read_json(info_path)
        if info["codebase_version"] != "v3.0" or info["fps"] != 30:
            raise ValueError("Phoenix requires native LeRobot v3 at 30Hz")
        receipt_path = root / "CONVERTED.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (receipt["action_semantics"] != config["action_semantics"]
                    or receipt["gripper_unit"] != config["gripper_unit"]):
                raise ValueError("Conversion semantics/units differ from preparation settings")
        if any(info["features"][key]["shape"] != [14] for key in (state_key, action_key)):
            raise ValueError("Selected state/action fields must be native joint14")
        if len(set(cameras)) != 3 or any(info["features"][key]["dtype"] != "video" for key in cameras):
            raise ValueError("Expected three ordered, distinct video feature keys")
        tasks = pq.read_table(task_path).to_pylist()
        task_map = {str(t.get("task_index", i)): t.get("task", t.get("__index_level_0__"))
                    for i, t in enumerate(tasks)}
        if len(task_map) != len(tasks) or any(not isinstance(t, str) or not t.strip() for t in task_map.values()):
            raise ValueError("Invalid task table")
        manifest["tasks"][str(rid)] = task_map
        parts = []
        for path in sorted((root / "data").rglob("*.parquet")):
            stat = path.stat()
            manifest["files"][str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            with pq.ParquetFile(path) as reader:
                for group in range(reader.num_row_groups):
                    # Read indices, rather than assuming min/max implies sorted contiguous rows.
                    indices = reader.read_row_group(group, columns=["index"])["index"].to_pylist()
                    if not indices or indices != list(range(indices[0], indices[0] + len(indices))):
                        raise ValueError("Non-contiguous or unordered native table indices")
                    parts.append(dict(path=str(path), row_group=group, start=indices[0], end=indices[-1] + 1))
        parts.sort(key=lambda row: row["start"])
        if (not parts or parts[0]["start"] != 0 or parts[-1]["end"] != info["total_frames"]
                or any(a["end"] != b["start"] for a, b in zip(parts, parts[1:]))):
            raise ValueError("Data row groups do not cover all declared frames exactly once")
        metadata_files = sorted((root / "meta/episodes").rglob("*.parquet"))
        episodes = sorted([row for p in metadata_files for row in pq.read_table(p).to_pylist()],
                          key=lambda row: row["dataset_from_index"])
        if (len(episodes) != info["total_episodes"] or not episodes
                or {r["episode_index"] for r in episodes} != set(range(len(episodes)))
                or episodes[0]["dataset_from_index"] != 0
                or episodes[-1]["dataset_to_index"] != info["total_frames"]
                or any(a["dataset_to_index"] != b["dataset_from_index"] for a, b in zip(episodes, episodes[1:]))):
            raise ValueError("Incomplete or overlapping episode metadata")
        for path in [info_path, task_path, *metadata_files]:
            manifest["metadata_sha256"][str(path)] = sha256(path)
        for row in episodes:
            begin, end, length = (int(row[k]) for k in ("dataset_from_index", "dataset_to_index", "length"))
            if length != end - begin or length <= 4:
                raise ValueError("Episode has invalid boundaries or no real future frame")
            views = []
            for key in cameras:
                path = root / info["video_path"].format(video_key=key,
                    chunk_index=row[f"videos/{key}/chunk_index"], file_index=row[f"videos/{key}/file_index"])
                stat = path.stat()
                start_s, end_s = (float(row[f"videos/{key}/{k}"]) for k in ("from_timestamp", "to_timestamp"))
                if not np.isfinite([start_s, end_s]).all() or end_s - start_s < (length - 1) / 30:
                    raise ValueError("Video interval does not cover the episode")
                views.append(dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                  start_s=start_s, end_s=end_s))
            manifest["episodes"].append(dict(root_id=str(rid), episode_index=int(row["episode_index"]),
                dataset_from_index=begin, length=length, fps=30, views=views,
                row_groups=[p for p in parts if p["end"] > begin and p["start"] < end]))
    episodes, holdout = manifest["episodes"], config["holdout"]
    if len(episodes) < 2 or not 0 < holdout < 1:
        raise ValueError("Need separate training and held-out episodes")
    def split_key(i):
        e = episodes[i]
        return hashlib.sha256(f"{config['split_seed']}:{e['root_id']}:{e['episode_index']}".encode()).hexdigest()
    count = max(1, min(len(episodes) - 1, round(len(episodes) * holdout)))
    val = set(sorted(range(len(episodes)), key=split_key)[:count])
    for i, episode in enumerate(episodes):
        episode["split"] = "val" if i in val else "train"
    totals = {kind: dict(count=0, mean=np.zeros(14), m2=np.zeros(14)) for kind in ("state", "action")}
    # Scan every episode's state/label/task identity; fit only the training split.
    for episode in episodes:
        begin, end = episode["dataset_from_index"], episode["dataset_from_index"] + episode["length"]
        for part in episode["row_groups"]:
            with pq.ParquetFile(part["path"]) as reader:
                table = reader.read_row_group(part["row_group"], columns=[state_key, action_key, "task_index",
                                                                       "episode_index", "frame_index", "timestamp"])
            offset = max(begin, part["start"]) - part["start"]
            table = table.slice(offset, min(end, part["end"]) - max(begin, part["start"]))
            start_frame = max(begin, part["start"]) - begin
            if (set(table["episode_index"].to_pylist()) != {episode["episode_index"]}
                    or table["frame_index"].to_pylist() != list(range(start_frame, start_frame + len(table)))):
                raise ValueError("Episode rows disagree with metadata")
            times = np.array(table["timestamp"].to_pylist())
            if not np.allclose(times, np.arange(start_frame, start_frame + len(table)) / 30, atol=1e-5, rtol=.001):
                raise ValueError("Native table must use an episode-relative 30Hz clock")
            tasks = table["task_index"].to_pylist()
            if len(set(tasks)) != 1 or str(tasks[0]) not in manifest["tasks"][episode["root_id"]]:
                raise ValueError("Invalid task identity or instruction boundary inside episode")
            for kind, key in (("state", state_key), ("action", action_key)):
                values = np.asarray(table[key].to_pylist(), dtype=np.float64)
                if values.shape != (len(table), 14) or not np.isfinite(values).all():
                    raise ValueError("Invalid finite joint14 state/action rows")
                if episode["split"] != "train":
                    continue
                t, n, mean = totals[kind], len(values), values.mean(0)
                delta, combined = mean - t["mean"], t["count"] + n
                t["m2"] += ((values - mean) ** 2).sum(0) + delta ** 2 * t["count"] * n / combined
                t["mean"] += delta * n / combined
                t["count"] = combined
    normalization = config.get("normalization")
    if bool(normalization) == bool(config["fit_normalization"]):
        raise ValueError("Choose an existing normalization file OR explicitly enable fit_normalization")
    stats = dict(schema="phoenix.joint14.normalization.v1", gripper_unit=config["gripper_unit"], train_only=True)
    for kind, t in totals.items():
        stats[kind] = dict(mean=t["mean"].tolist(), std=np.sqrt(t["m2"] / t["count"]).clip(1e-6).tolist(), count=t["count"])
    if normalization:
        from phoenix.model import Normalizer
        Normalizer(read_json(normalization), config["gripper_unit"])
    check_sources(manifest)
    output.mkdir(parents=True)
    atomic_json(output / "manifest.json", manifest)
    stats["manifest_sha256"] = sha256(output / "manifest.json")
    atomic_json(output / "train_statistics.json", stats)
    if normalization:
        shutil.copyfile(normalization, output / "normalization.json")
    else:
        atomic_json(output / "normalization.json", stats)
    atomic_json(output / "PREPARED.json", dict(schema="phoenix.prepared.v1", episodes=len(episodes),
                train_episodes=len(episodes) - count, val_episodes=count,
                manifest_sha256=sha256(output / "manifest.json"), normalization_sha256=sha256(output / "normalization.json")))
    return dict(status="PREPARED", output=str(output), episodes=len(episodes))
