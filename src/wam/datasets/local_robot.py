"""Local Robot80 samples with independent main/left/right RGB clips.

Use LocalRobotBatchSampler, not a shuffled DistributedSampler: upstream block
indices already address complete global batches. Both visual codecs must consume
the SAME returned ``video`` tensor, normalized to [-1, 1].
"""
from __future__ import annotations

import os
from collections import Counter, OrderedDict
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler, default_collate

from wam.datasets.local_robot_contracts import LocalRobotProjector, digest, range_start, sha256, task_at
from wam.datasets.local_robot_package import LocalRobotPackage, read_json
from wam.datasets.local_robot_text import LocalRobotTextCache
from wam.datasets.pretrain_lerobot_loader import _table_to_tensors
from wam.datasets.pretrain_sampling import PretrainLeRobotMixture
from wam.datasets.pretrain_wam_dataset import _PretrainGroupedNormalizer


class NativeStateReader:
    projector_class = LocalRobotProjector

    def __init__(self, cache_mb=128):
        self.limit = int(cache_mb) << 20
        self.cache, self.bytes = OrderedDict(), 0

    def read(self, episode, profile, start):
        record, root = episode["record"], episode["root"]
        task = task_at(episode["ranges"], start)
        projector = self.projector_class(profile)
        columns = sorted(set(projector.columns + ["index", "frame_index", "episode_index", "task_index", "timestamp"]
                             + self.extra_columns(record, profile)))
        if any("eef_xforward" in c for c in columns):
            raise ValueError("old transformed 26D EEF fields must not enter Robot80")
        begin, end = int(record["dataset_from_index"]) + start, int(record["dataset_from_index"]) + start + 33
        pieces = []
        for part in root["row_groups"]:
            a, b = int(part["start"]), int(part["end"])
            if b <= begin or a >= end:
                continue
            key = part["path"], part["row_group"], tuple(columns)
            if key in self.cache:
                table = self.cache[key]
                self.cache.move_to_end(key)
            else:
                identity = root["files"][part["path"]]
                stat = Path(part["path"]).stat()
                if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
                    raise ValueError("state file identity drift")
                with pq.ParquetFile(part["path"]) as file:
                    table = file.read_row_group(int(part["row_group"]), columns=columns, use_threads=False)
                if not set(columns).issubset(table.column_names):
                    raise ValueError("missing raw adapter field")
                if table.nbytes <= self.limit:
                    while self.cache and self.bytes + table.nbytes > self.limit:
                        self.bytes -= self.cache.popitem(last=False)[1].nbytes
                    self.cache[key] = table
                    self.bytes += table.nbytes
            pieces.append(table.slice(max(begin, a) - a, min(end, b) - max(begin, a)))
        if not pieces:
            raise ValueError("missing raw state rows")
        table = pa.concat_tables(pieces)
        if (len(table) != 33 or table["index"].to_pylist() != list(range(begin, end))
                or table["frame_index"].to_pylist() != list(range(start, start + 33))
                or set(table["episode_index"].to_pylist()) != {record["episode_index"]}
                or set(table["task_index"].to_pylist()) != {task}):
            raise ValueError("raw state/time/task identity mismatch")
        times = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        self.validate_timestamps(times, record, profile)
        data = _table_to_tensors(table)
        # Keep the validated clock: float32 can collapse distinct native stamps.
        data["timestamp"] = torch.from_numpy(times.copy())
        item = projector.project_window(data)
        self.attach_metadata(item, table, record, profile)
        return item, task

    def extra_columns(self, record, profile):
        return []

    def attach_metadata(self, item, table, record, profile):
        pass

    def validate_timestamps(self, times, record, profile):
        if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1 / float(record["fps"]), rtol=0.03, atol=1e-5):
            raise ValueError("non-native state timestamps; source needs a reviewed timestamp adapter")


def decode_views(record, profile, start, *, targets_by_key=None):
    import av
    import torch.nn.functional as F

    video = torch.full((3, 3, 9, 256, 256), -1.0)
    pad = torch.tensor([not key for key in profile["cameras"]], dtype=torch.bool)
    views = {v["key"]: v for v in record["selected_views"]}
    for slot, key in enumerate(profile["cameras"]):
        if not key:
            continue
        view = views[key]
        fps = float(record["fps"])
        context = (f"source={profile.get('source', record.get('source_id'))} "
                   f"episode={record.get('episode_index')} start={start} camera={key} "
                   f"path={view['path']} fps={fps} view_fps={view['fps']} "
                   f"available={view['available']} end_s={view['end_s']}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid video fps: {context}")
        targets = (float(view["start_s"]) + (start + np.arange(9) * 4) / fps
                   if targets_by_key is None else np.asarray(targets_by_key[key], dtype=np.float64))
        if (targets.shape != (9,) or not np.isfinite(targets).all()
                or np.any(np.diff(targets) < 0) or targets[0] < 0):
            raise ValueError(f"invalid per-camera timestamps: {context}")
        context += f" target_first={targets[0]:.9f} target_last={targets[-1]:.9f}"
        outside = targets[-1] >= view["end_s"]
        if profile.get('source') == 'abc130k':
            from wam.datasets.local_robot36_video import abc_camera_bounds
            if targets_by_key is None:
                raise ValueError(f"ABC requires explicit camera timestamps: {context}")
            try:
                lower, upper = abc_camera_bounds(record, view)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid ABC camera bounds: {context}") from exc
            outside = targets[0] < lower or targets[-1] > upper
            context += f" camera_lower={lower:.9f} camera_upper={upper:.9f}"
        view_fps = float(view['fps'])
        if (not view["available"] or not np.isfinite(view_fps)
                or abs(view_fps - fps) > 1e-5 or outside):
            raise ValueError(f"video clock/availability mismatch: {context}")
        frames = []
        with av.open(view["path"]) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            tolerance = max(0.02 / fps, 1.1 * float(stream.time_base))
            if targets_by_key is None:
                # CFR mux offsets can differ slightly from episode metadata.
                # Keep explicit camera clocks unchanged and bound high-FPS slack.
                tolerance = max(tolerance, min(0.002, 0.1 / fps))
            container.seek(max(0, int((targets[0] - tolerance) / float(stream.time_base))), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError(f"missing video PTS: {context}")
                pts = float(frame.pts * stream.time_base)
                if pts < targets[len(frames)] - tolerance:
                    continue
                if abs(pts - targets[len(frames)]) > tolerance:
                    raise ValueError(
                        "decoded RGB is not synchronized with state: "
                        f"{context} target={targets[len(frames)]:.9f} "
                        f"pts={pts:.9f} tolerance={tolerance:.9f}"
                    )
                rgb_frame = frame.to_ndarray(format="rgb24")
                # Repeated native camera stamps legitimately reference one frame.
                while len(frames) < 9 and abs(pts - targets[len(frames)]) <= tolerance:
                    frames.append(rgb_frame)
                if len(frames) == 9:
                    break
        if len(frames) != 9:
            raise ValueError(f"short video window: {context} decoded={len(frames)}/9")
        rgb = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
        if rgb.shape[-2:] != (256, 256):
            rgb = F.interpolate(rgb, (256, 256), mode="bilinear", align_corners=False, antialias=True)
        video[slot] = rgb.permute(1, 0, 2, 3) * 2 - 1
    if pad.all():
        raise ValueError("sample has no real camera")
    return video, pad


def augment_color(video, camera_is_pad, *, seed, identity, enabled):
    """BCS p=0.5, factors [0.9,1.1], one draw shared by all views/times."""
    params = torch.tensor([0.0, 1.0, 1.0, 1.0])
    if not enabled:
        return video, params
    rng = np.random.default_rng(int(digest([seed, "color", identity]), 16))
    if rng.random() >= 0.5:
        return video, params
    from torchvision.transforms.v2 import functional as F
    brightness, contrast, saturation = rng.uniform(0.9, 1.1, 3).tolist()
    out = video.clone()
    for slot in range(3):
        if camera_is_pad[slot]:
            continue
        rgb = ((video[slot] + 1) / 2).permute(1, 0, 2, 3)
        rgb = F.adjust_saturation(F.adjust_contrast(F.adjust_brightness(rgb, brightness), contrast), saturation)
        out[slot] = rgb.permute(1, 0, 2, 3) * 2 - 1
    return out, torch.tensor([1.0, brightness, contrast, saturation])


class _SourceView:
    """Minimal native-start interface consumed by the unmodified upstream sampler."""
    action_dim = state_dim = action_target_dim = state_target_dim = 80

    def __init__(self, package, profile, split):
        self.package, self.profile = package, profile
        self.name = profile["id"]
        self.spec = SimpleNamespace(allow_padding_at_end=False, sampling_family="real",
                                    source_name=profile["source"], group_id=self.name)
        rows = package.db.execute("SELECT id,nstarts FROM episodes WHERE profile=? AND split=? ORDER BY id",
                                  (self.name, split)).fetchall()
        self.trajectory_ids = np.asarray([r[0] for r in rows], dtype=np.int64)
        self.counts = np.asarray([r[1] for r in rows], dtype=np.int64)
        self.cumulative = np.cumsum(self.counts)
        self.video_keys = [k for k in profile["cameras"] if k]
        self.info = {"features": {k: {"shape": [256, 256, 3], "info": {"video.codec": "h264"}} for k in self.video_keys}}

    def set_target_dims(self, **kwargs):
        if set(kwargs.values()) != {80}:
            raise ValueError("local Robot interface must stay 80D")

    def valid_start_counts(self, allow_padding):
        if allow_padding:
            raise ValueError("local package forbids end padding")
        return self.counts

    def valid_start_for_trajectory_rank(self, pos, rank, allow_padding):
        if allow_padding:
            raise ValueError("local package forbids end padding")
        ranges = self.package.ranges(int(self.trajectory_ids[pos]))
        return range_start(ranges, int(rank))

    def __len__(self):
        return int(self.counts.sum())

    def _global_to_episode_frame(self, index):
        pos = int(np.searchsorted(self.cumulative, index, side="right"))
        before = int(self.cumulative[pos - 1]) if pos else 0
        return int(self.trajectory_ids[pos]), self.valid_start_for_trajectory_rank(pos, index - before, False)


class LocalRobot80Dataset(Dataset):
    prefer_sequential_indices = True
    package_class = LocalRobotPackage
    reader_class = NativeStateReader

    def __init__(self, package_root, *, split="train", batch_size=16, seed=42,
                 color_augmentation=True, language_augmentation=True, audit=False, allow_smoke_stats=False,
                 state_cache_mb=128, validation_samples_per_profile=32):
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if audit and allow_smoke_stats:
            raise ValueError("audit raw projection and normalized smoke are distinct modes")
        self.package = self.package_class(package_root, audit=audit or allow_smoke_stats)
        self.split, self.seed, self.batch_size = split, int(seed), int(batch_size)
        self.color_augmentation = bool(color_augmentation) and split == "train"
        self.language_augmentation = bool(language_augmentation) and split == "train"
        self.audit = bool(audit)
        self.reader = self.reader_class(state_cache_mb)
        self.text_caches, self.text_pid = {}, os.getpid()
        self.normalizer = None
        self.norm_policy = self.package.manifest["normalization"]
        if not audit:
            marker = read_json(self.package.root / "_NORM_COMPLETE")
            if marker["package_sha256"] != self.package.fingerprint:
                raise ValueError("normalization/package identity mismatch")
            stats = read_json(self.package.root / "norm.json")
            if stats.get("purpose") != "training" and not (allow_smoke_stats and stats.get("purpose") == "smoke"):
                raise ValueError("smoke statistics cannot be used for formal training")
            if stats["split"] != "train" or stats["package_sha256"] != self.package.fingerprint or stats["action_dim"] != 80 or stats["state_dim"] != 80:
                raise ValueError("normalization is not this package's train-only Robot80 stats")
            required = {p["stats_group"] for p in self.package.manifest["profiles"].values()}
            if set(stats["groups"]) != required or any(stats["groups"][g]["num_samples"] <= 0 for g in required):
                raise ValueError("incomplete grouped normalization")
            self.normalizer = _PretrainGroupedNormalizer(stats, **{
                "mode": self.norm_policy["stats"]["norm_mode"],
                "minimum_widths": self.norm_policy["stats"].get("minimum_widths"),
                "passthrough_dims": self.norm_policy["stats"].get("passthrough_dims")})
        self.datasets = [_SourceView(self.package, p, split) for p in self.package.manifest["profiles"].values()]
        self.datasets = [d for d in self.datasets if len(d)]
        if not self.datasets:
            raise ValueError("empty local split")
        # Complete blocks are pooled within each upstream source/decode stream.
        # Passing their capacity avoids counting the discarded stream tails as blocks.
        streams = Counter()
        inspector = object.__new__(PretrainLeRobotMixture)
        for ds in self.datasets:
            streams[(inspector._dataset_decode_bucket(ds), ds.spec.source_name)] += len(ds)
        complete_capacity = sum((n // 4) * 4 for n in streams.values())
        self.mixture = PretrainLeRobotMixture(self.datasets, training=split == "train", seed=seed,
            training_block_size=4, training_batch_size_per_rank=batch_size,
            training_frame_stride=1, allow_padding_at_end=False,
            epoch_length=complete_capacity if split == "train" else None,
            training_sampling_strategy="optimizer_stratified_without_replacement")
        self.eval_steps = []
        if split == "validation":
            if validation_samples_per_profile < 1:
                raise ValueError("validation_samples_per_profile must be positive")
            for i, ds in enumerate(self.datasets):
                # Fixed, legal, source-stratified panel; never borrowed from train.
                for flat in np.linspace(0, len(ds) - 1, min(validation_samples_per_profile, len(ds)), dtype=np.int64):
                    episode, start = ds._global_to_episode_frame(int(flat))
                    self.eval_steps.append((i, episode, start))
        self.fixed_validation_panel = [dict(index=j, source=self.datasets[i].profile["source"],
            seed=self.seed + j, episode_id=int(episode), start=int(start))
            for j, (i, episode, start) in enumerate(self.eval_steps)]

    def set_epoch(self, epoch):
        self.mixture.set_epoch(epoch)

    def __getstate__(self):
        return dict(self.__dict__, text_caches={}, text_pid=None,
                    reader=self.reader_class(self.reader.limit >> 20))

    def __len__(self):
        return len(self.mixture) if self.split == "train" else len(self.eval_steps)

    def _normalize(self, item, profile):
        if self.normalizer is None:
            return item
        group = profile["stats_group"]
        for key, kind, mask in (("action", "action", "action_mask"), ("proprio", "state", "proprio_mask")):
            value = self.normalizer.forward(item[key], group=group, kind=kind)
            rules = self.norm_policy["post_normalize_transforms"].get(kind, {}).get("by_stats_group", {}).get(group, [])
            for rule in rules:
                dims = rule["dims"]
                value[..., dims] = value[..., dims] * float(rule.get("scale", 1)) + float(rule.get("offset", 0))
                if "clamp" in rule:
                    value[..., dims] = value[..., dims].clamp(*rule["clamp"])
            value[~item[mask]] = 0
            if not torch.isfinite(value).all():
                raise ValueError("non-finite normalized feature")
            item[key] = value
        return item

    def read_window(self, episode_id, start, *, draw=0, with_video=True, with_text=True, normalize=True):
        ep = self.package.episode(episode_id)
        if ep["split"] != self.split:
            raise ValueError("attempt to read the other physical split")
        p = self.package.manifest["profiles"][ep["profile"]]
        item, task = self.reader.read(ep, p, int(start))
        if normalize:
            item = self._normalize(item, p)
        identity = [self.mixture.epoch if self.split == "train" else 0, ep["record"]["sample_id"], int(start), int(draw)]
        item.update(source_id=p["source"], stats_group=p["stats_group"], profile_id=p["id"],
                    episode_id=int(episode_id), sample_id=ep["record"]["sample_id"],
                    physical_episode_uid=ep["unit"], start=int(start), task_index=int(task),
                    camera_is_pad=torch.tensor([not key for key in p["cameras"]], dtype=torch.bool),
                    camera_keys=tuple(p["cameras"]), camera_roles=tuple(p["camera_roles"]),
                    video_num_views=3, video_is_pad=torch.zeros(9, dtype=torch.bool),
                    image_is_pad=torch.zeros(9, dtype=torch.bool),
                    action_loss_reduction="sample_mean", split=self.split,
                    normalization_applied=self.normalizer is not None and normalize)
        if with_video:
            video, item["camera_is_pad"] = decode_views(ep["record"], p, start)
            item["video"], item["color_params"] = augment_color(video, item["camera_is_pad"],
                seed=self.seed, identity=identity, enabled=self.color_augmentation)
        if with_text:
            if self.text_pid != os.getpid():
                self.text_caches, self.text_pid = {}, os.getpid()
            matches = [b for b in ep["record"]["t5_task_bindings"] if int(b["task_index"]) == task]
            if len(matches) != 1:
                raise ValueError("task has no unique T5 binding")
            binding = matches[0]
            key = binding["registry"], p["source"], binding["registry_sha256"]
            if key not in self.text_caches:
                self.text_caches[key] = LocalRobotTextCache(key[0], key[1], key[2])
            item.update(self.text_caches[key].resolve(binding, seed=self.seed, identity=identity, augment=self.language_augmentation))
        return item

    def __getitem__(self, index):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        _, episode, start = self.mixture._sample_step(int(index)) if self.split == "train" else self.eval_steps[int(index)]
        return self.read_window(episode, start, draw=int(index))

    @staticmethod
    def collate_fn(samples):
        for s in samples:
            if s["camera_is_pad"].dtype != torch.bool or s["camera_is_pad"].shape != (3,) or s["camera_is_pad"].all():
                raise ValueError("every sample must carry a valid camera_is_pad[3]")
            if s["video"].shape != (3, 3, 9, 256, 256) or s["action"].shape != (32, 80):
                raise ValueError("local Robot80 tensor contract mismatch")
        return default_collate(samples)


class LocalRobotBatchSampler(Sampler):
    """Direct global index partitioning; do not additionally shard this loader."""
    def __init__(self, dataset, *, rank=None, world_size=None):
        self.dataset = dataset
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else int(world_size)
        if not 0 <= self.rank < self.world_size or self.world_size != dataset.mixture.bucket_wave_world_size:
            raise ValueError("sampler/mixture distributed geometry mismatch")

    def __iter__(self):
        b = self.dataset.batch_size
        for begin in range(self.rank * b, len(self.dataset), self.world_size * b):
            stop = min(begin + b, len(self.dataset))
            if stop - begin == b or self.dataset.split == "validation":
                yield list(range(begin, stop))

    def __len__(self):
        b = self.dataset.batch_size
        n = max(0, len(self.dataset) - self.rank * b)
        return (n + self.world_size * b - 1) // (self.world_size * b)
