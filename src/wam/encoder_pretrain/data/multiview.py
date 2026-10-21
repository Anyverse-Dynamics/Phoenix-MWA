"""Metadata-driven all-view ego-video loading for the encoder distillation recipe."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler

from wam.encoder_pretrain.data.catalog import load_or_build_ego_video_catalog
from wam.encoder_pretrain.data.clips import (
    REJECTED_VIDEOS_FILENAME,
    BalancedDatasetBatchSampler,
    BucketedEgoVideoDataset,
    EgoVideoClipDataset,
    EgoVideoCollectionDataset,
    EgoVideoMixtureDataset,
    SharedBucketBatchSampler,
    SharedClipBucketLoader,
    TemperatureDatasetBatchSampler,
    UniformClipBucketLoaders,
    _BlockShuffleCursor,
    _PrefetchTrackedBatchSampler,
    _entry_dataset_paths,
    _infer_fps,
    _parquet_columns,
    _resolve_video_key,
    _select_ego_video_cfg,
    ego_video_collate_fn,
)


SAMPLING_GROUPS = ("ego_general", "robot")
VIEW_ROLES = ("primary", "auxiliary")
SAMPLING_COUNT_KEYS = (
    *(f"group/{group}" for group in SAMPLING_GROUPS),
    *(f"view/{role}" for role in VIEW_ROLES),
    *(f"cell/{group}/{role}" for group in SAMPLING_GROUPS for role in VIEW_ROLES),
    "total",
)


@dataclass(frozen=True)
class SamplingPartition:
    dataset_name: str
    sampling_group: str
    view_role: str
    offset: int
    size: int

    @property
    def identity(self) -> tuple[str, str, str, int, int]:
        return self.dataset_name, self.sampling_group, self.view_role, self.offset, self.size


def _ordered_video_keys(info: dict[str, Any], entry: Any) -> tuple[str, ...]:
    """Put the configured primary view first, then include every video feature."""

    primary_key = _resolve_video_key(
        info,
        entry.get("video_key", None),
        entry.get("video_key_candidates", None),
    )
    video_keys = [
        str(key)
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    return tuple([primary_key, *[key for key in video_keys if key != primary_key]])


class AutoMultiViewEgoVideoClipDataset(EgoVideoClipDataset):
    """One physical camera stream selected from metadata-discovered views."""

    def __init__(self, *args: Any, sampling_group: str, view_role: str, **kwargs: Any) -> None:
        if sampling_group not in SAMPLING_GROUPS:
            raise ValueError(f"Unsupported sampling_group={sampling_group!r}; expected one of {SAMPLING_GROUPS}")
        if view_role not in VIEW_ROLES:
            raise ValueError(f"Unsupported view_role={view_role!r}; expected one of {VIEW_ROLES}")
        self.sampling_group = sampling_group
        self.view_role = view_role
        super().__init__(*args, **kwargs)

    def _load_rejected_video_paths(self) -> set[str]:
        rejection_path = self.dataset_path / "meta" / REJECTED_VIDEOS_FILENAME
        if not rejection_path.is_file():
            return set()
        payload = json.loads(rejection_path.read_text())
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported rejected-video manifest: {rejection_path}")
        if payload.get("video_key") not in {None, self.video_key}:
            return set()
        return {str(item["path"]) for item in payload.get("videos", [])}

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["metadata"]["sampling_group"] = self.sampling_group
        sample["metadata"]["view_role"] = self.view_role
        return sample


class AutoMultiViewEgoVideoCollectionDataset(EgoVideoCollectionDataset):
    """A logical source laid out as one contiguous primary and auxiliary partition."""

    def __init__(
        self,
        dataset_name: str,
        datasets: list[AutoMultiViewEgoVideoClipDataset],
        sampling_group: str,
        num_primary_datasets: int,
    ) -> None:
        self.sampling_group = str(sampling_group)
        self.num_primary_datasets = int(num_primary_datasets)
        if not 0 < self.num_primary_datasets <= len(datasets):
            raise ValueError("Every logical all-view source requires at least one primary dataset")
        if any(dataset.sampling_group != self.sampling_group for dataset in datasets):
            raise ValueError(f"Physical datasets disagree on sampling_group for {dataset_name!r}")
        roles = [dataset.view_role for dataset in datasets]
        expected_roles = ["primary"] * self.num_primary_datasets + ["auxiliary"] * (
            len(datasets) - self.num_primary_datasets
        )
        if roles != expected_roles:
            raise ValueError(f"All-view source {dataset_name!r} must place primary streams before auxiliary streams")
        super().__init__(dataset_name, datasets)
        self.primary_length = sum(len(dataset) for dataset in datasets[: self.num_primary_datasets])

    def with_clip_spec(self, clip_len: int, frame_stride: int) -> "AutoMultiViewEgoVideoCollectionDataset":
        return type(self)(
            self.dataset_name,
            [dataset.with_clip_spec(clip_len, frame_stride) for dataset in self.datasets],
            self.sampling_group,
            self.num_primary_datasets,
        )

    def with_disjoint_clip_spec(
        self,
        clip_len: int,
        frame_stride: int,
        **kwargs: Any,
    ) -> "AutoMultiViewEgoVideoCollectionDataset":
        return type(self)(
            self.dataset_name,
            [dataset.with_disjoint_clip_spec(clip_len, frame_stride, **kwargs) for dataset in self.datasets],
            self.sampling_group,
            self.num_primary_datasets,
        )


def _preflight_auto_multiview_dataset_entries(
    data_cfg: Any,
    only_dataset: str | None,
) -> list[tuple[Any, list[tuple[Path, tuple[str, ...]]]]]:
    """Discover and validate every video feature for each selected physical dataset."""

    data_root_dir = Path(str(data_cfg.data_root_dir))
    selected = []
    for entry in data_cfg.datasets:
        if only_dataset is not None and entry.name != only_dataset:
            continue
        sampling_group = entry.get("sampling_group", None)
        if sampling_group not in SAMPLING_GROUPS:
            raise ValueError(
                f"Dataset {entry.name!r} has sampling_group={sampling_group!r}; expected one of {SAMPLING_GROUPS}"
            )
        physical_sources = []
        for dataset_path in _entry_dataset_paths(entry, data_root_dir):
            info_path = dataset_path / "meta/info.json"
            with info_path.open("r") as handle:
                info = json.load(handle)
            video_keys = _ordered_video_keys(info, entry)
            episode_files = sorted((dataset_path / "meta/episodes").glob("chunk-*/*.parquet"))
            if not episode_files:
                raise FileNotFoundError(
                    f"No LeRobot v3 episode parquet files found under {dataset_path / 'meta/episodes'}"
                )
            columns = _parquet_columns(episode_files[0])
            for video_key in video_keys:
                fps = float(entry.get("fps", None) or _infer_fps(info, video_key))
                target_fps = entry.get("target_fps", data_cfg.get("target_fps", None))
                if target_fps is not None and float(target_fps) > fps:
                    raise ValueError(
                        f"target_fps={target_fps} exceeds source fps={fps} for "
                        f"{entry.name} view {video_key!r} at {dataset_path}"
                    )
                video_prefix = f"videos/{video_key}"
                required_columns = {
                    "episode_index",
                    "length",
                    "data/chunk_index",
                    "data/file_index",
                    f"{video_prefix}/chunk_index",
                    f"{video_prefix}/file_index",
                    f"{video_prefix}/from_timestamp",
                }
                missing = sorted(required_columns - columns)
                if missing:
                    raise KeyError(f"{episode_files[0]} is missing required columns for {video_key!r}: {missing}")
            physical_sources.append((dataset_path, video_keys))
        selected.append((entry, physical_sources))

    if not selected:
        raise ValueError(f"No ego-video dataset matched only_dataset={only_dataset!r}")
    return selected


class AutoMultiViewEgoVideoMixtureDataset(EgoVideoMixtureDataset):
    """Expand metadata-discovered camera streams within each logical source."""

    @classmethod
    def from_config(cls, data_cfg: Any, only_dataset: str | None = None) -> "AutoMultiViewEgoVideoMixtureDataset":
        if str(data_cfg.get("video_view_mode", "all")) != "all":
            raise ValueError("Auto-multiview loading requires video_view_mode=all")
        dataset = load_or_build_ego_video_catalog(
            data_cfg,
            only_dataset,
            lambda: cls._from_config_uncached(data_cfg, only_dataset),
        )
        entries = {str(entry.name): entry for entry in data_cfg.datasets}
        global_backend_kwargs = dict(data_cfg.get("video_backend_kwargs", {}))
        for source in dataset.datasets:
            entry = entries[source.dataset_name]
            backend_kwargs = global_backend_kwargs | dict(entry.get("video_backend_kwargs", {}))
            physical_datasets = source.datasets if isinstance(source, EgoVideoCollectionDataset) else [source]
            for physical_dataset in physical_datasets:
                physical_dataset.video_backend_kwargs = backend_kwargs
        return dataset

    @classmethod
    def _from_config_uncached(cls, data_cfg: Any, only_dataset: str | None) -> "AutoMultiViewEgoVideoMixtureDataset":
        datasets = []
        for entry, physical_sources in _preflight_auto_multiview_dataset_entries(data_cfg, only_dataset):
            name = str(entry.name)
            sampling_group = str(entry.sampling_group)

            def make_dataset(dataset_path: Path, video_key: str, view_role: str):
                return AutoMultiViewEgoVideoClipDataset(
                    dataset_path=dataset_path,
                    dataset_name=name,
                    video_key=video_key,
                    sampling_group=sampling_group,
                    view_role=view_role,
                    fps=entry.get("fps", None),
                    target_fps=entry.get("target_fps", data_cfg.get("target_fps", None)),
                    clip_len=entry.get("clip_len", data_cfg.get("clip_len", 8)),
                    frame_stride=entry.get("frame_stride", data_cfg.get("frame_stride", 1)),
                    clip_stride=entry.get("clip_stride", data_cfg.get("clip_stride", 1)),
                    resize=entry.get("resize", data_cfg.get("resize", [224, 224])),
                    video_backend=entry.get("video_backend", data_cfg.get("video_backend", "decord")),
                    video_backend_kwargs=(
                        dict(data_cfg.get("video_backend_kwargs", {}))
                        | dict(entry.get("video_backend_kwargs", {}))
                    ),
                    include_action=entry.get("include_action", data_cfg.get("include_action", False)),
                    include_state=entry.get("include_state", data_cfg.get("include_state", False)),
                    action_key=entry.get("action_key", data_cfg.get("action_key", "action")),
                    state_key=entry.get("state_key", data_cfg.get("state_key", None)),
                )

            primary_datasets = [
                make_dataset(dataset_path, video_keys[0], "primary")
                for dataset_path, video_keys in physical_sources
            ]
            auxiliary_datasets = [
                make_dataset(dataset_path, video_key, "auxiliary")
                for dataset_path, video_keys in physical_sources
                for video_key in video_keys[1:]
            ]
            datasets.append(
                AutoMultiViewEgoVideoCollectionDataset(
                    name,
                    [*primary_datasets, *auxiliary_datasets],
                    sampling_group,
                    len(primary_datasets),
                )
            )
        return cls(datasets)

    @property
    def sampling_partitions(self) -> tuple[SamplingPartition, ...]:
        partitions = []
        source_offset = 0
        for source in self.datasets:
            if not isinstance(source, AutoMultiViewEgoVideoCollectionDataset):
                raise TypeError(f"Unexpected all-view source type: {type(source)}")
            if source.primary_length:
                partitions.append(
                    SamplingPartition(
                        source.dataset_name,
                        source.sampling_group,
                        "primary",
                        source_offset,
                        source.primary_length,
                    )
                )
            auxiliary_size = len(source) - source.primary_length
            if auxiliary_size:
                partitions.append(
                    SamplingPartition(
                        source.dataset_name,
                        source.sampling_group,
                        "auxiliary",
                        source_offset + source.primary_length,
                        auxiliary_size,
                    )
                )
            source_offset += len(source)
        return tuple(partitions)


def _normalized_target_probabilities(
    configured: Any,
    expected_keys: tuple[str, ...],
    config_name: str,
) -> dict[str, float]:
    probabilities = {str(key): float(value) for key, value in configured.items()}
    if set(probabilities) != set(expected_keys):
        raise ValueError(f"{config_name} must define exactly {expected_keys}, got {tuple(probabilities)}")
    if any(probability <= 0 for probability in probabilities.values()):
        raise ValueError(f"{config_name} probabilities must be positive, got {probabilities}")
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{config_name} probabilities must sum to 1, got {total}")
    return {key: probabilities[key] for key in expected_keys}


def _calibrate_cell_probabilities(
    base_probabilities: np.ndarray,
    group_targets: np.ndarray,
    view_targets: np.ndarray,
) -> np.ndarray:
    calibrated = np.asarray(base_probabilities, dtype=np.float64).copy()
    calibrated /= calibrated.sum()
    for _ in range(10_000):
        row_sums = calibrated.sum(axis=1)
        if np.any((group_targets > 0) & (row_sums == 0)):
            raise ValueError("Sampling-group targets are infeasible because a required group has no clips")
        calibrated *= (group_targets / row_sums)[:, None]
        column_sums = calibrated.sum(axis=0)
        if np.any((view_targets > 0) & (column_sums == 0)):
            raise ValueError("View-role targets are infeasible because a required role has no clips")
        calibrated *= (view_targets / column_sums)[None, :]
        if np.allclose(calibrated.sum(axis=1), group_targets, atol=1e-12, rtol=0.0) and np.allclose(
            calibrated.sum(axis=0), view_targets, atol=1e-12, rtol=0.0
        ):
            return calibrated
    raise ValueError("Sampling group/view targets are infeasible for the available joint partitions")


class StratifiedViewTemperatureBatchSampler(_PrefetchTrackedBatchSampler, Sampler[list[int]]):
    """Temperature-sample logical sources after matching group and view-role marginals."""

    def __init__(
        self,
        dataset: AutoMultiViewEgoVideoMixtureDataset,
        batch_size: int,
        temperature: float,
        sampling_group_probs: Any,
        view_role_probs: Any,
        shuffle_block_size: int = 262_144,
        seed: int = 42,
    ) -> None:
        self.partitions = dataset.sampling_partitions
        if not self.partitions:
            raise ValueError("Stratified view sampling requires at least one non-empty partition")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.temperature = float(temperature)
        self.sampling_group_probs = _normalized_target_probabilities(
            sampling_group_probs,
            SAMPLING_GROUPS,
            "sampling_group_probs",
        )
        self.view_role_probs = _normalized_target_probabilities(
            view_role_probs,
            VIEW_ROLES,
            "view_role_probs",
        )

        source_sizes = {source.dataset_name: len(source) for source in dataset.datasets}
        base_masses = np.asarray(
            [
                source_sizes[partition.dataset_name] ** self.temperature
                * partition.size
                / source_sizes[partition.dataset_name]
                for partition in self.partitions
            ],
            dtype=np.float64,
        )
        group_indices = {name: index for index, name in enumerate(SAMPLING_GROUPS)}
        view_indices = {name: index for index, name in enumerate(VIEW_ROLES)}
        base_cells = np.zeros((len(SAMPLING_GROUPS), len(VIEW_ROLES)), dtype=np.float64)
        for partition, mass in zip(self.partitions, base_masses, strict=True):
            base_cells[group_indices[partition.sampling_group], view_indices[partition.view_role]] += mass
        self.base_cell_probabilities = base_cells / base_cells.sum()
        self.calibrated_cell_probabilities = _calibrate_cell_probabilities(
            self.base_cell_probabilities,
            np.asarray([self.sampling_group_probs[name] for name in SAMPLING_GROUPS]),
            np.asarray([self.view_role_probs[name] for name in VIEW_ROLES]),
        )

        cell_multipliers = np.divide(
            self.calibrated_cell_probabilities,
            self.base_cell_probabilities,
            out=np.zeros_like(self.calibrated_cell_probabilities),
            where=self.base_cell_probabilities > 0,
        )
        base_partition_probabilities = base_masses / base_masses.sum()
        calibrated_masses = np.asarray(
            [
                probability
                * cell_multipliers[
                    group_indices[partition.sampling_group],
                    view_indices[partition.view_role],
                ]
                for partition, probability in zip(self.partitions, base_partition_probabilities, strict=True)
            ]
        )
        self.probabilities = torch.tensor(calibrated_masses / calibrated_masses.sum(), dtype=torch.float64)
        self.dataset_probabilities = {
            source.dataset_name: float(
                sum(
                    self.probabilities[index]
                    for index, partition in enumerate(self.partitions)
                    if partition.dataset_name == source.dataset_name
                )
            )
            for source in dataset.datasets
        }

        self.num_batches = math.ceil(sum(len(source) for source in dataset.datasets) / self.batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_offset = 0
        self.cursors = [
            _BlockShuffleCursor(partition.size, shuffle_block_size, self.seed + index * 10_009)
            for index, partition in enumerate(self.partitions)
        ]
        self._init_prefetch_tracking()

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for batch_index in range(self.batch_offset, self.num_batches):
            generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + batch_index)
            selected_partitions = torch.multinomial(
                self.probabilities,
                self.batch_size,
                replacement=True,
                generator=generator,
            )
            batch = torch.empty(self.batch_size, dtype=torch.long)
            for partition_index, (partition, cursor) in enumerate(zip(self.partitions, self.cursors, strict=True)):
                positions = torch.nonzero(selected_partitions == partition_index, as_tuple=False).flatten()
                if len(positions):
                    batch[positions] = cursor.take(len(positions)) + partition.offset
            self.batch_offset = batch_index + 1
            self._record_yielded_state()
            yield batch.tolist()
        self.epoch += 1
        self.batch_offset = 0

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "format": "stratified_view_temperature.v1",
            "epoch": self.epoch,
            "batch_offset": self.batch_offset,
            "partitions": [partition.identity for partition in self.partitions],
            "cursors": [cursor.state_dict() for cursor in self.cursors],
        }

    def _restore_state(self, state: dict[str, Any]) -> None:
        if state.get("format") != "stratified_view_temperature.v1":
            raise ValueError(f"Unsupported stratified view sampler state: {state.get('format')!r}")
        expected_partitions = [partition.identity for partition in self.partitions]
        restored_partitions = [tuple(identity) for identity in state["partitions"]]
        if restored_partitions != expected_partitions:
            raise ValueError("Stratified view sampler partitions differ from the checkpoint")
        self.epoch = int(state["epoch"])
        self.batch_offset = int(state["batch_offset"])
        for cursor, cursor_state in zip(self.cursors, state["cursors"], strict=True):
            cursor.load_state_dict(cursor_state)


def make_auto_multiview_bucketed_ego_video_dataloaders(cfg: Any) -> UniformClipBucketLoaders:
    """Build homogeneous length/stride buckets from the all-view mixture."""

    data_cfg = _select_ego_video_cfg(cfg)
    lengths = [int(value) for value in data_cfg.get("length_buckets", [4, 8, 12, 16, 24])]
    strides = [int(value) for value in data_cfg.get("stride_buckets", [1, 2, 4])]
    length_probs = [float(value) for value in data_cfg.get("length_probs", [1.0] * len(lengths))]
    stride_probs = [float(value) for value in data_cfg.get("stride_probs", [1.0] * len(strides))]
    if len(length_probs) != len(lengths) or len(stride_probs) != len(strides):
        raise ValueError("length_probs/stride_probs must align with their bucket lists")
    frame_budget = int(data_cfg.get("per_device_frame_budget", 24))
    shuffle_block_size = int(data_cfg.get("shuffle_block_size", 262_144))
    if frame_budget < max(lengths):
        raise ValueError(f"per_device_frame_budget={frame_budget} cannot fit the largest T bucket {max(lengths)}")

    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    base_data = _select_ego_video_cfg(base_cfg)
    base_data.clip_len = lengths[0]
    base_data.frame_stride = strides[0]
    base_dataset = AutoMultiViewEgoVideoMixtureDataset.from_config(base_data)

    bucket_keys = [(length, stride) for length in lengths for stride in strides]
    clip_sampling_mode = str(data_cfg.get("clip_sampling_mode", "sliding"))
    if clip_sampling_mode not in {"sliding", "disjoint_window"}:
        raise ValueError(f"Unsupported clip_sampling_mode={clip_sampling_mode!r}")
    configured_window_span = data_cfg.get("unique_window_span", "auto")
    required_window_span = max((length - 1) * stride + 1 for length, stride in bucket_keys)
    window_span = required_window_span if configured_window_span == "auto" else int(configured_window_span)
    if clip_sampling_mode == "disjoint_window" and window_span < required_window_span:
        raise ValueError(f"unique_window_span={window_span} cannot fit maximum required span {required_window_span}")

    loaders = {}
    stateful_samplers = {}
    for bucket_index, (length, stride) in enumerate(bucket_keys):
        dataset = (
            base_dataset.with_disjoint_clip_spec(
                length,
                stride,
                window_span=window_span,
                bucket_index=bucket_index,
                num_buckets=len(bucket_keys),
                seed=int(cfg.get("seed", 42)),
            )
            if clip_sampling_mode == "disjoint_window"
            else (
                base_dataset
                if (length, stride) == (lengths[0], strides[0])
                else base_dataset.with_clip_spec(length, stride)
            )
        )
        if len(dataset) == 0:
            continue
        num_workers = int(data_cfg.get("num_workers", 0))
        batch_size = max(1, frame_budget // length)
        kwargs = {
            "dataset": dataset,
            "num_workers": num_workers,
            "collate_fn": ego_video_collate_fn,
            "pin_memory": bool(data_cfg.get("pin_memory", False)),
        }
        sampling = str(data_cfg.get("dataset_sampling", "natural"))
        if sampling == "stratified_view_temperature":
            batch_sampler = StratifiedViewTemperatureBatchSampler(
                dataset,
                batch_size=batch_size,
                temperature=float(data_cfg.get("dataset_temperature", 0.5)),
                sampling_group_probs=data_cfg.sampling_group_probs,
                view_role_probs=data_cfg.view_role_probs,
                shuffle_block_size=shuffle_block_size,
                seed=int(cfg.get("seed", 42)) + length * 101 + stride,
            )
            kwargs["batch_sampler"] = batch_sampler
            stateful_samplers[(length, stride)] = batch_sampler
        elif sampling == "balanced":
            batch_sampler = BalancedDatasetBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle_block_size=shuffle_block_size,
                seed=int(cfg.get("seed", 42)) + length * 101 + stride,
            )
            kwargs["batch_sampler"] = batch_sampler
            stateful_samplers[(length, stride)] = batch_sampler
        elif sampling == "temperature":
            configured_weights = data_cfg.get("dataset_weights", None)
            batch_sampler = TemperatureDatasetBatchSampler(
                dataset,
                batch_size=batch_size,
                temperature=float(data_cfg.get("dataset_temperature", 0.5)),
                dataset_weights=(
                    [float(value) for value in configured_weights] if configured_weights is not None else None
                ),
                shuffle_block_size=shuffle_block_size,
                seed=int(cfg.get("seed", 42)) + length * 101 + stride,
            )
            kwargs["batch_sampler"] = batch_sampler
            stateful_samplers[(length, stride)] = batch_sampler
        else:
            kwargs.update(
                {
                    "batch_size": batch_size,
                    "shuffle": bool(data_cfg.get("shuffle", False)),
                    "drop_last": True,
                }
            )
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
            kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
        loaders[(length, stride)] = DataLoader(**kwargs)

    bucket_sampling = str(data_cfg.get("bucket_sampling", "marginal"))
    if bucket_sampling == "epoch_balanced":
        bucket_weights = {key: len(loader) for key, loader in loaders.items()}
    elif bucket_sampling == "marginal":
        bucket_weights = None
    else:
        raise ValueError(f"Unsupported bucket_sampling={bucket_sampling!r}")
    return UniformClipBucketLoaders(
        loaders,
        seed=int(cfg.get("seed", 42)),
        stateful_samplers=stateful_samplers,
        length_probabilities=dict(zip(lengths, length_probs, strict=True)),
        stride_probabilities=dict(zip(strides, stride_probs, strict=True)),
        bucket_weights=bucket_weights,
    )


class SamplingTrackedSharedClipBucketLoader(SharedClipBucketLoader):
    """Retain local group/view counts from the last batch for distributed logging."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_sampling_counts: dict[str, int] = {}

    def next(self) -> tuple[tuple[int, int], list[dict[str, Any]]]:
        key, batch = super().next()
        counts = {name: 0 for name in SAMPLING_COUNT_KEYS}
        for sample in batch:
            metadata = sample["metadata"]
            group = str(metadata["sampling_group"])
            role = str(metadata["view_role"])
            counts[f"group/{group}"] += 1
            counts[f"view/{role}"] += 1
            counts[f"cell/{group}/{role}"] += 1
        counts["total"] = len(batch)
        self.last_sampling_counts = counts
        return key, batch


def make_shared_auto_multiview_ego_video_dataloader(
    cfg: Any,
    *,
    world_size: int,
) -> SamplingTrackedSharedClipBucketLoader:
    """Use one worker pool for all buckets of the metadata-expanded mixture."""

    bucket_loaders = make_auto_multiview_bucketed_ego_video_dataloaders(cfg)
    if set(bucket_loaders.loaders) != set(bucket_loaders.stateful_samplers):
        raise ValueError("Shared bucket loading requires a stateful dataset sampler")

    bucket_datasets = {key: loader.dataset for key, loader in bucket_loaders.loaders.items()}
    sampler = SharedBucketBatchSampler(
        bucket_loaders.stateful_samplers,
        bucket_loaders.bucket_probabilities,
        world_size=world_size,
        seed=bucket_loaders.seed,
    )
    sampler.enable_prefetch_tracking()

    data_cfg = _select_ego_video_cfg(cfg)
    num_workers = int(data_cfg.get("num_workers", 0))
    worker_restart_interval_batches = int(data_cfg.get("worker_restart_interval_batches", 0))
    if worker_restart_interval_batches and num_workers == 0:
        raise ValueError("worker_restart_interval_batches requires num_workers > 0")
    persistent_workers = bool(data_cfg.get("persistent_workers", False))
    if worker_restart_interval_batches and not persistent_workers:
        raise ValueError("worker_restart_interval_batches requires persistent_workers=true")
    kwargs = {
        "dataset": BucketedEgoVideoDataset(bucket_datasets),
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": ego_video_collate_fn,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    loader = DataLoader(**kwargs)
    return SamplingTrackedSharedClipBucketLoader(
        loader,
        sampler,
        bucket_datasets,
        worker_restart_interval_batches=worker_restart_interval_batches,
    )


def format_stratified_view_sampling_statistics(
    bucket_loader: SamplingTrackedSharedClipBucketLoader,
    cfg: Any,
    *,
    num_processes: int,
) -> str:
    """Format the effective base and calibrated sampling distribution for every bucket."""

    data_cfg = _select_ego_video_cfg(cfg)
    lines = [
        "***** Stratified All-View Sampling Statistics *****",
        (
            f"  Processes = {num_processes} | Workers/rank = {bucket_loader.loader.num_workers} | "
            f"Frame budget/rank = {int(data_cfg.get('per_device_frame_budget', 24))}"
        ),
        (
            "  Targets = "
            + ", ".join(f"{name}:{float(data_cfg.sampling_group_probs[name]):.1%}" for name in SAMPLING_GROUPS)
            + " | "
            + ", ".join(f"{name}:{float(data_cfg.view_role_probs[name]):.1%}" for name in VIEW_ROLES)
        ),
    ]
    for key, sampler in sorted(bucket_loader.sampler.samplers.items()):
        if not isinstance(sampler, StratifiedViewTemperatureBatchSampler):
            raise TypeError(f"Bucket {key} does not use StratifiedViewTemperatureBatchSampler")
        base_cells = ", ".join(
            f"{group}/{role}:{sampler.base_cell_probabilities[group_index, role_index]:.1%}"
            for group_index, group in enumerate(SAMPLING_GROUPS)
            for role_index, role in enumerate(VIEW_ROLES)
        )
        calibrated_cells = ", ".join(
            f"{group}/{role}:{sampler.calibrated_cell_probabilities[group_index, role_index]:.1%}"
            for group_index, group in enumerate(SAMPLING_GROUPS)
            for role_index, role in enumerate(VIEW_ROLES)
        )
        dataset_mix = ", ".join(
            f"{name}:{probability:.1%}" for name, probability in sampler.dataset_probabilities.items()
        )
        lines.extend(
            [
                f"  Bucket T={key[0]}, stride={key[1]}, batch/rank={sampler.batch_size}",
                f"    Base cells = {base_cells}",
                f"    Calibrated cells = {calibrated_cells}",
                f"    Logical datasets = {dataset_mix}",
            ]
        )
    return "\n".join(lines)
