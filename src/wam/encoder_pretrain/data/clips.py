"""Raw ego-video dataloaders for LeRobot v3 datasets.

This module is intentionally independent from ``lerobot_datasets.py``.  It is
for large-scale ego-video pretraining where the primary unit is a short video
clip, not a VLA action-supervision sample with embodiment metadata.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from bisect import bisect_right
from collections import deque
from pathlib import Path
from types import GeneratorType
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from wam.encoder_pretrain.data.catalog import load_or_build_ego_video_catalog
from wam.encoder_pretrain.data.video import get_frames_by_timestamps

REJECTED_VIDEOS_FILENAME = "rejected_videos.json"


def ego_video_collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep raw PIL frames and metadata as a Python list."""

    return batch


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _read_existing_columns(path: Path, requested_columns: list[str]) -> pd.DataFrame:
    """Read only columns that exist in a parquet file."""

    try:
        import pyarrow.parquet as pq

        existing = set(pq.read_schema(path).names)
        columns = [col for col in requested_columns if col in existing]
        return pd.read_parquet(path, columns=columns)
    except ImportError:
        frame = pd.read_parquet(path)
        return frame[[col for col in requested_columns if col in frame.columns]]


def _first_text(value: Any) -> str:
    """Extract a useful text string from LeRobot task fields."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        left = _first_text(value.get("left"))
        right = _first_text(value.get("right"))
        if left and right and left != right:
            return f"Left hand: {left} Right hand: {right}"
        return left or right or _first_text(next(iter(value.values()), ""))
    if isinstance(value, np.ndarray):
        return _first_text(value.tolist())
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return ""
    return str(value).strip()


def _infer_video_key(info: dict[str, Any]) -> str:
    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    if len(video_keys) != 1:
        raise ValueError(
            f"video_key must be configured when a dataset has {len(video_keys)} video features: {video_keys}"
        )
    return video_keys[0]


def _resolve_video_key(
    info: dict[str, Any],
    video_key: str | None,
    video_key_candidates: list[str] | tuple[str, ...] | None = None,
) -> str:
    if video_key is not None:
        return str(video_key)
    if video_key_candidates is not None:
        features = info.get("features", {})
        for candidate in video_key_candidates:
            if features.get(str(candidate), {}).get("dtype") == "video":
                return str(candidate)
        raise KeyError(
            f"None of video_key_candidates are video features: {list(video_key_candidates)}"
        )
    return _infer_video_key(info)


def _infer_fps(info: dict[str, Any], video_key: str) -> float:
    video_feature = info.get("features", {}).get(video_key, {})
    video_info = video_feature.get("info", {}) or video_feature.get("video_info", {})
    if "video.fps" in video_info:
        return float(video_info["video.fps"])
    return float(info.get("fps", 30.0))


class EgoVideoClipDataset(Dataset):
    """A single LeRobot v3 ego-video dataset exposed as short raw clips."""

    def __init__(
        self,
        dataset_path: str | Path,
        dataset_name: str | None = None,
        video_key: str | None = None,
        video_key_candidates: list[str] | tuple[str, ...] | None = None,
        fps: float | None = None,
        target_fps: float | None = None,
        clip_len: int = 8,
        frame_stride: int = 1,
        clip_stride: int = 1,
        resize: list[int] | tuple[int, int] | None = (224, 224),
        video_backend: str = "decord",
        video_backend_kwargs: dict[str, Any] | None = None,
        include_action: bool = False,
        include_state: bool = False,
        action_key: str = "action",
        state_key: str | None = None,
    ) -> None:
        self.dataset_path = _as_path(dataset_path)
        self.dataset_name = dataset_name or self.dataset_path.name
        self.clip_len = int(clip_len)
        self.frame_stride = int(frame_stride)
        self.clip_stride = int(clip_stride)
        self._disjoint_window_span: int | None = None
        self._disjoint_bucket_index = 0
        self._disjoint_num_buckets = 1
        self._disjoint_seed = 0
        self.video_backend = video_backend
        self.video_backend_kwargs = dict(video_backend_kwargs or {})
        self.include_action = bool(include_action)
        self.include_state = bool(include_state)
        self.action_key = action_key
        self.state_key = state_key
        self.resize = tuple(resize) if resize is not None else None

        if self.clip_len <= 0:
            raise ValueError("clip_len must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.clip_stride <= 0:
            raise ValueError("clip_stride must be positive")
        if self.include_state and not self.state_key:
            raise ValueError(
                f"include_state=True for {self.dataset_name}, but no state_key was provided"
            )

        self.info = self._read_info()
        self.video_key = _resolve_video_key(self.info, video_key, video_key_candidates)
        self.fps = float(
            fps if fps is not None else _infer_fps(self.info, self.video_key)
        )
        self.target_fps = float(target_fps) if target_fps is not None else None
        if self.target_fps is not None and self.target_fps > self.fps:
            raise ValueError(
                f"target_fps={self.target_fps} exceeds source fps={self.fps} for {self.dataset_name}; "
                "upsampling would duplicate frames"
            )
        self.data_path_pattern = self.info.get(
            "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        )
        self.video_path_pattern = self.info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
        if self.video_key not in self.info.get("features", {}):
            raise KeyError(
                f"{self.video_key!r} is not a feature in {self.dataset_path / 'meta/info.json'}"
            )

        self.rejected_video_paths = self._load_rejected_video_paths()
        self.task_map = self._load_task_map()
        self.episodes: list[dict[str, Any]] = []
        self._clip_cumulative_counts: np.ndarray
        self._build_clip_index()

        self._cached_data_path: Path | None = None
        self._cached_data_frame: pd.DataFrame | None = None

    def _read_info(self) -> dict[str, Any]:
        with open(self.dataset_path / "meta/info.json", "r") as f:
            return json.load(f)

    def _load_rejected_video_paths(self) -> set[str]:
        rejection_path = self.dataset_path / "meta" / REJECTED_VIDEOS_FILENAME
        if not rejection_path.is_file():
            return set()
        payload = json.loads(rejection_path.read_text())
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported rejected-video manifest: {rejection_path}")
        manifest_video_key = payload.get("video_key")
        if manifest_video_key not in {None, self.video_key}:
            raise ValueError(
                f"Rejected-video manifest key {manifest_video_key!r} does not match {self.video_key!r}: "
                f"{rejection_path}"
            )
        return {str(item["path"]) for item in payload.get("videos", [])}

    def _load_task_map(self) -> dict[int, str]:
        task_path = self.dataset_path / "meta/tasks.parquet"
        if not task_path.exists():
            return {}

        tasks = pd.read_parquet(task_path)
        task_map: dict[int, str] = {}
        if "task_index" not in tasks.columns:
            return task_map

        for index_value, row in tasks.iterrows():
            text = _first_text(row["task"] if "task" in tasks.columns else index_value)
            if text:
                task_map[int(row["task_index"])] = text
        return task_map

    @property
    def _required_span(self) -> int:
        return (self.clip_len - 1) * self.frame_stride + 1

    def _sampling_length(self, source_length: int) -> int:
        if self.target_fps is None:
            return int(source_length)
        return math.floor((source_length - 1) * self.target_fps / self.fps) + 1

    def _clip_count(self, source_length: int) -> int:
        max_start = self._sampling_length(source_length) - self._required_span
        return 0 if max_start < 0 else max_start // self.clip_stride + 1

    def _disjoint_window_slot(self, episode_index: int) -> int:
        phase = (
            self._disjoint_seed + episode_index * 2_654_435_761
        ) % self._disjoint_num_buckets
        return (self._disjoint_bucket_index + phase) % self._disjoint_num_buckets

    def _episode_clip_count(self, episode_index: int, source_length: int) -> int:
        window_span = getattr(self, "_disjoint_window_span", None)
        if window_span is None:
            return self._clip_count(source_length)

        first_start = self._disjoint_window_slot(episode_index) * window_span
        max_start = self._sampling_length(source_length) - self._required_span
        if max_start < first_start:
            return 0
        return (max_start - first_start) // (
            window_span * self._disjoint_num_buckets
        ) + 1

    def _episode_clip_start(self, episode_index: int, local_clip_index: int) -> int:
        window_span = getattr(self, "_disjoint_window_span", None)
        if window_span is None:
            return local_clip_index * self.clip_stride
        window_index = (
            self._disjoint_window_slot(episode_index)
            + local_clip_index * self._disjoint_num_buckets
        )
        return window_index * window_span

    def _episode_files(self) -> list[Path]:
        files = sorted((self.dataset_path / "meta/episodes").glob("chunk-*/*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No LeRobot v3 episode parquet files found under {self.dataset_path / 'meta/episodes'}"
            )
        return files

    def _build_clip_index(self) -> None:
        video_prefix = f"videos/{self.video_key}"
        requested_columns = [
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            "tasks",
            "task_descriptions",
            f"{video_prefix}/chunk_index",
            f"{video_prefix}/file_index",
            f"{video_prefix}/from_timestamp",
            f"{video_prefix}/to_timestamp",
        ]

        clip_counts: list[int] = []

        for episode_file in self._episode_files():
            episodes = _read_existing_columns(episode_file, requested_columns)
            required_columns = {
                "episode_index",
                "length",
                "data/chunk_index",
                "data/file_index",
                f"{video_prefix}/chunk_index",
                f"{video_prefix}/file_index",
                f"{video_prefix}/from_timestamp",
            }
            missing = sorted(required_columns - set(episodes.columns))
            if missing:
                raise KeyError(f"{episode_file} is missing required columns: {missing}")

            for row in episodes.to_dict("records"):
                length = int(row["length"])
                clip_count = self._clip_count(length)
                if clip_count == 0:
                    continue

                video_chunk_index = int(row[f"{video_prefix}/chunk_index"])
                video_file_index = int(row[f"{video_prefix}/file_index"])
                relative_video_path = self.video_path_pattern.format(
                    video_key=self.video_key,
                    chunk_index=video_chunk_index,
                    file_index=video_file_index,
                )
                if Path(relative_video_path).as_posix() in self.rejected_video_paths:
                    continue
                video_path = self.dataset_path / relative_video_path

                lang = _first_text(row.get("task_descriptions")) or _first_text(
                    row.get("tasks")
                )
                if not lang and self.task_map:
                    task_texts = row.get("tasks")
                    if (
                        isinstance(task_texts, (list, tuple, np.ndarray))
                        and len(task_texts) > 0
                    ):
                        lang = _first_text(task_texts[0])
                if not lang:
                    lang = self.dataset_name

                episode = {
                    "episode_index": int(row["episode_index"]),
                    "length": length,
                    "data_chunk_index": int(row["data/chunk_index"]),
                    "data_file_index": int(row["data/file_index"]),
                    "dataset_from_index": (
                        int(row["dataset_from_index"])
                        if "dataset_from_index" in row
                        else None
                    ),
                    "dataset_to_index": (
                        int(row["dataset_to_index"])
                        if "dataset_to_index" in row
                        else None
                    ),
                    "video_chunk_index": video_chunk_index,
                    "video_file_index": video_file_index,
                    "video_from_timestamp": float(
                        row[f"{video_prefix}/from_timestamp"]
                    ),
                    "video_to_timestamp": (
                        float(row[f"{video_prefix}/to_timestamp"])
                        if f"{video_prefix}/to_timestamp" in row
                        and row[f"{video_prefix}/to_timestamp"] is not None
                        else None
                    ),
                    "video_path": video_path,
                    "lang": lang,
                }
                self.episodes.append(episode)
                clip_counts.append(clip_count)

        if not clip_counts:
            raise ValueError(
                f"No valid clips found for {self.dataset_name}. "
                f"Check clip_len={self.clip_len}, frame_stride={self.frame_stride}, and video_key={self.video_key!r}."
            )
        self._clip_cumulative_counts = np.cumsum(
            np.asarray(clip_counts, dtype=np.int64), dtype=np.int64
        )

    def _rebuild_clip_index(self) -> None:
        """Build an episode-level prefix sum instead of materializing every clip."""

        counts = np.fromiter(
            (
                self._episode_clip_count(episode_index, int(episode["length"]))
                for episode_index, episode in enumerate(self.episodes)
            ),
            dtype=np.int64,
            count=len(self.episodes),
        )
        self._clip_cumulative_counts = np.cumsum(counts, dtype=np.int64)

    def _clip_location(self, index: int) -> tuple[int, int]:
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(index)
        episode_index = int(
            np.searchsorted(self._clip_cumulative_counts, index, side="right")
        )
        previous = (
            0
            if episode_index == 0
            else int(self._clip_cumulative_counts[episode_index - 1])
        )
        clip_start = self._episode_clip_start(episode_index, index - previous)
        return episode_index, clip_start

    def __len__(self) -> int:
        if len(self._clip_cumulative_counts) == 0:
            return 0
        return int(self._clip_cumulative_counts[-1])

    def _data_path(self, episode: dict[str, Any]) -> Path:
        return self.dataset_path / self.data_path_pattern.format(
            chunk_index=episode["data_chunk_index"],
            file_index=episode["data_file_index"],
        )

    def _load_episode_data(self, episode: dict[str, Any]) -> pd.DataFrame:
        data_path = self._data_path(episode)
        if self._cached_data_path != data_path or self._cached_data_frame is None:
            columns = ["episode_index", "frame_index"]
            if self.include_action:
                columns.append(self.action_key)
            if self.include_state and self.state_key:
                columns.append(self.state_key)
            self._cached_data_frame = pd.read_parquet(data_path, columns=columns)
            self._cached_data_path = data_path

        data = self._cached_data_frame
        assert data is not None
        episode_data = data.loc[
            data["episode_index"] == episode["episode_index"]
        ].copy()
        if "frame_index" in episode_data.columns:
            episode_data = episode_data.sort_values("frame_index")
        return episode_data.reset_index(drop=True)

    def _frames_to_pil(self, frames: np.ndarray) -> list[Image.Image]:
        pil_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            image = Image.fromarray(frame).convert("RGB")
            if self.resize is not None:
                image = image.resize(self.resize)
            pil_frames.append(image)
        return pil_frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, clip_start = self._clip_location(index)
        episode = self.episodes[episode_index]
        frame_positions = np.arange(self.clip_len, dtype=np.int64) * self.frame_stride
        sample_indices = clip_start + frame_positions
        sampling_fps = self.target_fps or self.fps
        timestamps = (
            episode["video_from_timestamp"]
            + sample_indices.astype(np.float64) / sampling_fps
        )
        frame_indices = np.rint(
            sample_indices.astype(np.float64) * self.fps / sampling_fps
        ).astype(np.int64)

        frames = get_frames_by_timestamps(
            str(episode["video_path"]),
            timestamps,
            video_backend=self.video_backend,
            video_backend_kwargs=getattr(self, "video_backend_kwargs", {}),
        )
        if frames.shape[0] != self.clip_len:
            raise RuntimeError(
                f"Expected {self.clip_len} frames from {episode['video_path']}, got {frames.shape[0]}"
            )

        sample = {
            "frames": self._frames_to_pil(frames),
            "lang": episode["lang"],
            "dataset_name": self.dataset_name,
            "episode_index": episode["episode_index"],
            "frame_indices": frame_indices,
            "frame_positions": frame_positions,
            "timestamps": timestamps,
            "video_path": str(episode["video_path"]),
            "metadata": {
                "video_key": self.video_key,
                "fps": self.fps,
                "target_fps": self.target_fps,
                "clip_start_index": clip_start,
                "clip_len": self.clip_len,
                "frame_stride": self.frame_stride,
                "episode_length": episode["length"],
                "data_path": str(self._data_path(episode)),
                "video_from_timestamp": episode["video_from_timestamp"],
                "video_to_timestamp": episode["video_to_timestamp"],
            },
        }

        if self.include_action or self.include_state:
            episode_data = self._load_episode_data(episode)
            selected = episode_data.iloc[frame_indices]
            if self.include_action:
                sample["action"] = np.stack(
                    selected[self.action_key].to_numpy()
                ).astype(np.float32)
            if self.include_state and self.state_key:
                sample["state"] = np.stack(selected[self.state_key].to_numpy()).astype(
                    np.float32
                )

        return sample

    def with_clip_spec(self, clip_len: int, frame_stride: int) -> "EgoVideoClipDataset":
        """Create a cheap bucket view sharing the immutable episode catalog."""

        clone = copy.copy(self)
        clone.clip_len = int(clip_len)
        clone.frame_stride = int(frame_stride)
        if clone.clip_len <= 0 or clone.frame_stride <= 0:
            raise ValueError("clip_len and frame_stride must be positive")

        clone._rebuild_clip_index()
        clone._cached_data_path = None
        clone._cached_data_frame = None
        return clone

    def with_disjoint_clip_spec(
        self,
        clip_len: int,
        frame_stride: int,
        *,
        window_span: int,
        bucket_index: int,
        num_buckets: int,
        seed: int,
    ) -> "EgoVideoClipDataset":
        """Create a bucket view whose temporal windows are disjoint from every peer bucket view."""

        clone = copy.copy(self)
        clone.clip_len = int(clip_len)
        clone.frame_stride = int(frame_stride)
        clone._disjoint_window_span = int(window_span)
        clone._disjoint_bucket_index = int(bucket_index)
        clone._disjoint_num_buckets = int(num_buckets)
        clone._disjoint_seed = int(seed)
        if clone.clip_len <= 0 or clone.frame_stride <= 0:
            raise ValueError("clip_len and frame_stride must be positive")
        if clone._disjoint_window_span < clone._required_span:
            raise ValueError(
                f"window_span={clone._disjoint_window_span} cannot fit required span {clone._required_span}"
            )
        if not 0 <= clone._disjoint_bucket_index < clone._disjoint_num_buckets:
            raise ValueError("bucket_index must be in [0, num_buckets)")

        clone._rebuild_clip_index()
        clone._cached_data_path = None
        clone._cached_data_frame = None
        return clone


class EgoVideoCollectionDataset(Dataset):
    """One logical sampling source backed by multiple LeRobot v3 datasets."""

    def __init__(self, dataset_name: str, datasets: list[EgoVideoClipDataset]) -> None:
        if not datasets:
            raise ValueError(
                f"Ego-video collection {dataset_name!r} requires at least one physical dataset"
            )
        self.dataset_name = str(dataset_name)
        self.datasets = datasets
        self._cumulative_lengths = np.cumsum(
            [len(dataset) for dataset in datasets], dtype=np.int64
        )
        self.num_episodes = sum(len(dataset.episodes) for dataset in datasets)
        self.total_frames = sum(
            int(episode["length"])
            for dataset in datasets
            for episode in dataset.episodes
        )
        self.duration_hours = sum(
            sum(int(episode["length"]) for episode in dataset.episodes)
            / dataset.fps
            / 3600.0
            for dataset in datasets
        )
        self.fps = datasets[0].fps
        self.target_fps = datasets[0].target_fps
        video_keys = {dataset.video_key for dataset in datasets}
        self.video_key = next(iter(video_keys)) if len(video_keys) == 1 else "mixed"

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index = bisect_right(self._cumulative_lengths, index)
        previous = (
            0
            if dataset_index == 0
            else int(self._cumulative_lengths[dataset_index - 1])
        )
        return self.datasets[dataset_index][index - previous]

    def with_clip_spec(
        self, clip_len: int, frame_stride: int
    ) -> "EgoVideoCollectionDataset":
        return type(self)(
            self.dataset_name,
            [
                dataset.with_clip_spec(clip_len, frame_stride)
                for dataset in self.datasets
            ],
        )

    def with_disjoint_clip_spec(
        self, clip_len: int, frame_stride: int, **kwargs: Any
    ) -> "EgoVideoCollectionDataset":
        return type(self)(
            self.dataset_name,
            [
                dataset.with_disjoint_clip_spec(clip_len, frame_stride, **kwargs)
                for dataset in self.datasets
            ],
        )


def _expand_dataset_route(route: Any, fallback: Any, data_root_dir: Path) -> list[Path]:
    root = _as_path(
        route.get("path", fallback.get("path", data_root_dir / str(fallback.name)))
    )
    patterns = route.get("subdatasets", fallback.get("subdatasets", None))
    excluded = set(
        route.get("exclude_subdatasets", fallback.get("exclude_subdatasets", []))
    )
    skip_empty = bool(route.get("skip_empty", fallback.get("skip_empty", False)))
    require_success = bool(
        route.get("require_success", fallback.get("require_success", False))
    )
    include_root = bool(route.get("include_root", fallback.get("include_root", False)))

    def route_ready(path: Path) -> bool:
        if not (path / "meta/info.json").is_file():
            return False
        if not require_success:
            return True
        return (path / "_SUCCESS").is_file() and any(
            (path / "meta/episodes").glob("chunk-*/*.parquet")
        )

    if patterns is None:
        if require_success and not route_ready(root):
            raise FileNotFoundError(f"LeRobot v3 dataset is not complete: {root}")
        return [root]

    root_ready = route_ready(root)
    paths = [root] if include_root and root_ready else []
    for pattern in patterns:
        for path in sorted(root.glob(str(pattern))):
            if path.name in excluded or not route_ready(path):
                continue
            if skip_empty:
                with (path / "meta/info.json").open("r") as handle:
                    if int(json.load(handle).get("total_episodes", 0)) == 0:
                        continue
            paths.append(path)
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError(f"No LeRobot v3 datasets matched route under {root}")
    return paths


def _entry_dataset_paths(entry: Any, data_root_dir: Path) -> list[Path]:
    routes = entry.get("paths", None)
    if routes is None:
        return _expand_dataset_route(entry, entry, data_root_dir)
    return [
        path
        for route in routes
        for path in _expand_dataset_route(route, entry, data_root_dir)
    ]


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.read_schema(path).names)
    except ImportError:
        return set(pd.read_parquet(path).columns)


def _preflight_dataset_entries(
    data_cfg: Any, only_dataset: str | None
) -> list[tuple[Any, list[Path]]]:
    """Validate every selected physical dataset before any clip catalog is built."""

    data_root_dir = _as_path(data_cfg.data_root_dir)
    selected: list[tuple[Any, list[Path]]] = []
    for entry in data_cfg.datasets:
        if only_dataset is not None and entry.name != only_dataset:
            continue
        dataset_paths = _entry_dataset_paths(entry, data_root_dir)
        for dataset_path in dataset_paths:
            info_path = dataset_path / "meta/info.json"
            with info_path.open("r") as handle:
                info = json.load(handle)
            video_key = _resolve_video_key(
                info,
                entry.get("video_key", None),
                entry.get("video_key_candidates", None),
            )
            if video_key not in info.get("features", {}):
                raise KeyError(f"{video_key!r} is not a feature in {info_path}")

            fps = float(entry.get("fps", None) or _infer_fps(info, video_key))
            target_fps = entry.get("target_fps", data_cfg.get("target_fps", None))
            if target_fps is not None and float(target_fps) > fps:
                raise ValueError(
                    f"target_fps={target_fps} exceeds source fps={fps} for {entry.name} at {dataset_path}"
                )

            episode_files = sorted(
                (dataset_path / "meta/episodes").glob("chunk-*/*.parquet")
            )
            if not episode_files:
                episode_root = dataset_path / "meta/episodes"
                raise FileNotFoundError(
                    f"No LeRobot v3 episode parquet files found under {episode_root}"
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
            missing = sorted(required_columns - _parquet_columns(episode_files[0]))
            if missing:
                raise KeyError(
                    f"{episode_files[0]} is missing required columns: {missing}"
                )
        selected.append((entry, dataset_paths))

    if not selected:
        raise ValueError(f"No ego-video dataset matched only_dataset={only_dataset!r}")
    return selected


class EgoVideoMixtureDataset(Dataset):
    """A deterministic concatenation of multiple ego-video clip datasets."""

    def __init__(
        self, datasets: list[EgoVideoClipDataset | EgoVideoCollectionDataset]
    ) -> None:
        if not datasets:
            raise ValueError("EgoVideoMixtureDataset requires at least one dataset")
        self.datasets = datasets
        self._cumulative_lengths = np.cumsum(
            [len(dataset) for dataset in datasets], dtype=np.int64
        )

    @classmethod
    def from_config(
        cls, data_cfg: Any, only_dataset: str | None = None
    ) -> "EgoVideoMixtureDataset":
        dataset_entries = data_cfg.get("datasets", None)
        if dataset_entries is None:
            raise ValueError("ego_video_data.datasets must be configured")

        dataset = load_or_build_ego_video_catalog(
            data_cfg,
            only_dataset,
            lambda: cls._from_config_uncached(data_cfg, only_dataset),
        )
        entries = {str(entry.name): entry for entry in data_cfg.datasets}
        global_backend_kwargs = dict(data_cfg.get("video_backend_kwargs", {}))
        for source in dataset.datasets:
            entry = entries[source.dataset_name]
            backend_kwargs = global_backend_kwargs | dict(
                entry.get("video_backend_kwargs", {})
            )
            physical_datasets = (
                source.datasets
                if isinstance(source, EgoVideoCollectionDataset)
                else [source]
            )
            for physical_dataset in physical_datasets:
                physical_dataset.video_backend_kwargs = backend_kwargs
        return dataset

    @classmethod
    def _from_config_uncached(
        cls, data_cfg: Any, only_dataset: str | None
    ) -> "EgoVideoMixtureDataset":
        """Build the catalog from source metadata after cache policy is resolved."""

        datasets: list[EgoVideoClipDataset] = []
        for entry, dataset_paths in _preflight_dataset_entries(data_cfg, only_dataset):
            name = entry.name

            physical_datasets = [
                EgoVideoClipDataset(
                    dataset_path=dataset_path,
                    dataset_name=name,
                    video_key=entry.get("video_key", None),
                    video_key_candidates=entry.get("video_key_candidates", None),
                    fps=entry.get("fps", None),
                    target_fps=entry.get(
                        "target_fps", data_cfg.get("target_fps", None)
                    ),
                    clip_len=entry.get("clip_len", data_cfg.get("clip_len", 8)),
                    frame_stride=entry.get(
                        "frame_stride", data_cfg.get("frame_stride", 1)
                    ),
                    clip_stride=entry.get(
                        "clip_stride", data_cfg.get("clip_stride", 1)
                    ),
                    resize=entry.get("resize", data_cfg.get("resize", [224, 224])),
                    video_backend=entry.get(
                        "video_backend", data_cfg.get("video_backend", "decord")
                    ),
                    video_backend_kwargs=(
                        dict(data_cfg.get("video_backend_kwargs", {}))
                        | dict(entry.get("video_backend_kwargs", {}))
                    ),
                    include_action=entry.get(
                        "include_action", data_cfg.get("include_action", False)
                    ),
                    include_state=entry.get(
                        "include_state", data_cfg.get("include_state", False)
                    ),
                    action_key=entry.get(
                        "action_key", data_cfg.get("action_key", "action")
                    ),
                    state_key=entry.get("state_key", data_cfg.get("state_key", None)),
                )
                for dataset_path in dataset_paths
            ]
            datasets.append(
                physical_datasets[0]
                if len(physical_datasets) == 1
                else EgoVideoCollectionDataset(name, physical_datasets)
            )

        return cls(datasets)

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index = bisect_right(self._cumulative_lengths, index)
        previous = (
            0
            if dataset_index == 0
            else int(self._cumulative_lengths[dataset_index - 1])
        )
        return self.datasets[dataset_index][index - previous]

    def with_clip_spec(
        self, clip_len: int, frame_stride: int
    ) -> "EgoVideoMixtureDataset":
        return type(self)(
            [
                dataset.with_clip_spec(clip_len, frame_stride)
                for dataset in self.datasets
            ]
        )

    def with_disjoint_clip_spec(
        self, clip_len: int, frame_stride: int, **kwargs: Any
    ) -> "EgoVideoMixtureDataset":
        return type(self)(
            [
                dataset.with_disjoint_clip_spec(clip_len, frame_stride, **kwargs)
                for dataset in self.datasets
            ]
        )


class _BlockShuffleCursor:
    """Traverse a large integer range without replacement using bounded-memory shuffled blocks."""

    _SEED_MODULUS = (1 << 63) - 1

    def __init__(self, size: int, block_size: int, seed: int) -> None:
        self.size = int(size)
        self.block_size = int(block_size)
        self.seed = int(seed)
        if self.size <= 0:
            raise ValueError("Block-shuffle source size must be positive")
        if self.block_size <= 0:
            raise ValueError("shuffle_block_size must be positive")
        self.num_blocks = math.ceil(self.size / self.block_size)
        self.round = 0
        self.block_position = 0
        self.block_offset = 0
        self._block_order: torch.Tensor | None = None
        self._local_permutation: torch.Tensor | None = None

    def _generator(self, salt: int) -> torch.Generator:
        seed = (self.seed + self.round * 1_000_003 + int(salt)) % self._SEED_MODULUS
        return torch.Generator().manual_seed(seed)

    def _current_block(self) -> tuple[int, int, torch.Tensor]:
        if self._block_order is None:
            self._block_order = torch.randperm(
                self.num_blocks, generator=self._generator(17)
            )
        block_index = int(self._block_order[self.block_position])
        block_start = block_index * self.block_size
        block_length = min(self.block_size, self.size - block_start)
        if self._local_permutation is None:
            self._local_permutation = torch.randperm(
                block_length,
                generator=self._generator(10_007 + block_index * 97),
            )
        return block_start, block_length, self._local_permutation

    def _advance_block(self) -> None:
        self.block_position += 1
        self.block_offset = 0
        self._local_permutation = None
        if self.block_position == self.num_blocks:
            self.round += 1
            self.block_position = 0
            self._block_order = None

    def take(self, count: int) -> torch.Tensor:
        remaining = int(count)
        if remaining < 0:
            raise ValueError("count must be non-negative")
        parts = []
        while remaining:
            block_start, block_length, permutation = self._current_block()
            take = min(remaining, block_length - self.block_offset)
            parts.append(
                permutation[self.block_offset : self.block_offset + take] + block_start
            )
            self.block_offset += take
            remaining -= take
            if self.block_offset == block_length:
                self._advance_block()
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)

    def state_dict(self) -> dict[str, int]:
        return {
            "size": self.size,
            "block_size": self.block_size,
            "round": self.round,
            "block_position": self.block_position,
            "block_offset": self.block_offset,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        if (
            int(state["size"]) != self.size
            or int(state["block_size"]) != self.block_size
        ):
            raise ValueError(
                "Block-shuffle cursor shape differs from checkpoint: "
                f"checkpoint=(size={state['size']}, block_size={state['block_size']}), "
                f"current=(size={self.size}, block_size={self.block_size})"
            )
        self.round = int(state["round"])
        self.block_position = int(state["block_position"])
        self.block_offset = int(state["block_offset"])
        self._block_order = None
        self._local_permutation = None


class _PrefetchTrackedBatchSampler:
    """Commit sampler state only after a prefetched batch reaches the trainer."""

    def _init_prefetch_tracking(self) -> None:
        self._prefetch_tracking = False
        self._pending_states: deque[dict[str, Any]] = deque()
        self._committed_state: dict[str, Any] | None = None

    def _snapshot_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def _restore_state(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

    def enable_prefetch_tracking(self) -> None:
        self._prefetch_tracking = True
        self._pending_states.clear()
        self._committed_state = self._snapshot_state()

    def _record_yielded_state(self) -> None:
        if self._prefetch_tracking:
            self._pending_states.append(self._snapshot_state())

    def mark_consumed(self, count: int = 1) -> None:
        for _ in range(int(count)):
            if not self._pending_states:
                raise RuntimeError(
                    "Sampler consumption advanced beyond its prefetched state queue"
                )
            self._committed_state = self._pending_states.popleft()

    def state_dict(self) -> dict[str, Any]:
        state = (
            self._committed_state if self._prefetch_tracking else self._snapshot_state()
        )
        assert state is not None
        return copy.deepcopy(state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._restore_state(state)
        self._pending_states.clear()
        if self._prefetch_tracking:
            self._committed_state = self._snapshot_state()


class BalancedDatasetBatchSampler(_PrefetchTrackedBatchSampler, Sampler[list[int]]):
    """Yield fixed-ratio batches from every dataset in a mixture.

    Each epoch is long enough to traverse the largest source once. Smaller
    sources reshuffle and repeat. The compact ``epoch/batch_offset`` state is
    sufficient to reconstruct the exact index sequence after resume.
    """

    def __init__(
        self,
        dataset: EgoVideoMixtureDataset,
        batch_size: int,
        shuffle_block_size: int = 262_144,
        seed: int = 42,
    ) -> None:
        self.dataset_sizes = [len(source) for source in dataset.datasets]
        self.dataset_offsets = [
            0,
            *[int(value) for value in dataset._cumulative_lengths[:-1]],
        ]
        self.num_sources = len(self.dataset_sizes)
        self.batch_size = int(batch_size)
        if self.batch_size % self.num_sources:
            raise ValueError(
                f"Balanced batch_size={self.batch_size} must be divisible by {self.num_sources} datasets"
            )
        self.samples_per_source = self.batch_size // self.num_sources
        self.num_batches = max(
            (size + self.samples_per_source - 1) // self.samples_per_source
            for size in self.dataset_sizes
        )
        self.seed = int(seed)
        self.epoch = 0
        self.batch_offset = 0
        self.cursors = [
            _BlockShuffleCursor(
                size, shuffle_block_size, self.seed + source_index * 10_009
            )
            for source_index, size in enumerate(self.dataset_sizes)
        ]
        self._init_prefetch_tracking()

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for batch_index in range(self.batch_offset, self.num_batches):
            batch = torch.cat(
                [
                    cursor.take(self.samples_per_source)
                    + self.dataset_offsets[source_index]
                    for source_index, cursor in enumerate(self.cursors)
                ]
            ).tolist()
            self.batch_offset = batch_index + 1
            self._record_yielded_state()
            yield batch
        self.epoch += 1
        self.batch_offset = 0

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "format": "block_shuffle.v1",
            "epoch": self.epoch,
            "batch_offset": self.batch_offset,
            "cursors": [cursor.state_dict() for cursor in self.cursors],
        }

    def _restore_state(self, state: dict[str, Any]) -> None:
        if state.get("format") != "block_shuffle.v1":
            raise ValueError(
                f"Unsupported balanced sampler state: {state.get('format')}"
            )
        if len(state["cursors"]) != len(self.cursors):
            raise ValueError(
                f"Checkpoint has {len(state['cursors'])} source cursors, current sampler has {len(self.cursors)}"
            )
        self.epoch = int(state["epoch"])
        self.batch_offset = int(state["batch_offset"])
        for cursor, cursor_state in zip(self.cursors, state["cursors"], strict=True):
            cursor.load_state_dict(cursor_state)


class TemperatureDatasetBatchSampler(_PrefetchTrackedBatchSampler, Sampler[list[int]]):
    """Sample datasets with ``n_d ** temperature`` and samples uniformly within each source."""

    def __init__(
        self,
        dataset: EgoVideoMixtureDataset,
        batch_size: int,
        temperature: float = 0.5,
        dataset_weights: list[float] | None = None,
        shuffle_block_size: int = 262_144,
        seed: int = 42,
    ) -> None:
        self.dataset_sizes = [len(source) for source in dataset.datasets]
        self.dataset_offsets = [
            0,
            *[int(value) for value in dataset._cumulative_lengths[:-1]],
        ]
        if dataset_weights is None:
            weights = np.asarray(self.dataset_sizes, dtype=np.float64) ** float(
                temperature
            )
        else:
            if len(dataset_weights) != len(self.dataset_sizes):
                raise ValueError(
                    f"dataset_weights has {len(dataset_weights)} entries for {len(self.dataset_sizes)} datasets"
                )
            weights = np.asarray(dataset_weights, dtype=np.float64)
        sizes = np.asarray(self.dataset_sizes, dtype=np.int64)
        if np.any(weights[sizes > 0] <= 0):
            raise ValueError("All non-empty dataset sampling weights must be positive")
        weights[sizes == 0] = 0.0
        if weights.sum() <= 0:
            raise ValueError(
                "Temperature sampling requires at least one non-empty dataset"
            )
        self.probabilities = torch.tensor(weights / weights.sum(), dtype=torch.float64)
        self.batch_size = int(batch_size)
        self.num_batches = math.ceil(sum(self.dataset_sizes) / self.batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_offset = 0
        self.cursors = [
            (
                _BlockShuffleCursor(
                    size, shuffle_block_size, self.seed + source_index * 10_009
                )
                if size
                else None
            )
            for source_index, size in enumerate(self.dataset_sizes)
        ]
        self._init_prefetch_tracking()

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for batch_index in range(self.batch_offset, self.num_batches):
            generator = torch.Generator().manual_seed(
                self.seed + self.epoch * 1_000_003 + batch_index
            )
            sources = torch.multinomial(
                self.probabilities,
                self.batch_size,
                replacement=True,
                generator=generator,
            )
            batch = torch.empty(self.batch_size, dtype=torch.long)
            for source_index, cursor in enumerate(self.cursors):
                positions = torch.nonzero(
                    sources == source_index, as_tuple=False
                ).flatten()
                if len(positions):
                    assert cursor is not None
                    batch[positions] = (
                        cursor.take(len(positions)) + self.dataset_offsets[source_index]
                    )
            self.batch_offset = batch_index + 1
            self._record_yielded_state()
            yield batch.tolist()
        self.epoch += 1
        self.batch_offset = 0

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "format": "block_shuffle.v1",
            "epoch": self.epoch,
            "batch_offset": self.batch_offset,
            "cursors": [
                cursor.state_dict() if cursor is not None else None
                for cursor in self.cursors
            ],
        }

    def _restore_state(self, state: dict[str, Any]) -> None:
        if state.get("format") != "block_shuffle.v1":
            raise ValueError(
                f"Unsupported temperature sampler state: {state.get('format')}"
            )
        if len(state["cursors"]) != len(self.cursors):
            raise ValueError(
                f"Checkpoint has {len(state['cursors'])} source cursors, current sampler has {len(self.cursors)}"
            )
        self.epoch = int(state["epoch"])
        self.batch_offset = int(state["batch_offset"])
        for cursor, cursor_state in zip(self.cursors, state["cursors"], strict=True):
            if cursor is not None:
                cursor.load_state_dict(cursor_state)


class BucketedEgoVideoDataset(Dataset):
    """Route sampler-tagged indices to lightweight homogeneous clip views."""

    def __init__(self, datasets: dict[tuple[int, int], EgoVideoMixtureDataset]) -> None:
        if not datasets:
            raise ValueError("At least one bucket dataset is required")
        self.datasets = dict(datasets)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets.values())

    def __getitem__(self, request: tuple[tuple[int, int], int]) -> dict[str, Any]:
        key, index = request
        return self.datasets[key][int(index)]


class SharedBucketBatchSampler(
    _PrefetchTrackedBatchSampler, Sampler[list[tuple[tuple[int, int], int]]]
):
    """Schedule homogeneous bucket batches through one shared worker pool.

    Accelerate shards non-split batch samplers by consuming ``world_size``
    consecutive batches. Every such group therefore uses one bucket while its
    per-bucket sampler advances to distinct indices for each rank.
    """

    def __init__(
        self,
        samplers: dict[tuple[int, int], Sampler[list[int]]],
        bucket_probabilities: dict[tuple[int, int], float],
        *,
        world_size: int,
        seed: int,
    ) -> None:
        if not samplers:
            raise ValueError("At least one bucket sampler is required")
        if set(samplers) != set(bucket_probabilities):
            raise ValueError(
                "Bucket samplers and probabilities must have identical keys"
            )
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        total_probability = sum(float(value) for value in bucket_probabilities.values())
        if total_probability <= 0 or any(
            float(value) <= 0 for value in bucket_probabilities.values()
        ):
            raise ValueError("Bucket probabilities must be positive")

        self.samplers = dict(samplers)
        for sampler in self.samplers.values():
            if hasattr(sampler, "enable_prefetch_tracking"):
                sampler.enable_prefetch_tracking()
        self.bucket_probabilities = {
            key: float(value) / total_probability
            for key, value in bucket_probabilities.items()
        }
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.logical_step = 0
        self.rank_slot = 0
        self.num_logical_steps = sum(
            math.ceil(len(sampler) / self.world_size)
            for sampler in self.samplers.values()
        )
        self._iterators: dict[tuple[int, int], Any] = {}
        self._init_prefetch_tracking()

    def __len__(self) -> int:
        return self.num_logical_steps * self.world_size

    def bucket_for_step(self, step: int) -> tuple[int, int]:
        draw = random.Random(self.seed + int(step)).random()
        cumulative = 0.0
        final_key = next(reversed(self.bucket_probabilities))
        for key, probability in self.bucket_probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                return key
        return final_key

    def _next_indices(self, key: tuple[int, int]) -> list[int]:
        iterator = self._iterators.get(key)
        if iterator is None:
            iterator = iter(self.samplers[key])
            self._iterators[key] = iterator
        try:
            indices = next(iterator)
        except StopIteration:
            iterator = iter(self.samplers[key])
            self._iterators[key] = iterator
            indices = next(iterator)

        sampler = self.samplers[key]
        if hasattr(sampler, "mark_consumed"):
            sampler.mark_consumed()
        return indices

    def __iter__(self):
        for _ in range(len(self)):
            key = self.bucket_for_step(self.logical_step)
            indices = self._next_indices(key)
            batch = [(key, int(index)) for index in indices]

            self.rank_slot += 1
            if self.rank_slot == self.world_size:
                self.logical_step += 1
                self.rank_slot = 0
            self._record_yielded_state()
            yield batch

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "format": "shared_bucket_sampler.v1",
            "world_size": self.world_size,
            "logical_step": self.logical_step,
            "rank_slot": self.rank_slot,
            "samplers": {
                f"{length}:{stride}": sampler.state_dict()
                for (length, stride), sampler in self.samplers.items()
            },
        }

    def _restore_state(self, state: dict[str, Any]) -> None:
        if state.get("format") != "shared_bucket_sampler.v1":
            raise ValueError(
                f"Unsupported shared bucket sampler state: {state.get('format')}"
            )
        if int(state["world_size"]) != self.world_size:
            raise ValueError(
                f"Sampler world_size={state['world_size']} differs from current world_size={self.world_size}"
            )
        self.logical_step = int(state["logical_step"])
        self.rank_slot = int(state["rank_slot"])
        for (length, stride), sampler in self.samplers.items():
            sampler.load_state_dict(state["samplers"][f"{length}:{stride}"])
        self._iterators.clear()

    def load_legacy_state(self, state: dict[str, Any], logical_step: int) -> None:
        sampler_states = state.get("samplers", {})
        for (length, stride), sampler in self.samplers.items():
            key = f"{length}:{stride}"
            if key in sampler_states:
                sampler.load_state_dict(sampler_states[key])
        self.logical_step = int(logical_step)
        self.rank_slot = 0
        self._iterators.clear()
        self._pending_states.clear()
        if self._prefetch_tracking:
            self._committed_state = self._snapshot_state()


def _select_ego_video_cfg(cfg: Any) -> Any:
    datasets_cfg = cfg.datasets
    if "ego_video_data" not in datasets_cfg:
        raise KeyError(
            "datasets.ego_video_data must be configured for ego-video dataloading"
        )
    return datasets_cfg.ego_video_data


def make_ego_video_dataloader(cfg: Any, only_dataset: str | None = None) -> DataLoader:
    data_cfg = _select_ego_video_cfg(cfg)
    dataset = EgoVideoMixtureDataset.from_config(data_cfg, only_dataset=only_dataset)
    num_workers = int(data_cfg.get("num_workers", 1))

    dataloader_kwargs = {
        "batch_size": int(data_cfg.get("per_device_batch_size", 2)),
        "num_workers": num_workers,
        "collate_fn": ego_video_collate_fn,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
        "shuffle": bool(data_cfg.get("shuffle", False)),
        "drop_last": bool(data_cfg.get("drop_last", False)),
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(
            data_cfg.get("persistent_workers", False)
        )
        dataloader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))

    return DataLoader(dataset, **dataloader_kwargs)


class UniformClipBucketLoaders:
    """Sample valid pairs while keeping length/stride marginals uniform.

    Some finite episodes cannot support every Cartesian-product pair. Iterative
    proportional fitting creates a balanced coupling over non-empty pairs,
    retaining uniform marginals without padding or duplicated frames.
    """

    def __init__(
        self,
        loaders: dict[tuple[int, int], Any],
        seed: int = 42,
        stateful_samplers: dict[tuple[int, int], Sampler] | None = None,
        length_probabilities: dict[int, float] | None = None,
        stride_probabilities: dict[int, float] | None = None,
        bucket_weights: dict[tuple[int, int], float] | None = None,
    ) -> None:
        active_loaders = {}
        dropped_empty_buckets = []
        for key, loader in loaders.items():
            try:
                is_empty = len(loader) == 0
            except (AttributeError, TypeError):
                is_empty = False
            if is_empty:
                dropped_empty_buckets.append(key)
            else:
                active_loaders[key] = loader

        if not active_loaders:
            raise ValueError("At least one non-empty clip bucket is required")
        self.loaders = active_loaders
        self.dropped_empty_buckets = tuple(sorted(dropped_empty_buckets))
        self.lengths = sorted({key[0] for key in active_loaders})
        self.strides = sorted({key[1] for key in active_loaders})
        self.seed = int(seed)
        self.iterators: dict[tuple[int, int], Any] = {}
        self.stateful_samplers = {
            key: sampler
            for key, sampler in (stateful_samplers or {}).items()
            if key in self.loaders
        }
        for sampler in self.stateful_samplers.values():
            if hasattr(sampler, "enable_prefetch_tracking"):
                sampler.enable_prefetch_tracking()
        self.length_probability_map = dict(length_probabilities or {})
        self.stride_probability_map = dict(stride_probabilities or {})
        self.bucket_weight_map = {
            key: float(weight)
            for key, weight in (bucket_weights or {}).items()
            if key in self.loaders
        }
        self.length_probabilities = self._marginal(self.lengths, length_probabilities)
        self.stride_probabilities = self._marginal(self.strides, stride_probabilities)
        self.bucket_probabilities = (
            self._weighted_probabilities(self.bucket_weight_map)
            if self.bucket_weight_map
            else self._balanced_probabilities()
        )

    @staticmethod
    def _marginal(values: list[int], configured: dict[int, float] | None) -> np.ndarray:
        weights = np.asarray(
            [
                1.0 if configured is None else float(configured[value])
                for value in values
            ],
            dtype=np.float64,
        )
        if np.any(weights <= 0):
            raise ValueError(
                f"Bucket probabilities must be positive, got {weights.tolist()}"
            )
        return weights / weights.sum()

    def state_dict(self) -> dict[str, object]:
        return {
            "samplers": {
                f"{length}:{stride}": sampler.state_dict()
                for (length, stride), sampler in self.stateful_samplers.items()
            }
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        sampler_states = state.get("samplers", {})
        for (length, stride), sampler in self.stateful_samplers.items():
            key = f"{length}:{stride}"
            if key in sampler_states:
                sampler.load_state_dict(sampler_states[key])
        self.iterators.clear()

    def _balanced_probabilities(self) -> dict[tuple[int, int], float]:
        matrix = np.zeros((len(self.lengths), len(self.strides)), dtype=np.float64)
        for row, length in enumerate(self.lengths):
            for column, stride in enumerate(self.strides):
                if (length, stride) in self.loaders:
                    matrix[row, column] = 1.0
        if np.any(matrix.sum(axis=1) == 0) or np.any(matrix.sum(axis=0) == 0):
            raise ValueError(
                "Every configured length and stride bucket needs at least one valid pair"
            )

        target_rows = self.length_probabilities
        target_columns = self.stride_probabilities
        for _ in range(1000):
            matrix *= (target_rows / matrix.sum(axis=1))[:, None]
            matrix *= (target_columns / matrix.sum(axis=0))[None, :]
        if not np.allclose(matrix.sum(axis=1), target_rows, atol=1e-10):
            raise RuntimeError("Could not balance length-bucket marginals")
        if not np.allclose(matrix.sum(axis=0), target_columns, atol=1e-10):
            raise RuntimeError("Could not balance stride-bucket marginals")
        return {
            (length, stride): float(matrix[row, column])
            for row, length in enumerate(self.lengths)
            for column, stride in enumerate(self.strides)
            if matrix[row, column] > 0
        }

    def _weighted_probabilities(
        self, weights: dict[tuple[int, int], float]
    ) -> dict[tuple[int, int], float]:
        missing = self.loaders.keys() - weights.keys()
        if missing:
            raise ValueError(
                f"Missing explicit weights for clip buckets: {sorted(missing)}"
            )
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError(f"Clip bucket weights must be positive, got {weights}")
        total = sum(weights.values())
        return {key: weights[key] / total for key in self.loaders}

    def bucket_for_step(self, step: int) -> tuple[int, int]:
        rng = random.Random(self.seed + int(step))
        draw = rng.random()
        cumulative = 0.0
        final_key = next(reversed(self.bucket_probabilities))
        for key, probability in self.bucket_probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                return key
        return final_key

    def next(self, step: int) -> tuple[tuple[int, int], list[dict[str, Any]]]:
        key = self.bucket_for_step(step)
        iterator = self.iterators.get(key)
        if iterator is None:
            iterator = iter(self.loaders[key])
            self.iterators[key] = iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(self.loaders[key])
            self.iterators[key] = iterator
            batch = next(iterator)
        if batch is None:
            raise RuntimeError(
                f"Clip bucket {key} yielded None. Prepared loaders with no full batches "
                "must be removed before scheduling."
            )
        sampler = self.stateful_samplers.get(key)
        if sampler is not None and hasattr(sampler, "mark_consumed"):
            batch_sampler = self.loaders[key].batch_sampler
            num_processes = int(getattr(batch_sampler, "num_processes", 1))
            split_batches = bool(getattr(batch_sampler, "split_batches", False))
            sampler.mark_consumed(1 if split_batches else num_processes)
        return key, batch


class SharedClipBucketLoader:
    """Expose one prepared DataLoader as the trainer's bucket-stream interface."""

    def __init__(
        self,
        loader: DataLoader,
        sampler: SharedBucketBatchSampler,
        bucket_datasets: dict[tuple[int, int], EgoVideoMixtureDataset],
        worker_restart_interval_batches: int = 0,
    ) -> None:
        if worker_restart_interval_batches < 0:
            raise ValueError("worker_restart_interval_batches must be non-negative")
        self.loader = loader
        self.sampler = sampler
        self.bucket_datasets = dict(bucket_datasets)
        self.worker_restart_interval_batches = int(worker_restart_interval_batches)
        self._batches_since_worker_restart = 0
        self._iterator: Any | None = None

    @property
    def bucket_probabilities(self) -> dict[tuple[int, int], float]:
        return self.sampler.bucket_probabilities

    def with_loader(self, loader: DataLoader) -> "SharedClipBucketLoader":
        return type(self)(
            loader,
            self.sampler,
            self.bucket_datasets,
            worker_restart_interval_batches=self.worker_restart_interval_batches,
        )

    def _restart_workers(self) -> None:
        """Release native decoder heaps and rewind discarded prefetch."""

        committed_state = self.sampler.state_dict()
        iterator = self._iterator
        assert iterator is not None
        if isinstance(iterator, GeneratorType):
            iterator.close()
            self.loader.end()
        worker_loader = (
            self.loader.base_dataloader
            if hasattr(self.loader, "base_dataloader")
            else self.loader
        )
        worker_iterator = worker_loader._iterator
        assert worker_iterator is not None
        worker_iterator._shutdown_workers()
        worker_loader._iterator = None
        self._iterator = None
        del iterator
        self.sampler.load_state_dict(committed_state)
        self._batches_since_worker_restart = 0

    def next(self) -> tuple[tuple[int, int], list[dict[str, Any]]]:
        if self._iterator is None:
            self._iterator = iter(self.loader)
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            batch = next(self._iterator)
        if not batch:
            raise RuntimeError("Shared clip DataLoader yielded an empty batch")

        keys = {
            (
                int(sample["metadata"]["clip_len"]),
                int(sample["metadata"]["frame_stride"]),
            )
            for sample in batch
        }
        if len(keys) != 1:
            raise RuntimeError(
                f"Shared clip DataLoader mixed bucket keys in one batch: {sorted(keys)}"
            )
        self.sampler.mark_consumed(self.sampler.world_size)
        self._batches_since_worker_restart += 1
        if (
            self.worker_restart_interval_batches
            and self._batches_since_worker_restart
            == self.worker_restart_interval_batches
        ):
            self._restart_workers()
        return keys.pop(), batch

    def state_dict(self) -> dict[str, Any]:
        return self.sampler.state_dict()

    def load_state_dict(self, state: dict[str, Any], *, logical_step: int = 0) -> None:
        if state.get("format") == "shared_bucket_sampler.v1":
            self.sampler.load_state_dict(state)
        else:
            self.sampler.load_legacy_state(state, logical_step)
        self._batches_since_worker_restart = 0
        self._iterator = None


def format_ego_video_dataloader_statistics(
    bucket_loaders: UniformClipBucketLoaders,
    cfg: Any,
    *,
    num_processes: int = 1,
    gradient_accumulation_steps: int = 1,
) -> str:
    """Format the effective bucketed sampling configuration for startup logs."""

    data_cfg = _select_ego_video_cfg(cfg)
    first_loader = next(iter(bucket_loaders.loaders.values()))
    sources = first_loader.dataset.datasets
    sampling = str(data_cfg.get("dataset_sampling", "natural"))
    if sampling == "temperature":
        sampling = f"temperature={float(data_cfg.get('dataset_temperature', 0.5)):g}"

    lines = [
        "***** Ego-Video DataLoader Statistics *****",
        (
            f"  Sources = {len(sources)} | Buckets = {len(bucket_loaders.loaders)} | Sampling = {sampling}"
        ),
        (
            f"  Workers/bucket/rank = {first_loader.num_workers} | "
            f"Max workers/rank = {first_loader.num_workers * len(bucket_loaders.loaders)} | "
            f"Pin memory = {first_loader.pin_memory} | Persistent workers = {first_loader.persistent_workers} | "
            f"Prefetch factor = {first_loader.prefetch_factor}"
        ),
        (
            f"  Processes = {num_processes} | Gradient accumulation = {gradient_accumulation_steps} | "
            f"Frame budget/rank = {int(data_cfg.get('per_device_frame_budget', 24))}"
        ),
    ]

    for source in sources:
        if isinstance(source, EgoVideoCollectionDataset):
            num_episodes = source.num_episodes
            source_frames = source.total_frames
            duration_hours = source.duration_hours
        else:
            num_episodes = len(source.episodes)
            source_frames = sum(int(episode["length"]) for episode in source.episodes)
            duration_hours = source_frames / source.fps / 3600.0
        target_fps = source.target_fps if source.target_fps is not None else source.fps
        lines.append(
            f"  Source {source.dataset_name}: episodes={num_episodes:,}, "
            f"frames={source_frames:,}, duration={duration_hours:.2f}h, "
            f"fps={source.fps:g}->{target_fps:g}, video_key={source.video_key}"
        )

    expected_samples_per_micro_step = 0.0
    expected_frames_per_micro_step = 0.0
    for (length, stride), loader in sorted(bucket_loaders.loaders.items()):
        source_sizes = [len(source) for source in loader.dataset.datasets]
        batch_sampler = loader.batch_sampler
        batch_size = int(loader.batch_size or batch_sampler.batch_size)
        if hasattr(batch_sampler, "probabilities"):
            source_probabilities = batch_sampler.probabilities.tolist()
        elif hasattr(batch_sampler, "samples_per_source"):
            source_probabilities = [1.0 / len(source_sizes)] * len(source_sizes)
        else:
            total_clips = sum(source_sizes)
            source_probabilities = [size / total_clips for size in source_sizes]
        bucket_probability = bucket_loaders.bucket_probabilities[(length, stride)]
        expected_samples_per_micro_step += bucket_probability * batch_size
        expected_frames_per_micro_step += bucket_probability * batch_size * length
        clip_summary = ", ".join(
            f"{source.dataset_name}:{size:,}"
            for source, size in zip(loader.dataset.datasets, source_sizes, strict=True)
        )
        mix_summary = ", ".join(
            f"{source.dataset_name}:{probability:.1%}"
            for source, probability in zip(
                loader.dataset.datasets, source_probabilities, strict=True
            )
        )
        lines.append(
            f"  Bucket T={length}, stride={stride}: p={bucket_probability:.1%}, "
            f"clips={sum(source_sizes):,} ({clip_summary}), batch/rank={batch_size}, "
            f"frames/rank={batch_size * length}, batches/cycle={len(loader):,}, mix=({mix_summary})"
        )

    effective_multiplier = int(num_processes) * int(gradient_accumulation_steps)
    lines.append(
        f"  Expected/micro-step/rank: samples={expected_samples_per_micro_step:.2f}, "
        f"frames={expected_frames_per_micro_step:.2f}"
    )
    lines.append(
        f"  Expected/optimization-step/global: "
        f"samples={expected_samples_per_micro_step * effective_multiplier:.2f}, "
        f"frames={expected_frames_per_micro_step * effective_multiplier:.2f}"
    )
    if bucket_loaders.dropped_empty_buckets:
        lines.append(
            f"  Dropped empty buckets = {list(bucket_loaders.dropped_empty_buckets)}"
        )
    return "\n".join(lines)


def format_shared_ego_video_dataloader_statistics(
    bucket_loader: SharedClipBucketLoader,
    cfg: Any,
    *,
    num_processes: int,
    gradient_accumulation_steps: int,
) -> str:
    """Format effective sampling and the single shared worker-pool size."""

    data_cfg = _select_ego_video_cfg(cfg)
    first_dataset = next(iter(bucket_loader.bucket_datasets.values()))
    sampling = str(data_cfg.get("dataset_sampling", "natural"))
    if sampling == "temperature":
        sampling = f"temperature={float(data_cfg.get('dataset_temperature', 0.5)):g}"

    lines = [
        "***** Shared Ego-Video DataLoader Statistics *****",
        (
            f"  Sources = {len(first_dataset.datasets)} | Buckets = {len(bucket_loader.bucket_datasets)} | "
            f"Sampling = {sampling}"
        ),
        (
            f"  Workers/rank = {bucket_loader.loader.num_workers} total | "
            f"Pin memory = {bucket_loader.loader.pin_memory} | "
            f"Persistent workers = {bucket_loader.loader.persistent_workers} | "
            f"Prefetch factor = {bucket_loader.loader.prefetch_factor} | "
            f"Worker restart interval = {bucket_loader.worker_restart_interval_batches} batches"
        ),
        (
            f"  Processes = {num_processes} | Gradient accumulation = {gradient_accumulation_steps} | "
            f"Frame budget/rank = {int(data_cfg.get('per_device_frame_budget', 24))}"
        ),
    ]

    sources = first_dataset.datasets
    for source in sources:
        if isinstance(source, EgoVideoCollectionDataset):
            num_episodes = source.num_episodes
            source_frames = source.total_frames
            duration_hours = source.duration_hours
        else:
            num_episodes = len(source.episodes)
            source_frames = sum(int(episode["length"]) for episode in source.episodes)
            duration_hours = source_frames / source.fps / 3600.0
        target_fps = source.target_fps if source.target_fps is not None else source.fps
        lines.append(
            f"  Source {source.dataset_name}: episodes={num_episodes:,}, frames={source_frames:,}, "
            f"duration={duration_hours:.2f}h, fps={source.fps:g}->{target_fps:g}, video_key={source.video_key}"
        )

    expected_samples_per_micro_step = 0.0
    expected_frames_per_micro_step = 0.0
    for key, dataset in sorted(bucket_loader.bucket_datasets.items()):
        length, stride = key
        sampler = bucket_loader.sampler.samplers[key]
        source_sizes = [len(source) for source in dataset.datasets]
        if hasattr(sampler, "probabilities"):
            source_probabilities = sampler.probabilities.tolist()
        else:
            source_probabilities = [1.0 / len(source_sizes)] * len(source_sizes)
        batch_size = int(sampler.batch_size)
        bucket_probability = bucket_loader.bucket_probabilities[key]
        expected_samples_per_micro_step += bucket_probability * batch_size
        expected_frames_per_micro_step += bucket_probability * batch_size * length
        clip_summary = ", ".join(
            f"{source.dataset_name}:{size:,}"
            for source, size in zip(dataset.datasets, source_sizes, strict=True)
        )
        mix_summary = ", ".join(
            f"{source.dataset_name}:{probability:.1%}"
            for source, probability in zip(
                dataset.datasets, source_probabilities, strict=True
            )
        )
        lines.append(
            f"  Bucket T={length}, stride={stride}: p={bucket_probability:.1%}, "
            f"clips={sum(source_sizes):,} ({clip_summary}), batch/rank={batch_size}, "
            f"frames/rank={batch_size * length}, "
            f"global-batches/cycle={math.ceil(len(sampler) / num_processes):,}, mix=({mix_summary})"
        )

    effective_multiplier = int(num_processes) * int(gradient_accumulation_steps)
    lines.append(
        f"  Expected/micro-step/rank: samples={expected_samples_per_micro_step:.2f}, "
        f"frames={expected_frames_per_micro_step:.2f}"
    )
    lines.append(
        f"  Expected/optimization-step/global: "
        f"samples={expected_samples_per_micro_step * effective_multiplier:.2f}, "
        f"frames={expected_frames_per_micro_step * effective_multiplier:.2f}"
    )
    return "\n".join(lines)


def make_bucketed_ego_video_dataloaders(cfg: Any) -> UniformClipBucketLoaders:
    """Build one homogeneous loader for every configured ``(T, stride)`` pair."""

    data_cfg = _select_ego_video_cfg(cfg)
    lengths = [
        int(value) for value in data_cfg.get("length_buckets", [4, 8, 12, 16, 24])
    ]
    strides = [int(value) for value in data_cfg.get("stride_buckets", [1, 2, 4])]
    length_probs = [
        float(value) for value in data_cfg.get("length_probs", [1.0] * len(lengths))
    ]
    stride_probs = [
        float(value) for value in data_cfg.get("stride_probs", [1.0] * len(strides))
    ]
    if len(length_probs) != len(lengths) or len(stride_probs) != len(strides):
        raise ValueError("length_probs/stride_probs must align with their bucket lists")
    frame_budget = int(data_cfg.get("per_device_frame_budget", 24))
    shuffle_block_size = int(data_cfg.get("shuffle_block_size", 262_144))
    if frame_budget < max(lengths):
        raise ValueError(
            f"per_device_frame_budget={frame_budget} cannot fit the largest T bucket {max(lengths)}"
        )

    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    base_data = _select_ego_video_cfg(base_cfg)
    base_data.clip_len = lengths[0]
    base_data.frame_stride = strides[0]
    base_dataset = EgoVideoMixtureDataset.from_config(base_data)

    bucket_keys = [(length, stride) for length in lengths for stride in strides]
    clip_sampling_mode = str(data_cfg.get("clip_sampling_mode", "sliding"))
    if clip_sampling_mode not in {"sliding", "disjoint_window"}:
        raise ValueError(f"Unsupported clip_sampling_mode={clip_sampling_mode!r}")
    configured_window_span = data_cfg.get("unique_window_span", "auto")
    required_window_span = max(
        (length - 1) * stride + 1 for length, stride in bucket_keys
    )
    window_span = (
        required_window_span
        if configured_window_span == "auto"
        else int(configured_window_span)
    )
    if clip_sampling_mode == "disjoint_window" and window_span < required_window_span:
        raise ValueError(
            f"unique_window_span={window_span} cannot fit maximum required span {required_window_span}"
        )

    loaders: dict[tuple[int, int], DataLoader] = {}
    stateful_samplers: dict[tuple[int, int], Sampler] = {}
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
        if sampling == "balanced":
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
                    [float(value) for value in configured_weights]
                    if configured_weights is not None
                    else None
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
            kwargs["persistent_workers"] = bool(
                data_cfg.get("persistent_workers", False)
            )
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


def make_shared_bucketed_ego_video_dataloader(
    cfg: Any, *, world_size: int
) -> SharedClipBucketLoader:
    """Build one worker pool while retaining homogeneous bucket batches."""

    bucket_loaders = make_bucketed_ego_video_dataloaders(cfg)
    if set(bucket_loaders.loaders) != set(bucket_loaders.stateful_samplers):
        raise ValueError(
            "Shared bucket loading requires dataset_sampling=balanced or temperature"
        )

    bucket_datasets = {
        key: loader.dataset for key, loader in bucket_loaders.loaders.items()
    }
    sampler = SharedBucketBatchSampler(
        bucket_loaders.stateful_samplers,
        bucket_loaders.bucket_probabilities,
        world_size=world_size,
        seed=bucket_loaders.seed,
    )
    sampler.enable_prefetch_tracking()

    data_cfg = _select_ego_video_cfg(cfg)
    num_workers = int(data_cfg.get("num_workers", 0))
    worker_restart_interval_batches = int(
        data_cfg.get("worker_restart_interval_batches", 0)
    )
    if worker_restart_interval_batches and num_workers == 0:
        raise ValueError("worker_restart_interval_batches requires num_workers > 0")
    persistent_workers = bool(data_cfg.get("persistent_workers", False))
    if worker_restart_interval_batches and not persistent_workers:
        raise ValueError(
            "worker_restart_interval_batches requires persistent_workers=true"
        )
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
    return SharedClipBucketLoader(
        loader,
        sampler,
        bucket_datasets,
        worker_restart_interval_batches=worker_restart_interval_batches,
    )
