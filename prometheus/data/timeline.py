from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from prometheus.data.episode import EpisodeView
from prometheus.data.layout import timestamp_ms_to_ns


@dataclass(frozen=True, slots=True)
class TimelineSeries:
    key: str
    time_ns: np.ndarray
    values: Any
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TimelineSample:
    key: str
    time_ns: int
    value: Any
    metadata: dict[str, Any]


class TimelineSource(Protocol):
    """UI-facing read API shared by offline and future live sources."""

    def keys(self) -> tuple[str, ...]: ...

    def time_range_ns(self) -> tuple[int, int]: ...

    def series(self, key: str) -> TimelineSeries: ...

    def sample_at(self, key: str, time_ns: int) -> TimelineSample | None: ...


class EpisodeTimelineSource:
    """Build timeline series from an EpisodeView without duplicating layout rules."""

    def __init__(self, episode: EpisodeView):
        self.episode = episode
        self._series: dict[str, TimelineSeries] | None = None

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._load_series()))

    def time_range_ns(self) -> tuple[int, int]:
        non_empty = [series.time_ns for series in self._load_series().values() if len(series.time_ns)]
        if not non_empty:
            return 0, 0
        return min(int(values.min()) for values in non_empty), max(int(values.max()) for values in non_empty)

    def series(self, key: str) -> TimelineSeries:
        return self._load_series()[str(key)]

    def sample_at(self, key: str, time_ns: int) -> TimelineSample | None:
        series = self.series(key)
        if not len(series.time_ns):
            return None
        index = int(np.argmin(np.abs(series.time_ns.astype(np.int64) - int(time_ns))))
        return TimelineSample(
            key=series.key,
            time_ns=int(series.time_ns[index]),
            value=value_at(series.values, index),
            metadata=dict(series.metadata),
        )

    def media_path(self, key: str) -> Path | None:
        media = self.series(key).metadata.get("media")
        if not media:
            return None
        return self.episode.media_path(str(media))

    def _load_series(self) -> dict[str, TimelineSeries]:
        if self._series is None:
            series: dict[str, TimelineSeries] = {}
            self._add_robot_series(series)
            self._add_action_series(series)
            self._add_camera_series(series)
            self._add_tactile_series(series)
            self._series = series
        return self._series

    def _add_robot_series(self, series: dict[str, TimelineSeries]) -> None:
        try:
            robot_state = self.episode.robot_state()
        except FileNotFoundError:
            return
        part_items = [(str(part), stream) for part, stream in robot_state.items() if isinstance(stream, dict)]
        for part, stream in part_items:
            time_ns = ms_list_to_ns(stream.get("timestamps", []))
            for field in ("joint", "vel", "effort", "eef"):
                values = list(stream.get(field, []))
                if not values:
                    continue
                series[f"robot/{part}/{field}"] = TimelineSeries(
                    key=f"robot/{part}/{field}",
                    time_ns=time_ns,
                    values=rows_to_array(values),
                    metadata={"source": self.episode.layout.robot_state_file, "part": part, "field": field},
                )
        self._add_combined_robot_series(series, part_items)

    def _add_combined_robot_series(self, series: dict[str, TimelineSeries], part_items: list[tuple[str, dict[str, Any]]]) -> None:
        for field, key in (("joint", "robot/qpos"), ("vel", "robot/qvel"), ("effort", "robot/effort")):
            time_ns: np.ndarray | None = None
            arrays = []
            parts = []
            for part, stream in part_items:
                values = list(stream.get(field, []))
                if not values or any(value is None for value in values):
                    arrays = []
                    break
                part_time_ns = ms_list_to_ns(stream.get("timestamps", []))
                if time_ns is None:
                    time_ns = part_time_ns
                elif not np.array_equal(time_ns, part_time_ns):
                    raise ValueError(f"cannot build {key}: robot part timestamps are not aligned")
                arrays.append(rows_to_array(values))
                parts.append(part)
            if arrays and time_ns is not None:
                series[key] = TimelineSeries(
                    key=key,
                    time_ns=time_ns,
                    values=np.concatenate(arrays, axis=1),
                    metadata={"source": self.episode.layout.robot_state_file, "field": field, "parts": parts},
                )

    def _add_action_series(self, series: dict[str, TimelineSeries]) -> None:
        traces = self.episode.action_traces()
        if not traces:
            return
        time_ns = np.asarray([int(trace["scheduler_stamp_ns"]) for trace in traces], dtype=np.int64)
        series["action/sent"] = TimelineSeries(
            key="action/sent",
            time_ns=time_ns,
            values=np.stack([np.asarray(trace["action"], dtype=np.float32).reshape(-1) for trace in traces], axis=0),
            metadata={"source": self.episode.layout.action_traces_file, "action_space": str(traces[0].get("action_space", ""))},
        )
        for key, field in (
            ("action/sent_action_count", "sent_action_count"),
            ("action/queue_len_after_send", "queue_len_after_send"),
        ):
            series[key] = TimelineSeries(
                key=key,
                time_ns=time_ns,
                values=np.asarray([int(trace[field]) for trace in traces], dtype=np.int64),
                metadata={"source": self.episode.layout.action_traces_file},
            )
        series["action/send_latency_ms"] = TimelineSeries(
            key="action/send_latency_ms",
            time_ns=time_ns,
            values=np.asarray(
                [
                    max(0.0, (int(trace["send_done_ns"]) - int(trace["send_start_ns"])) / 1_000_000.0)
                    for trace in traces
                ],
                dtype=np.float32,
            ),
            metadata={"source": self.episode.layout.action_traces_file},
        )

    def _add_camera_series(self, series: dict[str, TimelineSeries]) -> None:
        metadata = self.episode.image_metadata()
        for name, timestamps in self.episode.image_timestamps().items():
            item = dict(metadata.get(name, {}))
            key = f"sensor/image/{name}"
            series[key] = TimelineSeries(
                key=key,
                time_ns=ms_list_to_ns(timestamps),
                values=np.arange(len(timestamps), dtype=np.int64),
                metadata={"source": self.episode.layout.image_timestamps_file, "stream": name, "media": item.get("video", ""), **item},
            )

    def _add_tactile_series(self, series: dict[str, TimelineSeries]) -> None:
        timestamps = self.episode.tactile_timestamps()
        tactile_metadata = self.episode.tactile_metadata()
        for name, stream in self.episode.tactile_pointcloud().items():
            item = dict(tactile_metadata.get(name, {}))
            key = f"sensor/tactile/{name}/pointcloud"
            series[key] = TimelineSeries(
                key=key,
                time_ns=ms_list_to_ns(timestamps.get(name, [])),
                values=list(stream.get("data", [])),
                metadata={"source": self.episode.layout.tactile_pointcloud_file, "sensor": name, **item},
            )
        for image_name, item in self.episode.tactile_image_metadata().items():
            sensor = timestamp_ref_name(str(item.get("timestamps", "")))
            key = f"sensor/tactile/{sensor}/image/{image_name}"
            series[key] = TimelineSeries(
                key=key,
                time_ns=ms_list_to_ns(timestamps.get(sensor, [])),
                values=np.arange(len(timestamps.get(sensor, [])), dtype=np.int64),
                metadata={"source": self.episode.layout.tactile_image_metadata_file, "sensor": sensor, "media": item.get("video", ""), **dict(item)},
            )


def ms_list_to_ns(values: Any) -> np.ndarray:
    return np.asarray([timestamp_ms_to_ns(value) for value in values], dtype=np.int64)


def rows_to_array(values: list[Any]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float32)
    if any(value is None for value in values):
        return np.asarray(values, dtype=object)
    return np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in values], axis=0)


def value_at(values: Any, index: int) -> Any:
    if isinstance(values, np.ndarray):
        return values[index].copy() if isinstance(values[index], np.ndarray) else values[index]
    return values[index]


def timestamp_ref_name(value: str) -> str:
    if "::" not in value:
        return value
    return value.split("::", 1)[1]
