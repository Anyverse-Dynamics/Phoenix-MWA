from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from prometheus.core.action import ActionTrace
from prometheus.data.episode import atomic_json_dump, atomic_pickle_dump
from prometheus.data.layout import (
    ROBOT_RECORD_ID,
    SENSOR_RECORD_ID,
    RobotPart,
    V2Layout,
    V2_LAYOUT,
    stamp_ns_to_ms,
)


class EpisodeWriter:
    """Write one V2-compatible episode from numpy data facts.

    The writer is deliberately not a runtime session. It only persists data that
    upstream code has already named and timestamped.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        layout: V2Layout = V2_LAYOUT,
        record_robot: bool = True,
        record_sensors: bool = True,
        write_visualizations: bool = False,
        visualization_dpi: int = 220,
    ):
        self.root = Path(root).expanduser()
        self.layout = layout
        self.record_robot = bool(record_robot)
        self.record_sensors = bool(record_sensors)
        self.write_visualizations = bool(write_visualizations)
        self.visualization_dpi = int(visualization_dpi)
        if self.visualization_dpi <= 0:
            raise ValueError("EpisodeWriter visualization_dpi must be positive")
        self.episode_dir: Path | None = None
        self.episode_id = ""
        self.metadata: dict[str, Any] = {}
        self._closed = False

        self.robot_parts: tuple[RobotPart, ...] = ()
        self.robot_required_fields: tuple[str, ...] = ()
        self.robot_state: dict[str, dict[str, list[Any]]] = {}
        self.robot_action = robot_vector_series()
        self.robot_command = robot_vector_series()
        self.robot_state_topic = ""
        self.robot_eef_topics: dict[str, str] = {}
        self.action_traces: list[dict[str, Any]] = []

        self.image_timestamps: dict[str, list[int]] = {}
        self.image_metadata: dict[str, dict[str, Any]] = {}
        self.tactile_timestamps: dict[str, list[int]] = {}
        self.tactile_pointcloud: dict[str, dict[str, list[Any]]] = {}
        self.tactile_metadata: dict[str, dict[str, Any]] = {}
        self.tactile_image_metadata: dict[str, dict[str, Any]] = {}

    def open(self, episode_id: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        if self.episode_dir is not None:
            raise RuntimeError("EpisodeWriter is already open")
        self.episode_id = str(episode_id)
        self.metadata = dict(metadata or {})
        self.episode_dir = self.layout.episode_dir(self.root, self.episode_id)
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        if self.record_robot:
            self._write_manifest(ROBOT_RECORD_ID, "robot", "recording")
        if self.record_sensors:
            self._write_manifest(SENSOR_RECORD_ID, "sensor", "recording")

    def write_robot_state(
        self,
        *,
        stamp_ns: int,
        qpos: Any,
        parts: Sequence[RobotPart | Mapping[str, Any]],
        qvel: Any | None = None,
        effort: Any | None = None,
        eef_pose: Any | None = None,
        state_topic: str = "",
        eef_topics: Mapping[str, str] | None = None,
        required_fields: Sequence[str] | None = None,
    ) -> None:
        self._require_open()
        if not self.record_robot:
            return
        robot_parts = tuple(RobotPart.from_value(part) for part in parts)
        self._configure_robot_parts(robot_parts, required_fields, qvel=qvel, effort=effort, eef_pose=eef_pose)
        qpos_array = _vector_required(qpos, "qpos")
        qvel_array = _vector_optional(qvel, allow_nan=True)
        effort_array = _vector_optional(effort, allow_nan=True)
        eef_array = _vector_optional(eef_pose)
        self.robot_state_topic = str(state_topic or self.robot_state_topic)
        for part, topic in dict(eef_topics or {}).items():
            self.robot_eef_topics[str(part)] = str(topic)

        stamp_ms = stamp_ns_to_ms(stamp_ns)
        for part in self.robot_parts:
            stream = self.robot_state[part.name]
            stream["joint"].append(_slice_or_none(qpos_array, start=part.start, dim=part.dim, field_name="qpos"))
            stream["vel"].append(_slice_or_none(qvel_array, start=part.start, dim=part.dim, field_name="qvel"))
            stream["effort"].append(_slice_or_none(effort_array, start=part.start, dim=part.dim, field_name="effort"))
            stream["eef"].append(_slice_eef_or_none(eef_array, part))
            stream["timestamps"].append(stamp_ms)

    def write_robot_action(
        self,
        *,
        stamp_ns: int,
        names: Sequence[str],
        position: Any,
        velocity: Any | None = None,
        effort: Any | None = None,
        frame_id: str = "",
    ) -> None:
        self._write_robot_vector(
            self.robot_action,
            stamp_ns=stamp_ns,
            names=names,
            position=position,
            velocity=velocity,
            effort=effort,
            frame_id=frame_id,
        )

    def write_robot_command(
        self,
        *,
        stamp_ns: int,
        names: Sequence[str],
        position: Any,
        velocity: Any | None = None,
        effort: Any | None = None,
        frame_id: str = "",
    ) -> None:
        self._write_robot_vector(
            self.robot_command,
            stamp_ns=stamp_ns,
            names=names,
            position=position,
            velocity=velocity,
            effort=effort,
            frame_id=frame_id,
        )

    def write_action_trace(self, trace: ActionTrace | Mapping[str, Any]) -> None:
        self._require_open()
        self.action_traces.append(action_trace_payload(trace))

    def write_camera_frame(
        self,
        *,
        name: str,
        stamp_ns: int,
        topic: str = "",
        encoding: str = "",
        video: str | None = None,
        lossless: bool = False,
        container: str | None = None,
        codec: str | None = None,
        pixel_format: str = "native",
    ) -> None:
        self._require_open()
        if not self.record_sensors:
            return
        stream_name = str(name)
        is_lossless = bool(lossless)
        self.image_timestamps.setdefault(stream_name, []).append(stamp_ns_to_ms(stamp_ns))
        frames = len(self.image_timestamps[stream_name])
        resolved_pixel_format = str(pixel_format)
        if resolved_pixel_format == "native":
            resolved_pixel_format = "gray" if is_lossless else "native"
        self.image_metadata[stream_name] = {
            "topic": str(topic),
            "encoding": str(encoding),
            "frames": frames,
            "video": str(video or self.layout.image_video_name(stream_name, lossless=is_lossless)),
            "container": str(container or ("matroska" if is_lossless else "mp4")),
            "codec": str(codec or ("ffv1" if is_lossless else "mp4v")),
            "pixel_format": resolved_pixel_format,
            "lossless": is_lossless,
        }

    def write_tactile_bundle(
        self,
        *,
        sensor_name: str,
        stamp_ns: int,
        pointcloud: Any | None = None,
        pointcloud_topic: str = "",
        images: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._require_open()
        if not self.record_sensors:
            return
        if pointcloud is None and not images:
            raise ValueError("write_tactile_bundle requires pointcloud or images")
        name = str(sensor_name)
        self.tactile_timestamps.setdefault(name, []).append(stamp_ns_to_ms(stamp_ns))
        if pointcloud is not None:
            self.tactile_pointcloud.setdefault(name, {"data": []})["data"].append(
                np.asarray(pointcloud, dtype=np.float32).copy()
            )
            self.tactile_metadata[name] = {
                "topic": str(pointcloud_topic),
                "samples": len(self.tactile_pointcloud[name]["data"]),
                "storage": self.layout.tactile_pointcloud_file,
                "timestamps": f"{self.layout.tactile_timestamps_file}::{name}",
                "timestamp_policy": "shared_by_complete_tactile_bundle",
            }
        for image_name, image_metadata in dict(images or {}).items():
            self._record_tactile_image_metadata(name, str(image_name), image_metadata)

    def close(self, *, accept: bool = True) -> None:
        if self._closed:
            return
        self._require_open()
        try:
            if accept:
                self._save()
            else:
                # Rejected episodes still retain diagnostic targets already sent.
                # This does not turn the episode into an accepted training sample.
                if self.action_traces:
                    atomic_pickle_dump(self.action_traces, self.layout.action_traces_path(self.episode_dir))
                if self.record_robot:
                    self._write_manifest(ROBOT_RECORD_ID, "robot", "error", "episode rejected")
                if self.record_sensors:
                    self._write_manifest(SENSOR_RECORD_ID, "sensor", "error", "episode rejected")
        finally:
            self._closed = True

    def _save(self) -> None:
        try:
            if self.record_robot:
                self._write_manifest(ROBOT_RECORD_ID, "robot", "saving")
            if self.record_sensors:
                self._write_manifest(SENSOR_RECORD_ID, "sensor", "saving")
            self._raise_if_no_samples()
            self._write_outputs()
            if self.record_robot:
                self._write_manifest(ROBOT_RECORD_ID, "robot", "complete")
            if self.record_sensors:
                self._write_manifest(SENSOR_RECORD_ID, "sensor", "complete")
        except Exception as exc:
            if self.record_robot:
                self._write_manifest(ROBOT_RECORD_ID, "robot", "error", str(exc))
            if self.record_sensors:
                self._write_manifest(SENSOR_RECORD_ID, "sensor", "error", str(exc))
            raise

    def _write_outputs(self) -> None:
        assert self.episode_dir is not None
        if self.record_robot:
            self._sort_robot_state_by_timestamp()
            atomic_pickle_dump(self.robot_state, self.layout.robot_state_path(self.episode_dir))
            atomic_pickle_dump(self.robot_action, self.layout.robot_action_path(self.episode_dir))
            atomic_pickle_dump(self.robot_command, self.layout.robot_command_path(self.episode_dir))
            atomic_json_dump(self._robot_control_metadata_payload(), self.layout.robot_control_metadata_path(self.episode_dir))
        if self.action_traces:
            atomic_pickle_dump(self.action_traces, self.layout.action_traces_path(self.episode_dir))
        if self.record_sensors:
            if self.image_timestamps:
                atomic_pickle_dump(self.image_timestamps, self.layout.image_timestamps_path(self.episode_dir))
                atomic_json_dump(self.image_metadata, self.layout.image_metadata_path(self.episode_dir))
            if self.tactile_timestamps:
                atomic_pickle_dump(self.tactile_timestamps, self.layout.tactile_timestamps_path(self.episode_dir))
            if self.tactile_pointcloud:
                atomic_pickle_dump(self.tactile_pointcloud, self.layout.tactile_pointcloud_path(self.episode_dir))
                atomic_json_dump(self.tactile_metadata, self.layout.tactile_metadata_path(self.episode_dir))
            if self.tactile_image_metadata:
                atomic_json_dump(self.tactile_image_metadata, self.layout.tactile_image_metadata_path(self.episode_dir))
        if self.write_visualizations:
            self._write_visualizations()

    def _write_visualizations(self) -> None:
        assert self.episode_dir is not None
        try:
            from prometheus.recording.episode_plots import write_episode_visualizations

            result = write_episode_visualizations(
                self.episode_dir,
                robot_state=self.robot_state if self.record_robot else {},
                tactile_pointcloud=self.tactile_pointcloud if self.record_sensors else {},
                tactile_timestamps=self.tactile_timestamps if self.record_sensors else {},
                dpi=self.visualization_dpi,
            )
        except Exception as exc:
            # Diagnostics must never invalidate otherwise complete raw episode data.
            result = {
                "status": "error",
                "outputs": [],
                "errors": {"initialization": f"{type(exc).__name__}: {exc}"},
            }
        self.metadata["visualizations"] = result

    def _configure_robot_parts(
        self,
        parts: tuple[RobotPart, ...],
        required_fields: Sequence[str] | None,
        *,
        qvel: Any | None,
        effort: Any | None,
        eef_pose: Any | None,
    ) -> None:
        if not parts:
            raise ValueError("write_robot_state requires at least one RobotPart")
        inferred_required = ["joint"]
        if qvel is not None:
            inferred_required.append("vel")
        if effort is not None:
            inferred_required.append("effort")
        if eef_pose is not None:
            inferred_required.append("eef")
        required = tuple(str(field) for field in (required_fields or inferred_required))
        if self.robot_parts:
            if self.robot_parts != parts:
                raise ValueError("robot parts changed during episode")
            if self.robot_required_fields != required:
                raise ValueError("robot required fields changed during episode")
            return
        self.robot_parts = parts
        self.robot_required_fields = required
        self.robot_state = {part.name: robot_side_series() for part in parts}

    def _write_robot_vector(
        self,
        stream: dict[str, list[Any]],
        *,
        stamp_ns: int,
        names: Sequence[str],
        position: Any,
        velocity: Any | None,
        effort: Any | None,
        frame_id: str,
    ) -> None:
        self._require_open()
        if not self.record_robot:
            return
        stream["name"].append([str(name) for name in names])
        stream["position"].append(_vector_optional(position))
        stream["velocity"].append(_vector_optional(velocity, allow_nan=True))
        stream["effort"].append(_vector_optional(effort, allow_nan=True))
        stream["frame_id"].append(str(frame_id))
        stream["timestamps"].append(stamp_ns_to_ms(stamp_ns))

    def _record_tactile_image_metadata(
        self,
        sensor_name: str,
        image_name: str,
        metadata: Mapping[str, Any],
    ) -> None:
        old = self.tactile_image_metadata.get(image_name, {})
        frames = int(old.get("frames", 0)) + 1
        video = metadata.get("video") or self.layout.tactile_image_video_name(image_name)
        payload = {
            "topic": str(metadata.get("topic", old.get("topic", ""))),
            "encoding": str(metadata.get("encoding", old.get("encoding", ""))),
            "frames": frames,
            "video": str(video),
            "container": str(metadata.get("container", old.get("container", "matroska"))),
            "codec": str(metadata.get("codec", old.get("codec", "ffv1"))),
            "pixel_format": str(metadata.get("pixel_format", old.get("pixel_format", "bgr0"))),
            "lossless": bool(metadata.get("lossless", old.get("lossless", True))),
            "timestamps": f"{self.layout.tactile_timestamps_file}::{sensor_name}",
            "timestamp_policy": "shared_by_complete_tactile_bundle",
        }
        fps = metadata.get("fps", old.get("fps"))
        if fps is not None:
            payload["fps"] = float(fps)
        self.tactile_image_metadata[image_name] = payload

    def _sort_robot_state_by_timestamp(self) -> None:
        if not self.robot_state:
            return
        timestamps = next(iter(self.robot_state.values()))["timestamps"]
        order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        if order == list(range(len(timestamps))):
            return
        for stream in self.robot_state.values():
            for field_name in ("joint", "vel", "effort", "eef", "timestamps"):
                stream[field_name] = [stream[field_name][index] for index in order]

    def _raise_if_no_samples(self) -> None:
        if self.record_robot:
            robot_samples = sum(len(stream["timestamps"]) for stream in self.robot_state.values())
            if robot_samples <= 0:
                raise RuntimeError("episode received zero robot samples")
        if self.record_sensors:
            sensor_samples = sum(len(value) for value in self.image_timestamps.values()) + sum(
                len(value) for value in self.tactile_timestamps.values()
            )
            if sensor_samples <= 0:
                raise RuntimeError("episode received zero sensor samples")

    def _robot_control_metadata_payload(self) -> dict[str, Any]:
        return {
            "action_topic": "",
            "command_topic": "",
            "state_topic": self.robot_state_topic,
            "eef_topics": dict(sorted(self.robot_eef_topics.items())),
            "left_eef_topic": self.robot_eef_topics.get("left", ""),
            "right_eef_topic": self.robot_eef_topics.get("right", ""),
            "robot_parts": [part.payload() for part in self.robot_parts],
            "robot_required_fields": list(self.robot_required_fields),
            "action_samples": len(self.robot_action["timestamps"]),
            "command_samples": len(self.robot_command["timestamps"]),
            "storage": {
                "state": self.layout.robot_state_file,
                "action": self.layout.robot_action_file,
                "command": self.layout.robot_command_file,
            },
            "action_source": "prometheus_v3_episode_writer",
            "command_source": "prometheus_v3_episode_writer",
            "note": "V3 keeps raw state/sensor data V2-compatible; scheduler traces use action_traces.pkl.",
        }

    def _manifest_payload(self, record_id: str, record_mode: str, status: str, detail: str = "") -> dict[str, Any]:
        payload = {
            "record_id": record_id,
            "status": status,
            "record_mode": record_mode,
            "updated_at_unix": time.time(),
            "episode_dir": "" if self.episode_dir is None else str(self.episode_dir),
            "metadata": dict(self.metadata),
            "camera_frames": {name: len(values) for name, values in sorted(self.image_timestamps.items())},
            "tactile_samples": {
                name: len(stream["data"]) for name, stream in sorted(self.tactile_pointcloud.items())
            },
            "tactile_image_frames": {
                name: int(meta.get("frames", 0)) for name, meta in sorted(self.tactile_image_metadata.items())
            },
            "tactile_timestamp_samples": {
                name: len(values) for name, values in sorted(self.tactile_timestamps.items())
            },
            "robot_samples": {
                part: len(stream["timestamps"]) for part, stream in sorted(self.robot_state.items())
            },
            "robot_action_samples": len(self.robot_action["timestamps"]),
            "robot_command_samples": len(self.robot_command["timestamps"]),
            "action_trace_samples": len(self.action_traces),
            "incomplete_robot_samples": self._incomplete_robot_sample_counts(),
        }
        if detail:
            payload["detail"] = detail
        return payload

    def _write_manifest(self, record_id: str, record_mode: str, status: str, detail: str = "") -> None:
        assert self.episode_dir is not None
        atomic_json_dump(
            self._manifest_payload(record_id, record_mode, status, detail),
            self.layout.manifest_path(self.episode_dir, record_id),
        )

    def _incomplete_robot_sample_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for part, stream in self.robot_state.items():
            count = 0
            for index in range(len(stream["timestamps"])):
                if any(_required_value_missing(stream, part=part, field=field, index=index) for field in self.robot_required_fields):
                    count += 1
            result[part] = count
        return result

    def _require_open(self) -> None:
        if self.episode_dir is None:
            raise RuntimeError("EpisodeWriter.open() must be called before writing")
        if self._closed:
            raise RuntimeError("EpisodeWriter is already closed")


def robot_side_series() -> dict[str, list[Any]]:
    return {"joint": [], "vel": [], "effort": [], "eef": [], "timestamps": []}


def robot_vector_series() -> dict[str, list[Any]]:
    return {"name": [], "position": [], "velocity": [], "effort": [], "frame_id": [], "timestamps": []}


def action_trace_payload(trace: ActionTrace | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(trace, Mapping):
        payload = dict(trace)
        payload["action"] = np.asarray(payload["action"], dtype=np.float32).copy()
        return payload
    return {
        "sent_action_count": int(trace.sent_action_count),
        "action_space": str(trace.action_space),
        "action": np.asarray(trace.action, dtype=np.float32).copy(),
        "scheduler_stamp_ns": int(trace.scheduler_stamp_ns),
        "send_start_ns": int(trace.send_start_ns),
        "send_done_ns": int(trace.send_done_ns),
        "queue_len_after_send": int(trace.queue_len_after_send),
        "metadata": dict(trace.metadata),
    }


def _vector_required(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} contains NaN or inf")
    return array


def _vector_optional(value: Any | None, *, allow_nan: bool = False) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    if array.size == 0:
        return None
    invalid = np.isinf(array) if allow_nan else ~np.isfinite(array)
    if np.any(invalid):
        raise ValueError("vector contains NaN or inf")
    return array


def _slice_or_none(value: np.ndarray | None, *, start: int, dim: int, field_name: str) -> np.ndarray | None:
    if value is None:
        return None
    if start + dim > value.size:
        raise ValueError(f"{field_name} length {value.size} is too small for slice [{start}:{start + dim}]")
    return value[start : start + dim].copy()


def _slice_eef_or_none(value: np.ndarray | None, part: RobotPart) -> np.ndarray | None:
    if value is None or part.eef_start is None:
        return None
    return _slice_or_none(value, start=part.eef_start, dim=part.eef_dim, field_name="eef_pose")


def _required_value_missing(stream: Mapping[str, list[Any]], *, part: str, field: str, index: int) -> bool:
    if field == "eef" or field == f"{part}_eef":
        return stream["eef"][index] is None
    if field in {"joint", "vel", "effort"}:
        return stream[field][index] is None
    return False
