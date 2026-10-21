from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from prometheus.core.metadata import (
    KIND_ROBOT_EEF,
    KIND_ROBOT_STATE,
    KIND_SENSOR_IMAGE,
    KIND_SENSOR_POINTCLOUD,
    META_FIELD,
    META_KIND,
    META_LOSSLESS,
    META_PART,
    META_SENSOR,
    META_STREAM_NAME,
)
from prometheus.core.sample import StreamSample
from prometheus.data import EpisodeWriter, RobotPart


class ImageWriter(Protocol):
    """Encode one image frame and return EpisodeWriter-compatible media metadata."""

    def write_frame(
        self,
        *,
        name: str,
        image: np.ndarray,
        stamp_ns: int,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class RecordingStats:
    robot_samples: int = 0
    incomplete_robot_snapshots: int = 0
    camera_frames: int = 0
    tactile_bundles: int = 0
    incomplete_tactile_bundles: int = 0
    duplicate_tactile_samples: int = 0

    def payload(self) -> dict[str, int]:
        return {
            "robot_samples": self.robot_samples,
            "incomplete_robot_snapshots": self.incomplete_robot_snapshots,
            "camera_frames": self.camera_frames,
            "tactile_bundles": self.tactile_bundles,
            "incomplete_tactile_bundles": self.incomplete_tactile_bundles,
            "duplicate_tactile_samples": self.duplicate_tactile_samples,
        }


class EpisodeRecorder:
    """Convert timestamped StreamSample facts into one open EpisodeWriter.

    这个类是录制语义层，不拥有 ROS subscriber、硬件生命周期或后台线程。
    入口程序通常把 `record_sample` 包进 critical `AsyncSampleSink`。
    机器人/触觉的具体切片和 bundle 字段来自 profile，而不是 recorder 默认值。
    """

    def __init__(
        self,
        *,
        writer: EpisodeWriter,
        image_writer: ImageWriter | None = None,
        robot_parts: Sequence[RobotPart | Mapping[str, Any]] = (),
        camera_streams: Sequence[str] = (),
        tactile_fields_by_sensor: Mapping[str, Sequence[str]] | None = None,
        tactile_bundle_timeout_ns: int = 500_000_000,
        max_pending_robot_snapshots: int = 256,
    ):
        self.writer = writer
        self.image_writer = image_writer
        self.camera_streams = _unique_names(camera_streams, "camera_streams")
        self._seen_camera_streams: set[str] = set()
        self.stats = RecordingStats()
        self._robot = _RobotSnapshotAssembler(
            writer=writer,
            stats=self.stats,
            parts=tuple(RobotPart.from_value(part) for part in robot_parts),
            max_pending=max_pending_robot_snapshots,
        )
        self._tactile = _TactileBundleAssembler(
            writer=writer,
            image_writer=image_writer,
            stats=self.stats,
            fields_by_sensor={
                str(k): tuple(str(v) for v in values)
                for k, values in dict(tactile_fields_by_sensor or {}).items()
            },
            timeout_ns=int(tactile_bundle_timeout_ns),
        )

    def record_sample(self, sample: StreamSample) -> None:
        kind = sample.metadata.get(META_KIND)
        if kind == KIND_ROBOT_STATE:
            self._robot.add_state(sample)
            return
        if kind == KIND_ROBOT_EEF:
            self._robot.add_eef(sample)
            return
        if kind == KIND_SENSOR_IMAGE:
            if self._tactile.accepts(sample):
                self._tactile.add(sample)
            else:
                self._write_camera_frame(sample)
            return
        if kind == KIND_SENSOR_POINTCLOUD:
            self._tactile.add(sample)

    def flush(self) -> None:
        self._robot.flush()
        self._tactile.flush()

    def close(self, *, accept: bool = True) -> None:
        error: BaseException | None = None
        try:
            if accept:
                self.flush()
                self._raise_if_missing_expected_streams()
        except BaseException as exc:
            error = exc
        try:
            _close_image_writer(self.image_writer)
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None:
            self.writer.metadata["recording_stats"] = self.stats.payload()
            self.writer.close(accept=False)
            raise error
        self.writer.metadata["recording_stats"] = self.stats.payload()
        self.writer.close(accept=accept)

    def _write_camera_frame(self, sample: StreamSample) -> None:
        name = _stream_name(sample)
        media = _write_image(self.image_writer, sample)
        self.writer.write_camera_frame(
            name=name,
            stamp_ns=sample.stamp_ns,
            topic=str(sample.metadata.get("topic", "")),
            encoding=str(sample.metadata.get("encoding", "")),
            video=media.get("video"),
            lossless=bool(sample.metadata.get(META_LOSSLESS, media.get("lossless", False))),
            container=media.get("container"),
            codec=media.get("codec"),
            pixel_format=str(media.get("pixel_format", "native")),
        )
        self._seen_camera_streams.add(name)
        self.stats.camera_frames += 1

    def _raise_if_missing_expected_streams(self) -> None:
        missing_camera = sorted(set(self.camera_streams).difference(self._seen_camera_streams))
        missing_tactile = self._tactile.missing_sensors()
        if missing_camera or missing_tactile:
            parts = []
            if missing_camera:
                parts.append("camera=" + ",".join(missing_camera))
            if missing_tactile:
                parts.append("tactile=" + ",".join(missing_tactile))
            raise RuntimeError("episode missing configured streams: " + "; ".join(parts))


@dataclass(slots=True)
class _RobotPending:
    stamp_ns: int
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    effort: np.ndarray | None = None
    state_topic: str = ""
    eef_by_part: dict[str, np.ndarray] = field(default_factory=dict)
    eef_topics: dict[str, str] = field(default_factory=dict)


class _RobotSnapshotAssembler:
    """Exact-stamp 聚合 robot/qpos/qvel/effort 和各个 eef pose。

    同一个 stamp 的本体状态/eef 全部到齐才写入，保持 V2 record_drag 的帧语义。
    """

    def __init__(
        self,
        *,
        writer: EpisodeWriter,
        stats: RecordingStats,
        parts: tuple[RobotPart, ...],
        max_pending: int,
    ):
        self.writer = writer
        self.stats = stats
        self.parts = parts
        self.required_eef_parts = tuple(part.name for part in parts if part.eef_start is not None)
        self.max_pending = int(max_pending)
        self.pending: dict[int, _RobotPending] = {}

    def add_state(self, sample: StreamSample) -> None:
        if not self.parts:
            raise RuntimeError("EpisodeRecorder requires robot_parts for robot state samples")
        pending = self._pending(sample.stamp_ns)
        field = str(sample.metadata.get(META_FIELD, ""))
        if field == "qpos":
            if pending.qpos is not None:
                raise ValueError(f"duplicate robot qpos sample at stamp_ns={sample.stamp_ns}")
            pending.qpos = _required_vector(sample.value, "robot qpos")
            pending.state_topic = str(sample.metadata.get("topic", ""))
        elif field == "qvel":
            if pending.qvel is not None:
                raise ValueError(f"duplicate robot qvel sample at stamp_ns={sample.stamp_ns}")
            pending.qvel = _required_vector(sample.value, "robot qvel")
        elif field == "effort":
            if pending.effort is not None:
                raise ValueError(f"duplicate robot effort sample at stamp_ns={sample.stamp_ns}")
            pending.effort = _required_vector(sample.value, "robot effort")
        else:
            raise ValueError("robot state sample requires metadata['field'] in {qpos,qvel,effort}")
        self._commit_if_complete(sample.stamp_ns)

    def add_eef(self, sample: StreamSample) -> None:
        if not self.parts:
            raise RuntimeError("EpisodeRecorder requires robot_parts for robot eef samples")
        part = str(sample.metadata.get(META_PART, ""))
        if not part:
            raise ValueError("robot eef sample requires metadata['part']")
        pending = self._pending(sample.stamp_ns)
        if part in pending.eef_by_part:
            raise ValueError(f"duplicate robot eef sample part={part!r} stamp_ns={sample.stamp_ns}")
        pending.eef_by_part[part] = _required_vector(sample.value, f"{part} eef")
        pending.eef_topics[part] = str(sample.metadata.get("topic", ""))
        self._commit_if_complete(sample.stamp_ns)

    def flush(self) -> None:
        self.stats.incomplete_robot_snapshots += len(self.pending)
        self.pending.clear()

    def _pending(self, stamp_ns: int) -> _RobotPending:
        pending = self.pending.setdefault(int(stamp_ns), _RobotPending(stamp_ns=int(stamp_ns)))
        if len(self.pending) > self.max_pending:
            oldest = min(self.pending)
            self.pending.pop(oldest)
            self.stats.incomplete_robot_snapshots += 1
            raise RuntimeError(f"robot snapshot did not complete before pending limit: stamp_ns={oldest}")
        return pending

    def _commit_if_complete(self, stamp_ns: int) -> None:
        pending = self.pending[int(stamp_ns)]
        if pending.qpos is None or pending.qvel is None or pending.effort is None:
            return
        if any(part not in pending.eef_by_part for part in self.required_eef_parts):
            return
        eef_pose = _robot_eef_vector(self.parts, pending.eef_by_part) if self.required_eef_parts else None
        self.writer.write_robot_state(
            stamp_ns=pending.stamp_ns,
            qpos=pending.qpos,
            qvel=pending.qvel,
            effort=pending.effort,
            eef_pose=eef_pose,
            parts=self.parts,
            state_topic=pending.state_topic,
            eef_topics=pending.eef_topics,
        )
        self.stats.robot_samples += 1
        del self.pending[int(stamp_ns)]


@dataclass(slots=True)
class _TactilePending:
    stamp_ns: int
    first_recv_ns: int
    samples: dict[str, StreamSample] = field(default_factory=dict)


class _TactileBundleAssembler:
    """Exact-stamp 聚合一个 tactile sensor 的 image/pointcloud bundle。

    `fields_by_sensor` 是 profile 给出的完整字段集合；缺字段会计为 incomplete。
    """

    def __init__(
        self,
        *,
        writer: EpisodeWriter,
        image_writer: ImageWriter | None,
        stats: RecordingStats,
        fields_by_sensor: Mapping[str, Sequence[str]],
        timeout_ns: int,
    ):
        self.writer = writer
        self.image_writer = image_writer
        self.stats = stats
        self.fields_by_sensor = {
            str(sensor): tuple(str(field) for field in fields)
            for sensor, fields in fields_by_sensor.items()
        }
        self.timeout_ns = int(timeout_ns)
        self.pending: dict[tuple[str, int], _TactilePending] = {}
        self.last_completed_stamp: dict[str, int] = {}
        self.completed_sensors: set[str] = set()

    def accepts(self, sample: StreamSample) -> bool:
        sensor = str(sample.metadata.get(META_SENSOR, ""))
        field = str(sample.metadata.get(META_FIELD, ""))
        return sensor in self.fields_by_sensor and field in self.fields_by_sensor[sensor]

    def add(self, sample: StreamSample) -> None:
        sensor = str(sample.metadata.get(META_SENSOR, ""))
        field = str(sample.metadata.get(META_FIELD, ""))
        if not self.accepts(sample):
            raise ValueError(f"unexpected tactile sample sensor={sensor!r} field={field!r}")
        self._expire_old(sample.recv_ns)
        key = (sensor, int(sample.stamp_ns))
        if self.last_completed_stamp.get(sensor) == int(sample.stamp_ns):
            self.stats.duplicate_tactile_samples += 1
            return
        pending = self.pending.setdefault(key, _TactilePending(stamp_ns=sample.stamp_ns, first_recv_ns=sample.recv_ns))
        if field in pending.samples:
            self.stats.duplicate_tactile_samples += 1
            return
        pending.samples[field] = sample
        if all(name in pending.samples for name in self.fields_by_sensor[sensor]):
            self._commit(sensor, pending)
            self.last_completed_stamp[sensor] = int(sample.stamp_ns)
            self.completed_sensors.add(sensor)
            del self.pending[key]

    def flush(self) -> None:
        self.stats.incomplete_tactile_bundles += len(self.pending)
        self.pending.clear()

    def missing_sensors(self) -> list[str]:
        return sorted(set(self.fields_by_sensor).difference(self.completed_sensors))

    def _expire_old(self, recv_ns: int) -> None:
        if self.timeout_ns < 0:
            return
        expired = [
            key for key, pending in self.pending.items() if int(recv_ns) - int(pending.first_recv_ns) > self.timeout_ns
        ]
        for key in expired:
            del self.pending[key]
        self.stats.incomplete_tactile_bundles += len(expired)

    def _commit(self, sensor: str, pending: _TactilePending) -> None:
        pointcloud: np.ndarray | None = None
        pointcloud_topic = ""
        images: dict[str, Mapping[str, Any]] = {}
        for sample in pending.samples.values():
            kind = sample.metadata.get(META_KIND)
            if kind == KIND_SENSOR_POINTCLOUD:
                pointcloud = np.asarray(sample.value, dtype=np.float32).copy()
                pointcloud_topic = str(sample.metadata.get("topic", ""))
            elif kind == KIND_SENSOR_IMAGE:
                images[_stream_name(sample)] = _image_metadata(sample, _write_image(self.image_writer, sample))
        self.writer.write_tactile_bundle(
            sensor_name=sensor,
            stamp_ns=pending.stamp_ns,
            pointcloud=pointcloud,
            pointcloud_topic=pointcloud_topic,
            images=images,
        )
        self.stats.tactile_bundles += 1


def _write_image(image_writer: ImageWriter | None, sample: StreamSample) -> Mapping[str, Any]:
    if image_writer is None:
        raise RuntimeError("EpisodeRecorder requires image_writer for sensor image samples")
    image = np.asarray(sample.value).copy()
    return dict(
        image_writer.write_frame(
            name=_stream_name(sample),
            image=image,
            stamp_ns=sample.stamp_ns,
            metadata=sample.metadata,
        )
    )


def _close_image_writer(image_writer: ImageWriter | None) -> None:
    close = getattr(image_writer, "close", None)
    if callable(close):
        close()


def _image_metadata(sample: StreamSample, media: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "topic": str(sample.metadata.get("topic", "")),
        "encoding": str(sample.metadata.get("encoding", "")),
        "lossless": bool(sample.metadata.get(META_LOSSLESS, media.get("lossless", True))),
        **dict(media),
    }


def _stream_name(sample: StreamSample) -> str:
    name = str(sample.metadata.get(META_STREAM_NAME, sample.key))
    if not name:
        raise ValueError("sensor sample requires stream_name or key")
    return name


def _unique_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not item for item in result):
        raise ValueError(f"{name} must not contain empty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _required_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf")
    return array


def _robot_eef_vector(parts: Sequence[RobotPart], eef_by_part: Mapping[str, np.ndarray]) -> np.ndarray:
    size = max(part.eef_start + part.eef_dim for part in parts if part.eef_start is not None)
    eef_pose = np.full(size, np.nan, dtype=np.float32)
    for part in parts:
        if part.eef_start is None:
            continue
        eef = _required_vector(eef_by_part[part.name], f"{part.name} eef")
        if eef.size != part.eef_dim:
            raise ValueError(f"{part.name} eef length {eef.size} does not match expected {part.eef_dim}")
        eef_pose[part.eef_start : part.eef_start + part.eef_dim] = eef
    return eef_pose
