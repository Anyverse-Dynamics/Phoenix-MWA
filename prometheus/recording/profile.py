from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from prometheus.core.metadata import KIND_SENSOR_IMAGE, META_FIELD, META_KIND, META_SENSOR, META_STREAM_NAME


def robot_recording_parts(robot: Any) -> tuple[Any, ...]:
    """Robot profile owns the V2-compatible joint/eef slices."""

    parts = tuple(getattr(robot, "recording_parts", ()))
    if not parts:
        raise RuntimeError(f"robot {type(robot).__name__} does not expose recording_parts")
    return parts


def tactile_fields_by_sensor(nodes: Sequence[Any]) -> dict[str, tuple[str, ...]]:
    """Collect complete tactile bundle definitions from enabled sensor publishers."""

    result: dict[str, tuple[str, ...]] = {}
    for node in nodes:
        provider = getattr(node, "recording_tactile_fields_by_sensor", None)
        if not callable(provider):
            continue
        for sensor_name, fields in dict(provider()).items():
            sensor = str(sensor_name)
            if sensor in result:
                raise ValueError(f"duplicate tactile recording profile for sensor {sensor!r}")
            result[sensor] = tuple(str(field) for field in fields)
    return result


def camera_streams_from_topics(
    topics: Sequence[Any],
    tactile_fields: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return expected non-tactile image streams for this workflow run."""

    result: list[str] = []
    for topic in topics:
        metadata = dict(getattr(topic, "metadata", {}))
        if metadata.get(META_KIND) != KIND_SENSOR_IMAGE:
            continue
        sensor = str(metadata.get(META_SENSOR, ""))
        field = str(metadata.get(META_FIELD, ""))
        if field in tactile_fields.get(sensor, ()):
            continue
        name = str(metadata.get(META_STREAM_NAME, getattr(topic, "key", "")))
        if not name:
            raise ValueError(f"image topic {getattr(topic, 'topic', '')!r} requires stream_name or key")
        if name in result:
            raise ValueError(f"duplicate camera stream name {name!r}")
        result.append(name)
    return tuple(result)


def next_episode_id(root: Path, *, prefix: str, start_index: int) -> str:
    index = int(start_index)
    while True:
        episode_id = f"{prefix}_{index:06d}"
        if not (root / episode_id).exists():
            return episode_id
        index += 1
