from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from prometheus.data.layout import ROBOT_RECORD_ID, SENSOR_RECORD_ID, V2Layout, V2_LAYOUT


@dataclass(frozen=True, slots=True)
class EpisodeView:
    """Read-only view over one episode directory.

    The view is intentionally thin: all file names come from the layout, so
    offline tools and writers cannot drift into different interpretations.
    """

    path: Path
    layout: V2Layout = V2_LAYOUT

    def __init__(self, path: str | Path, *, layout: V2Layout = V2_LAYOUT):
        object.__setattr__(self, "path", Path(path).expanduser())
        object.__setattr__(self, "layout", layout)

    def robot_state(self) -> dict[str, Any]:
        return load_pickle_required(self.layout.robot_state_path(self.path))

    def robot_action(self) -> dict[str, Any]:
        return load_pickle_optional(self.layout.robot_action_path(self.path), default={})

    def robot_command(self) -> dict[str, Any]:
        return load_pickle_optional(self.layout.robot_command_path(self.path), default={})

    def robot_control_metadata(self) -> dict[str, Any]:
        return load_json_optional(self.layout.robot_control_metadata_path(self.path))

    def image_timestamps(self) -> dict[str, list[int]]:
        return load_pickle_optional(self.layout.image_timestamps_path(self.path), default={})

    def image_metadata(self) -> dict[str, Any]:
        return load_json_optional(self.layout.image_metadata_path(self.path))

    def tactile_timestamps(self) -> dict[str, list[int]]:
        return load_pickle_optional(self.layout.tactile_timestamps_path(self.path), default={})

    def tactile_pointcloud(self) -> dict[str, Any]:
        return load_pickle_optional(self.layout.tactile_pointcloud_path(self.path), default={})

    def tactile_metadata(self) -> dict[str, Any]:
        return load_json_optional(self.layout.tactile_metadata_path(self.path))

    def tactile_image_metadata(self) -> dict[str, Any]:
        return load_json_optional(self.layout.tactile_image_metadata_path(self.path))

    def action_traces(self) -> list[dict[str, Any]]:
        return load_pickle_optional(self.layout.action_traces_path(self.path), default=[])

    def manifest(self, record_id: str) -> dict[str, Any]:
        return load_json_optional(self.layout.manifest_path(self.path, record_id))

    def robot_manifest(self) -> dict[str, Any]:
        return self.manifest(ROBOT_RECORD_ID)

    def sensor_manifest(self) -> dict[str, Any]:
        return self.manifest(SENSOR_RECORD_ID)

    def timeline(self) -> "EpisodeTimelineSource":
        from prometheus.data.timeline import EpisodeTimelineSource

        return EpisodeTimelineSource(self)

    def media_path(self, media_name: str) -> Path:
        path = (self.path / str(media_name)).resolve()
        root = self.path.resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"media path is outside episode: {path}")
        return path


def atomic_json_dump(value: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    _atomic_write(target, lambda tmp: _json_dump(value, tmp))


def atomic_pickle_dump(value: Any, path: str | Path) -> None:
    target = Path(path)
    _atomic_write(target, lambda tmp: _pickle_dump(value, tmp))


def load_pickle_required(path: str | Path) -> Any:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def load_pickle_optional(path: str | Path, *, default: Any) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return load_pickle_required(target)


def load_json_optional(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, Mapping):
        raise ValueError(f"{target} must contain a JSON object")
    return dict(value)


def _atomic_write(path: Path, write_tmp: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write_tmp(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _json_dump(value: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(dict(value), file, indent=2, sort_keys=True)


def _pickle_dump(value: Any, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
