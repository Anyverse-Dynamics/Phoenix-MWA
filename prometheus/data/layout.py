from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


ROBOT_RECORD_ID = "prometheus_robot_recorder"
SENSOR_RECORD_ID = "prometheus_sensor_recorder"


@runtime_checkable
class EpisodeLayout(Protocol):
    """Episode file naming contract shared by writer, reader, and timeline."""

    robot_state_file: str
    robot_action_file: str
    robot_command_file: str
    robot_control_metadata_file: str
    image_timestamps_file: str
    image_metadata_file: str
    tactile_timestamps_file: str
    tactile_pointcloud_file: str
    tactile_metadata_file: str
    tactile_image_metadata_file: str
    action_traces_file: str

    def episode_dir(self, root: str | Path, episode_id: str) -> Path: ...

    def manifest_path(self, episode_dir: str | Path, record_id: str) -> Path: ...


@dataclass(frozen=True, slots=True)
class RobotPart:
    """Slice rule for one robot part inside flat robot arrays.

    V2 stores robot state by part: left/right each get 7 qpos and one 7D eef pose.
    Keeping the rule explicit makes the writer generic for future robots.
    """

    name: str
    start: int
    dim: int
    eef_start: int | None = None
    eef_dim: int = 7

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "dim", int(self.dim))
        object.__setattr__(self, "eef_start", None if self.eef_start is None else int(self.eef_start))
        object.__setattr__(self, "eef_dim", int(self.eef_dim))
        if self.start < 0 or self.dim <= 0:
            raise ValueError("RobotPart start must be nonnegative and dim must be positive")
        if self.eef_start is not None and (self.eef_start < 0 or self.eef_dim <= 0):
            raise ValueError("RobotPart eef_start must be nonnegative and eef_dim must be positive")

    @classmethod
    def from_value(cls, value: "RobotPart | Mapping[str, Any]") -> "RobotPart":
        if isinstance(value, RobotPart):
            return value
        return cls(**dict(value))

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "dim": self.dim,
            "eef_start": self.eef_start,
            "eef_dim": self.eef_dim,
        }


@dataclass(frozen=True, slots=True)
class V2Layout:
    """Mainline Prometheus V2-compatible episode layout.

    This class owns file names and timestamp units. It does not know robot or
    sensor topology; those are supplied as writer inputs and stored as metadata.
    """

    robot_state_file: str = "robot_state_dict.pkl"
    robot_action_file: str = "robot_action_dict.pkl"
    robot_command_file: str = "robot_command_dict.pkl"
    robot_control_metadata_file: str = "robot_control_metadata.json"
    image_timestamps_file: str = "image_timestamps.pkl"
    image_metadata_file: str = "image_metadata.json"
    tactile_timestamps_file: str = "tactile_timestamps.pkl"
    tactile_pointcloud_file: str = "tactile_pointcloud_dict.pkl"
    tactile_metadata_file: str = "tactile_metadata.json"
    tactile_image_metadata_file: str = "tactile_image_metadata.json"
    action_traces_file: str = "action_traces.pkl"

    def episode_dir(self, root: str | Path, episode_id: str) -> Path:
        return Path(root).expanduser() / str(episode_id)

    def manifest_path(self, episode_dir: str | Path, record_id: str) -> Path:
        return Path(episode_dir) / f"record_manifest_{record_id}.json"

    def robot_state_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.robot_state_file

    def robot_action_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.robot_action_file

    def robot_command_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.robot_command_file

    def robot_control_metadata_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.robot_control_metadata_file

    def image_timestamps_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.image_timestamps_file

    def image_metadata_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.image_metadata_file

    def tactile_timestamps_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.tactile_timestamps_file

    def tactile_pointcloud_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.tactile_pointcloud_file

    def tactile_metadata_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.tactile_metadata_file

    def tactile_image_metadata_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.tactile_image_metadata_file

    def action_traces_path(self, episode_dir: str | Path) -> Path:
        return Path(episode_dir) / self.action_traces_file

    def image_video_name(self, name: str, *, lossless: bool = False) -> str:
        return f"{name}.mkv" if lossless else f"{name}.mp4"

    def tactile_image_video_name(self, name: str) -> str:
        return f"{name}.mkv"


V2_LAYOUT = V2Layout()


def stamp_ns_to_ms(stamp_ns: int) -> int:
    return int(stamp_ns) // 1_000_000


def timestamp_ms_to_ns(timestamp_ms: int | float) -> int:
    return int(timestamp_ms) * 1_000_000
