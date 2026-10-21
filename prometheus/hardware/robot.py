from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from prometheus.core.sample import freeze_mapping


@dataclass(frozen=True, slots=True)
class RobotState:
    """一次 robot.read_state() 得到的机器人本体快照。

    eef_pose 对外固定为 [x, y, z, qw, qx, qy, qz]，不能暴露 SDK 内部 rpy。
    """

    qpos: np.ndarray
    joint_names: tuple[str, ...] = ()
    qvel: np.ndarray | None = None
    effort: np.ndarray | None = None
    eef_pose: np.ndarray | None = None
    stamp_ns: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "qpos", _readonly_array(self.qpos))
        object.__setattr__(self, "joint_names", tuple(str(name) for name in self.joint_names))
        object.__setattr__(self, "qvel", _optional_readonly_array(self.qvel))
        object.__setattr__(self, "effort", _optional_readonly_array(self.effort))
        object.__setattr__(self, "eef_pose", _optional_readonly_array(self.eef_pose))
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@runtime_checkable
class Robot(Protocol):
    """最小 SDK 直控协议。

    ROS2 发布、数据持久化、UI、policy server 都不应该进入这个接口。
    """

    robot_id: str
    action_space: str
    action_dim: int
    control_hz: float

    def connect(self) -> None: ...

    def read_state(self) -> RobotState: ...

    def start_control(self) -> None: ...

    def send_action(self, action: np.ndarray) -> None: ...

    def stop_control(self) -> None: ...

    def close(self) -> None: ...

    def status(self) -> Mapping[str, Any]: ...


def _readonly_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).copy()
    array.setflags(write=False)
    return array


def _optional_readonly_array(value: Any | None) -> np.ndarray | None:
    return None if value is None else _readonly_array(value)
