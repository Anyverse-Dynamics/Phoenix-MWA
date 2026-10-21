from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class StreamSample:
    """一个带时间戳的数据事实。

    它只表达“某个 key 在某个时间产生了某个 value”，不表达控制命令，
    也不携带业务语义。ROS、SDK、simulator 都应该先转成这个对象。
    `stamp_ns` 是数据本身的时间，`recv_ns` 是本进程收到它的时间。
    """

    key: str
    value: np.ndarray
    stamp_ns: int
    recv_ns: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        object.__setattr__(self, "recv_ns", int(self.stamp_ns if self.recv_ns is None else self.recv_ns))
        object.__setattr__(self, "value", freeze_array(self.value))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @classmethod
    def now(cls, key: str, value: Any, *, metadata: Mapping[str, Any] | None = None) -> "StreamSample":
        now_ns = time.time_ns()
        return cls(key=key, value=value, stamp_ns=now_ns, recv_ns=now_ns, metadata=metadata or {})


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): freeze_value(item) for key, item in dict(value or {}).items()})


def freeze_array(value: Any) -> np.ndarray:
    array = np.asarray(value).copy()
    if array.dtype == object:
        raise TypeError("StreamSample.value must be a numpy array, not an object container")
    array.setflags(write=False)
    return array


def freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return freeze_mapping(value)
    if isinstance(value, list):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    return value
