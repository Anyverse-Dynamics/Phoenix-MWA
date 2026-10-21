from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from prometheus.core.metadata import META_STAMP_SOURCE, STAMP_SOURCE_RECV_TIME, STAMP_SOURCE_ROS_HEADER
from prometheus.core.sample import StreamSample, freeze_mapping


SampleCallback = Callable[[StreamSample], None]
SENSOR_DATA_MSG_TYPES = {
    "sensor_msgs.msg.Image",
    "sensor_msgs.msg.PointCloud2",
    "sensor_msgs.msg.CameraInfo",
}


@dataclass(frozen=True, slots=True)
class RosTopicSpec:
    """一个 ROS topic 到 StreamSample 的转换规则。

    metadata 只放跨 ROS 边界后仍然有用的信息，例如 kind、sensor、field。
    不在这里塞 recorder/UI 专用状态，避免 ROS ingest 变成业务层。
    """

    key: str
    topic: str
    msg_type: str
    converter: str = "auto"
    qos_depth: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "topic", str(self.topic))
        object.__setattr__(self, "msg_type", str(self.msg_type))
        object.__setattr__(self, "converter", str(self.converter))
        object.__setattr__(self, "qos_depth", None if self.qos_depth is None else int(self.qos_depth))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @classmethod
    def from_value(cls, value: "RosTopicSpec | Mapping[str, Any]") -> "RosTopicSpec":
        if isinstance(value, RosTopicSpec):
            return value
        data = dict(value)
        return cls(**data)

    def to_sample(
        self,
        msg: Any,
        *,
        recv_ns: int | None = None,
    ) -> StreamSample:
        return ros_msg_to_sample(msg, spec=self, recv_ns=recv_ns)


class RosIngest:
    """ROS2 subscriber backend that emits StreamSample.

    它只负责 ROS msg -> StreamSample。fanout、缓存、持久化和 UI 都应由
    入口或更上层组件决定。callback 运行在 ROS spin 线程里，必须是快路径。
    """

    def __init__(
        self,
        *,
        topics: Sequence[RosTopicSpec | Mapping[str, Any]],
        callback: SampleCallback,
        node_name: str = "prometheus_ros_ingest",
        ros_domain_id: int | None = None,
        spin_timeout_s: float = 0.01,
        qos_depth: int = 3,
    ):
        self.topics = tuple(RosTopicSpec.from_value(topic) for topic in topics)
        self.callback = callback
        self.node_name = str(node_name)
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        self.spin_timeout_s = float(spin_timeout_s)
        if self.spin_timeout_s < 0:
            raise ValueError("RosIngest spin_timeout_s must be non-negative")
        self.qos_depth = int(qos_depth)
        if self.qos_depth <= 0:
            raise ValueError("RosIngest qos_depth must be positive")
        self._context: Any | None = None
        self._node: Any | None = None
        self._rclpy: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        import rclpy
        from rclpy.node import Node

        class _Node(Node):
            pass

        self._rclpy = rclpy
        self._context = rclpy.Context()
        try:
            rclpy.init(args=None, context=self._context, domain_id=self.ros_domain_id)
            self._node = _Node(self.node_name, context=self._context)
            for spec in self.topics:
                depth = spec.qos_depth if spec.qos_depth is not None else self.qos_depth
                self._node.create_subscription(
                    _resolve_msg_type(spec.msg_type),
                    spec.topic,
                    self._make_callback(spec),
                    int(depth),
                )
            self._stop.clear()
            self._error = None
            self._thread = threading.Thread(target=self._spin, name=self.node_name, daemon=True)
            self._thread.start()
        except Exception:
            self.close()
            raise

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("RosIngest failed") from self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._rclpy is not None and self._context is not None and self._context.ok():
            self._rclpy.shutdown(context=self._context)
        self._context = None
        self._rclpy = None

    def _make_callback(self, spec: RosTopicSpec):
        def callback(msg: Any) -> None:
            try:
                self.callback(spec.to_sample(msg, recv_ns=time.time_ns()))
            except BaseException as exc:
                self._error = exc
                raise

        return callback

    def _spin(self) -> None:
        assert self._rclpy is not None and self._node is not None
        from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

        if any(spec.msg_type in SENSOR_DATA_MSG_TYPES for spec in self.topics):
            # Deserialize independent image/pointcloud streams concurrently.
            # Control and state callbacks retain single-threaded ordering.
            executor = MultiThreadedExecutor(num_threads=max(2, min(8, len(self.topics))), context=self._context)
        else:
            executor = SingleThreadedExecutor(context=self._context)
        executor.add_node(self._node)
        try:
            while not self._stop.is_set() and self._context is not None and self._context.ok():
                executor.spin_once(timeout_sec=self.spin_timeout_s)
        except BaseException as exc:
            self._error = exc
        finally:
            executor.remove_node(self._node)
            executor.shutdown()


def ros_msg_to_sample(
    msg: Any,
    *,
    spec: RosTopicSpec,
    recv_ns: int | None = None,
) -> StreamSample:
    recv = time.time_ns() if recv_ns is None else int(recv_ns)
    header_stamp_ns = _stamp_ns(msg)
    stamp_ns, stamp_source = _select_stamp(header_stamp_ns, recv_ns=recv)
    value = convert_ros_msg(msg, converter=spec.converter)
    metadata = {
        "topic": spec.topic,
        "msg_type": spec.msg_type,
        "converter": resolved_converter(msg, spec.converter),
        "frame_id": _frame_id(msg),
        **_msg_metadata(msg),
        **dict(spec.metadata),
        META_STAMP_SOURCE: stamp_source,
    }
    return StreamSample(
        key=spec.key,
        value=value,
        stamp_ns=stamp_ns,
        recv_ns=recv,
        metadata=metadata,
    )


def convert_ros_msg(msg: Any, *, converter: str = "auto") -> Any:
    converter = resolved_converter(msg, converter)
    if converter == "joint_state":
        return _vector(msg.position)
    if converter == "joint_position":
        return _vector(msg.position)
    if converter == "joint_velocity":
        return _vector(msg.velocity)
    if converter == "joint_effort":
        return _vector(msg.effort)
    if converter == "pose_wxyz":
        pose = msg.pose
        return np.asarray(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ],
            dtype=np.float32,
        )
    if converter == "image":
        return _image_msg_to_numpy(msg)
    if converter == "camera_info":
        d = _vector(msg.d)
        if len(d) != 5:
            raise ValueError(f"CameraInfo.d must have 5 values for numpy stream, got {len(d)}")
        return np.concatenate(
            [
                np.asarray([msg.height, msg.width], dtype=np.float32),
                d,
                np.asarray(msg.k, dtype=np.float32).reshape(9),
                np.asarray(msg.r, dtype=np.float32).reshape(9),
                np.asarray(msg.p, dtype=np.float32).reshape(12),
            ]
        ).astype(np.float32, copy=False)
    if converter == "pointcloud2":
        return _pointcloud2_msg_to_numpy(msg)
    if converter == "int":
        return int(msg.data)
    if converter == "scalar":
        value = msg.data
        return int(value) if isinstance(value, (bool, int, np.integer)) else float(value)
    raise ValueError(f"unknown ROS converter: {converter}")


def resolved_converter(msg: Any, converter: str) -> str:
    converter = str(converter)
    if converter != "auto":
        return converter
    if hasattr(msg, "name") and hasattr(msg, "position"):
        return "joint_state"
    if hasattr(msg, "data") and hasattr(msg, "height") and hasattr(msg, "width"):
        return "image"
    if hasattr(msg, "k") and hasattr(msg, "p"):
        return "camera_info"
    if hasattr(msg, "fields") and hasattr(msg, "point_step") and hasattr(msg, "row_step"):
        return "pointcloud2"
    if hasattr(msg, "pose"):
        return "pose_wxyz"
    if hasattr(msg, "data"):
        return "scalar"
    raise ValueError(f"cannot infer ROS converter for {type(msg).__name__}")


def _resolve_msg_type(type_name: str) -> type:
    module_name, _, class_name = str(type_name).replace("/", ".").rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"invalid ROS message type {type_name!r}")
    return getattr(importlib.import_module(module_name), class_name)


def safe_ros_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def _select_stamp(header_stamp_ns: int, *, recv_ns: int) -> tuple[int, str]:
    if not header_stamp_ns:
        return recv_ns, STAMP_SOURCE_RECV_TIME
    return int(header_stamp_ns), STAMP_SOURCE_ROS_HEADER


def _stamp_ns(msg: Any) -> int:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _frame_id(msg: Any) -> str:
    return str(getattr(getattr(msg, "header", None), "frame_id", ""))


def _msg_metadata(msg: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if hasattr(msg, "encoding"):
        metadata["encoding"] = str(msg.encoding)
    if hasattr(msg, "height"):
        metadata["height"] = int(msg.height)
    if hasattr(msg, "width"):
        metadata["width"] = int(msg.width)
    if hasattr(msg, "fields"):
        metadata["fields"] = tuple(str(field.name) for field in msg.fields)
    if hasattr(msg, "point_step"):
        metadata["point_step"] = int(msg.point_step)
    if hasattr(msg, "row_step"):
        metadata["row_step"] = int(msg.row_step)
    return metadata


def _vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1).copy()


def _image_msg_to_numpy(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    channels = 1
    dtype = np.uint8
    if encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding in {"mono16", "16uc1"}:
        dtype = np.uint16
    elif encoding in {"32fc1"}:
        dtype = np.float32
    elif encoding not in {"mono8", "8uc1"}:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    height = int(msg.height)
    width = int(msg.width)
    step_items = int(msg.step) // np.dtype(dtype).itemsize
    data = np.frombuffer(memoryview(msg.data), dtype=dtype).reshape(height, step_items)
    if channels == 1:
        return data[:, :width].copy()
    return data[:, : width * channels].reshape(height, width, channels).copy()


def _pointcloud2_msg_to_numpy(msg: Any) -> np.ndarray:
    fields = tuple(msg.fields)
    if not fields:
        return np.empty((0, 0), dtype=np.float32)
    for index, field in enumerate(fields):
        expected_offset = index * 4
        if int(field.offset) != expected_offset or int(field.datatype) != 7 or int(field.count) != 1:
            raise ValueError("pointcloud2 converter only supports contiguous FLOAT32 fields")
    point_step = int(msg.point_step)
    field_count = len(fields)
    if point_step != field_count * 4:
        raise ValueError(f"pointcloud2 point_step must be {field_count * 4}, got {point_step}")
    height = int(msg.height)
    width = int(msg.width)
    row_step = int(msg.row_step)
    if row_step < point_step * width:
        raise ValueError("pointcloud2 row_step is smaller than point_step * width")
    raw = np.frombuffer(memoryview(msg.data), dtype=np.uint8)
    rows = raw.reshape(height, row_step)
    packed = rows[:, : point_step * width].reshape(height * width, point_step)
    return packed.reshape(height * width * field_count, 4).view(np.float32).reshape(height * width, field_count).copy()
