from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from prometheus.core.metadata import KIND_ROBOT_EEF, KIND_ROBOT_STATE, META_FIELD, META_KIND, META_PART
from prometheus.core.sample import freeze_mapping
from prometheus.hardware.robot import Robot, RobotState
from prometheus.hardware.ros import RosTopicSpec


@dataclass(frozen=True, slots=True)
class PoseTopicSpec:
    """RobotState.eef_pose 中一个 7D pose slice 的 ROS 发布规则。

    pose 约定为 [x, y, z, qw, qx, qy, qz]，start 指向 eef_pose 向量中的起点。
    """

    topic: str
    frame_id: str
    start: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "topic", str(self.topic))
        object.__setattr__(self, "frame_id", str(self.frame_id))
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @classmethod
    def from_value(cls, value: "PoseTopicSpec | Mapping[str, Any]") -> "PoseTopicSpec":
        if isinstance(value, PoseTopicSpec):
            return value
        return cls(**dict(value))


class RobotStatePublisher:
    """Publish Robot.read_state() as ROS2 JointState/PoseStamped.

    这个类不拥有 robot lifecycle；调用方负责在 start 前 connect robot，在 close
    后决定是否 close robot。这样不会变成新的 Hardware 大容器。
    """

    def __init__(
        self,
        *,
        robot: Robot,
        state_topic: str,
        pose_topics: Sequence[PoseTopicSpec | Mapping[str, Any]] = (),
        node_name: str = "prometheus_robot_state_publisher",
        ros_domain_id: int | None = None,
        hz: float | None = None,
        qos_depth: int = 10,
        state_frame_id: str | None = None,
    ):
        if robot is None:
            raise ValueError("RobotStatePublisher requires a robot")
        self.robot = robot
        self.state_topic = str(state_topic)
        self.pose_topics = tuple(PoseTopicSpec.from_value(item) for item in pose_topics)
        self.node_name = str(node_name)
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        self.hz = float(getattr(robot, "control_hz", 0.0) if hz is None else hz)
        if self.hz <= 0:
            raise ValueError("RobotStatePublisher hz must be positive")
        self.qos_depth = int(qos_depth)
        if self.qos_depth <= 0:
            raise ValueError("RobotStatePublisher qos_depth must be positive")
        self.state_frame_id = str(state_frame_id or f"{robot.robot_id}/state")
        self.publish_count = 0
        self.last_read_start_ns = 0
        self.last_read_done_ns = 0
        self._context: Any | None = None
        self._node: Any | None = None
        self._executor: Any | None = None
        self._rclpy: Any | None = None
        self._JointState: Any | None = None
        self._PoseStamped: Any | None = None
        self._Time: Any | None = None
        self._state_pub: Any | None = None
        self._pose_pubs: list[tuple[PoseTopicSpec, Any]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        import rclpy
        from builtin_interfaces.msg import Time
        from geometry_msgs.msg import PoseStamped
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        class _Node(Node):
            pass

        self._rclpy = rclpy
        self._JointState = JointState
        self._PoseStamped = PoseStamped
        self._Time = Time
        self._context = rclpy.Context()
        try:
            rclpy.init(args=None, context=self._context, domain_id=self.ros_domain_id)
            self._node = _Node(self.node_name, context=self._context)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._state_pub = self._node.create_publisher(JointState, self.state_topic, self.qos_depth)
            self._pose_pubs = [
                (spec, self._node.create_publisher(PoseStamped, spec.topic, self.qos_depth))
                for spec in self.pose_topics
            ]
            self._node.create_timer(1.0 / self.hz, self.publish_once)
            self._stop.clear()
            self._error = None
            self._thread = threading.Thread(target=self._spin, name=self.node_name, daemon=True)
            self._thread.start()
        except Exception:
            self.close()
            raise

    def publish_once(self) -> None:
        """Read one robot snapshot and publish all configured ROS messages."""

        assert self._JointState is not None and self._PoseStamped is not None
        assert self._state_pub is not None
        read_start_ns = time.time_ns()
        state = self.robot.read_state()
        read_done_ns = time.time_ns()
        self.last_read_start_ns = read_start_ns
        self.last_read_done_ns = read_done_ns
        stamp = self._stamp_for_state(state, fallback_ns=read_done_ns)
        self._state_pub.publish(
            robot_state_to_joint_state_msg(
                state,
                msg_cls=self._JointState,
                stamp=stamp,
                frame_id=self.state_frame_id,
            )
        )
        for spec, pub in self._pose_pubs:
            pub.publish(
                robot_state_to_pose_msg(
                    state,
                    spec=spec,
                    msg_cls=self._PoseStamped,
                    stamp=stamp,
                )
            )
        self.publish_count += 1

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("RobotStatePublisher failed") from self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._executor is not None and self._node is not None:
            self._executor.remove_node(self._node)
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._rclpy is not None and self._context is not None and self._context.ok():
            self._rclpy.shutdown(context=self._context)
        self._context = None
        self._rclpy = None
        self._state_pub = None
        self._pose_pubs = []

    def _spin(self) -> None:
        assert self._executor is not None and self._context is not None
        try:
            while not self._stop.is_set() and self._context.ok():
                self._executor.spin_once(timeout_sec=0.02)
        except BaseException as exc:
            self._error = exc

    def _stamp_for_state(self, state: RobotState, *, fallback_ns: int) -> Any:
        assert self._Time is not None
        if state.stamp_ns > 0:
            return time_msg_from_ns(state.stamp_ns, msg_cls=self._Time)
        return time_msg_from_ns(fallback_ns, msg_cls=self._Time)


def robot_state_to_joint_state_msg(
    state: RobotState,
    *,
    msg_cls: type,
    stamp: Any,
    frame_id: str,
) -> Any:
    qpos = np.asarray(state.qpos, dtype=np.float32).reshape(-1)
    names = tuple(state.joint_names)
    if names and len(names) != len(qpos):
        raise ValueError(f"joint_names length {len(names)} does not match qpos length {len(qpos)}")
    msg = msg_cls()
    msg.header.stamp = stamp
    msg.header.frame_id = str(frame_id)
    msg.name = list(names)
    msg.position = [float(value) for value in qpos]
    if state.qvel is not None:
        msg.velocity = _ros_float_list(state.qvel, expected=len(qpos), field_name="qvel")
    if state.effort is not None:
        msg.effort = _ros_float_list(state.effort, expected=len(qpos), field_name="effort")
    return msg


def robot_state_to_pose_msg(
    state: RobotState,
    *,
    spec: PoseTopicSpec,
    msg_cls: type,
    stamp: Any,
) -> Any:
    if state.eef_pose is None:
        raise ValueError(f"cannot publish pose {spec.topic!r}: RobotState.eef_pose is None")
    pose = np.asarray(state.eef_pose, dtype=np.float32).reshape(-1)
    start = int(spec.start)
    if start < 0 or start + 7 > len(pose):
        raise ValueError(f"pose slice [{start}:{start + 7}] is outside eef_pose length {len(pose)}")
    return pose_wxyz_to_pose_stamped_msg(
        pose[start : start + 7],
        msg_cls=msg_cls,
        stamp=stamp,
        frame_id=spec.frame_id,
    )


def pose_wxyz_to_pose_stamped_msg(
    pose_wxyz: Any,
    *,
    msg_cls: type,
    stamp: Any,
    frame_id: str,
) -> Any:
    pose = np.asarray(pose_wxyz, dtype=np.float32).reshape(7)
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose contains NaN or inf")
    msg = msg_cls()
    msg.header.stamp = stamp
    msg.header.frame_id = str(frame_id)
    msg.pose.position.x = float(pose[0])
    msg.pose.position.y = float(pose[1])
    msg.pose.position.z = float(pose[2])
    msg.pose.orientation.w = float(pose[3])
    msg.pose.orientation.x = float(pose[4])
    msg.pose.orientation.y = float(pose[5])
    msg.pose.orientation.z = float(pose[6])
    return msg


def robot_observation_topic_specs(
    *,
    state_topic: str,
    pose_topics: Sequence[PoseTopicSpec | Mapping[str, Any]] = (),
) -> tuple[RosTopicSpec, ...]:
    """ROS subscriber specs for RobotStatePublisher outputs.

    这里集中写 robot observation 的 metadata 语义，入口程序不要重复拼。
    """

    specs: list[RosTopicSpec] = [
        RosTopicSpec(
            key="robot/qpos",
            topic=state_topic,
            msg_type="sensor_msgs.msg.JointState",
            converter="joint_position",
            metadata={META_KIND: KIND_ROBOT_STATE, META_FIELD: "qpos"},
        ),
        RosTopicSpec(
            key="robot/qvel",
            topic=state_topic,
            msg_type="sensor_msgs.msg.JointState",
            converter="joint_velocity",
            metadata={META_KIND: KIND_ROBOT_STATE, META_FIELD: "qvel"},
        ),
        RosTopicSpec(
            key="robot/effort",
            topic=state_topic,
            msg_type="sensor_msgs.msg.JointState",
            converter="joint_effort",
            metadata={META_KIND: KIND_ROBOT_STATE, META_FIELD: "effort"},
        ),
    ]
    for pose_spec in tuple(PoseTopicSpec.from_value(item) for item in pose_topics):
        part = str(pose_spec.metadata.get(META_PART, pose_spec.metadata.get("part", "")))
        if not part:
            raise ValueError(f"pose topic {pose_spec.topic!r} requires metadata['part']")
        specs.append(
            RosTopicSpec(
                key=f"robot/{part}_eef_pose",
                topic=pose_spec.topic,
                msg_type="geometry_msgs.msg.PoseStamped",
                converter="pose_wxyz",
                metadata={META_KIND: KIND_ROBOT_EEF, META_PART: part},
            )
        )
    return tuple(specs)


def time_msg_from_ns(stamp_ns: int, *, msg_cls: type) -> Any:
    stamp_ns = int(stamp_ns)
    msg = msg_cls()
    msg.sec = stamp_ns // 1_000_000_000
    msg.nanosec = stamp_ns % 1_000_000_000
    return msg


def _ros_float_list(values: Any, *, expected: int, field_name: str) -> list[float]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(vector) != int(expected):
        raise ValueError(f"{field_name} length {len(vector)} does not match qpos length {expected}")
    return [float(value) for value in vector]
