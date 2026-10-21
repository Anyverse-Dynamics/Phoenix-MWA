from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Mapping

import numpy as np

from prometheus.core.metadata import META_STAMP_SOURCE, STAMP_SOURCE_ROBOT_READ_DONE
from prometheus.data.layout import RobotPart
from prometheus.hardware.robot import RobotState
from prometheus.utils.geometry import rpy_to_quaternion_wxyz


ARX5_ACTION_SPACES = ("abs_qpos", "delta_qpos", "abs_eef", "delta_eef")
QPOS_ACTION_SPACES = ("abs_qpos", "delta_qpos")
EEF_ACTION_SPACES = ("abs_eef", "delta_eef")
SIDE_VECTOR_START = {"left": 0, "right": 7}
SIDE_EEF6_START = {"left": 0, "right": 6}

BACKGROUND_SEND_RECV = True
GRAVITY_COMPENSATION = True
TEACH_KD_SCALE = 0.1
POSITION_KP_SCALE = 1.0
POSITION_KD_SCALE = 1.0
GRIPPER_KP_SCALE = 1.0
GRIPPER_KD_SCALE = 1.0
GRIPPER_COMMAND_MARGIN_M = 0.002
GRIPPER_WIDTH_M = 0.082
DEFAULT_PREVIEW_TIME_S = 0.08
LOG_LEVEL = "INFO"
MODEL = "X5"
ARX5_PARTS = (
    RobotPart(name="left", start=0, dim=7, eef_start=0, eef_dim=7),
    RobotPart(name="right", start=7, dim=7, eef_start=7, eef_dim=7),
)


@dataclass(frozen=True, slots=True)
class Arx5Config:
    can: Mapping[str, str]
    gripper_open_readout: Mapping[str, float]
    control_hz: float = 120.0
    gripper_width_m: float = GRIPPER_WIDTH_M
    action_space: str = "abs_qpos"
    preview_time_s: float = DEFAULT_PREVIEW_TIME_S


def requires_connected(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapper(self: "Arx5", *args: Any, **kwargs: Any) -> Any:
        if not self.is_connected:
            raise RuntimeError("ARX5 controller is not connected")
        return method(self, *args, **kwargs)

    return wrapper


class Arx5:
    """ARX5 bimanual SDK adapter.

    只实现 Robot 协议：connect/read_state/start_control/send_action/stop_control/close。
    不创建 ROS node，不落盘，不理解业务状态。
    """

    sides = ("left", "right")
    arm_dof = 6
    action_dim = 14

    def __init__(
        self,
        *,
        robot_id: str = "arx5",
        can: Mapping[str, str],
        gripper_open_readout: Mapping[str, float],
        control_hz: float = 120.0,
        gripper_width_m: float = GRIPPER_WIDTH_M,
        action_space: str = "abs_qpos",
        sdk_module: Any | None = None,
    ):
        if action_space not in ARX5_ACTION_SPACES:
            raise ValueError(f"ARX5 action_space must be one of {ARX5_ACTION_SPACES}")
        if float(control_hz) <= 0:
            raise ValueError("ARX5 control_hz must be positive")
        self.robot_id = str(robot_id)
        self.config = Arx5Config(
            can=_side_mapping(can, "can"),
            gripper_open_readout=_side_mapping(gripper_open_readout, "gripper_open_readout"),
            control_hz=float(control_hz),
            gripper_width_m=float(gripper_width_m),
            action_space=str(action_space),
        )
        self._sdk_module = sdk_module
        self._arx5: Any | None = None
        self._controllers: dict[str, Any] = {}
        self._robot_configs: dict[str, Any] = {}
        self._controller_configs: dict[str, Any] = {}
        self._side_executor: ThreadPoolExecutor | None = None
        self._closed = True
        self.control_mode = ""

    @property
    def action_space(self) -> str:
        return self.config.action_space

    @property
    def control_hz(self) -> float:
        return self.config.control_hz

    @property
    def recording_parts(self) -> tuple[RobotPart, ...]:
        """ARX5 的 V2 落盘切片事实；不属于通用 data 默认值。"""

        return ARX5_PARTS

    def pose_topics(self, *, topic_root: str = "/prometheus/robots") -> tuple[dict[str, Any], ...]:
        """ARX5 的 ROS observation profile，由 workflow 装配时读取。"""

        return pose_topics_for_arx5(robot_id=self.robot_id, topic_root=topic_root)

    @property
    def joint_names(self) -> tuple[str, ...]:
        names = []
        for side in self.sides:
            names.extend([f"{side}_joint_{idx + 1}" for idx in range(self.arm_dof)])
            names.append(f"{side}_gripper")
        return tuple(names)

    @property
    def is_connected(self) -> bool:
        return len(self._controllers) == len(self.sides) and not self._closed

    @property
    def controller_type(self) -> str:
        return controller_type_for_action_space(self.action_space)

    @property
    def command_kind(self) -> str:
        return command_kind_for_action_space(self.action_space)

    def connect(self) -> None:
        if self.is_connected:
            return
        self._arx5 = self._sdk_module if self._sdk_module is not None else _import_arx5()
        self._side_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{self.robot_id}_arx5")
        self._closed = False
        try:
            for side in self.sides:
                robot_cfg = self._arx5.RobotConfigFactory.get_instance().get_config(MODEL)
                robot_cfg.gripper_open_readout = float(self.config.gripper_open_readout[side])
                robot_cfg.gripper_width = float(self.config.gripper_width_m)

                ctrl_cfg = self._arx5.ControllerConfigFactory.get_instance().get_config(
                    self.controller_type,
                    robot_cfg.joint_dof,
                )
                ctrl_cfg.background_send_recv = BACKGROUND_SEND_RECV
                ctrl_cfg.gravity_compensation = GRAVITY_COMPENSATION
                ctrl_cfg.default_preview_time = float(self.config.preview_time_s)

                can_name = str(self.config.can[side])
                if self.controller_type == "joint_controller":
                    controller = self._arx5.Arx5JointController(robot_cfg, ctrl_cfg, can_name)
                elif self.controller_type == "cartesian_controller":
                    controller = self._arx5.Arx5CartesianController(robot_cfg, ctrl_cfg, can_name)
                else:
                    raise ValueError(f"unsupported ARX5 controller type {self.controller_type!r}")
                controller.set_log_level(getattr(self._arx5.LogLevel, LOG_LEVEL))
                self._controllers[side] = controller
                self._robot_configs[side] = robot_cfg
                self._controller_configs[side] = ctrl_cfg
            self.set_teach_mode()
        except Exception:
            self.close()
            raise

    @requires_connected
    def read_state(self) -> RobotState:
        read_start_ns = time.time_ns()
        side_states = self._read_side_states()
        read_done_ns = time.time_ns()
        qpos = np.concatenate([side_states[side]["qpos"] for side in self.sides]).astype(np.float32)
        qvel = np.concatenate([side_states[side]["qvel"] for side in self.sides]).astype(np.float32)
        effort = np.concatenate([side_states[side]["effort"] for side in self.sides]).astype(np.float32)
        eef_pose = np.concatenate([side_states[side]["eef_wxyz"] for side in self.sides]).astype(np.float32)
        return RobotState(
            joint_names=self.joint_names,
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            eef_pose=eef_pose,
            stamp_ns=read_done_ns,
            metadata={
                "robot_id": self.robot_id,
                "eef_pose_order": "xyz_wxyz",
                META_STAMP_SOURCE: STAMP_SOURCE_ROBOT_READ_DONE,
                "read_start_ns": read_start_ns,
                "read_done_ns": read_done_ns,
                "parts": {
                    "left": {"qpos_start": 0, "qpos_dim": 7, "eef_start": 0, "eef_dim": 7},
                    "right": {"qpos_start": 7, "qpos_dim": 7, "eef_start": 7, "eef_dim": 7},
                },
            },
        )

    @requires_connected
    def start_control(self) -> None:
        self.set_position_mode()

    @requires_connected
    def send_action(self, action: np.ndarray) -> None:
        if self.control_mode != "position":
            raise RuntimeError("ARX5 control is not started; call start_control() before send_action()")
        action_space = self.action_space
        if action_space == "abs_qpos":
            self.send_joint_positions(action)
            return
        if action_space == "delta_qpos":
            current = np.asarray(self.read_state().qpos, dtype=np.float32).reshape(self.action_dim)
            self.send_joint_positions(current + np.asarray(action, dtype=np.float32).reshape(self.action_dim))
            return
        if action_space == "abs_eef":
            self.send_eef_positions(action)
            return
        if action_space == "delta_eef":
            current = self._eef_action_from_side_states(self._read_side_states())
            self.send_eef_positions(current + np.asarray(action, dtype=np.float32).reshape(self.action_dim))
            return
        raise ValueError(f"unknown ARX5 action_space {action_space!r}")
    
    def stop_control(self) -> None:
        if self.control_mode == "position":
            self.set_teach_mode()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.is_connected:
                self.stop_control()
                self.reset_to_home() # 自动回到home
                return
        finally:
            self._controllers.clear()
            self._robot_configs.clear()
            self._controller_configs.clear()
            if self._side_executor is not None:
                self._side_executor.shutdown(wait=True)
            self._side_executor = None
            self._closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "robot_id": self.robot_id,
            "connected": self.is_connected,
            "action_space": self.action_space,
            "command_kind": self.command_kind,
            "controller_type": self.controller_type,
            "control_mode": self.control_mode,
            "control_hz": self.control_hz,
            "can": dict(self.config.can),
        }

    @requires_connected
    def set_teach_mode(self) -> None:
        def set_side_teach_mode(side: str) -> None:
            controller = self._controllers[side]
            controller.set_to_damping()
            gain = controller.get_gain()
            gain.kd()[:] *= TEACH_KD_SCALE
            gain.gripper_kd *= TEACH_KD_SCALE
            controller.set_gain(gain)

        self._run_sides(set_side_teach_mode)
        self.control_mode = "teach"

    @requires_connected
    def set_position_mode(self) -> None:
        def set_side_position_mode(side: str) -> None:
            controller = self._controllers[side]
            state = controller.get_joint_state()
            if self.command_kind == "qpos":
                cmd = self._arx5.JointState(self.arm_dof)
                cmd.pos()[:] = np.asarray(state.pos(), dtype=np.float64)
                cmd.gripper_pos = self._clip_gripper_command(float(state.gripper_pos))
                controller.set_joint_cmd(cmd)
            elif self.command_kind == "eef":
                pose_6d = self._eef_pose_6d_from_joint_pos(controller, np.asarray(state.pos(), dtype=np.float64))
                check_finite("current eef pose", pose_6d)
                cmd = self._arx5.EEFState(
                    np.asarray(pose_6d, dtype=np.float64),
                    self._clip_gripper_command(float(state.gripper_pos)),
                )
                controller.set_eef_cmd(cmd)

            ctrl_cfg = self._controller_configs[side]
            gain = self._arx5.Gain(self._robot_configs[side].joint_dof)
            gain.kp()[:] = np.asarray(ctrl_cfg.default_kp, dtype=np.float64) * POSITION_KP_SCALE
            gain.kd()[:] = np.asarray(ctrl_cfg.default_kd, dtype=np.float64) * POSITION_KD_SCALE
            gain.gripper_kp = float(ctrl_cfg.default_gripper_kp) * GRIPPER_KP_SCALE
            gain.gripper_kd = float(ctrl_cfg.default_gripper_kd) * GRIPPER_KD_SCALE
            controller.set_gain(gain)

        self._run_sides(set_side_position_mode)
        self.control_mode = "position"

    @requires_connected
    def reset_to_home(self) -> None:
        self._run_sides(lambda side: self._controllers[side].reset_to_home())

    @requires_connected
    def send_joint_positions(self, qpos: np.ndarray) -> np.ndarray:
        if self.command_kind != "qpos":
            raise RuntimeError(f"ARX5 action_space={self.action_space!r} uses {self.command_kind!r}, not qpos")
        qpos = np.asarray(qpos, dtype=np.float32).reshape(self.action_dim)
        check_finite("qpos command", qpos)

        def send_side_joint_position(side: str) -> None:
            target = side_vector(qpos, side)
            cmd = self._arx5.JointState(self.arm_dof)
            cmd.pos()[:] = target[:6]
            cmd.gripper_pos = self._clip_gripper_command(float(target[6]))
            self._controllers[side].set_joint_cmd(cmd)

        self._run_sides(send_side_joint_position)
        return qpos

    @requires_connected
    def send_eef_positions(self, eef_action: np.ndarray) -> np.ndarray:
        if self.command_kind != "eef":
            raise RuntimeError(f"ARX5 action_space={self.action_space!r} uses {self.command_kind!r}, not eef")
        command = np.asarray(eef_action, dtype=np.float32).reshape(self.action_dim).copy()
        check_finite("eef command", command)

        def send_side_eef_position(side: str) -> None:
            target = side_vector(command, side)
            target[6] = self._clip_gripper_command(float(target[6]))
            cmd = self._arx5.EEFState(np.asarray(target[:6], dtype=np.float64), float(target[6]))
            self._controllers[side].set_eef_cmd(cmd)

        self._run_sides(send_side_eef_position)
        return command

    def _read_side_states(self) -> dict[str, dict[str, np.ndarray]]:
        def read_side_state(side: str) -> dict[str, np.ndarray]:
            controller = self._controllers[side]
            state = controller.get_joint_state()
            arm_pos64 = np.asarray(state.pos(), dtype=np.float64).copy()
            arm_pos = arm_pos64.astype(np.float32)
            arm_vel = np.asarray(state.vel(), dtype=np.float32).copy()
            arm_effort = np.asarray(state.torque(), dtype=np.float32).copy()
            gripper_opening = float(np.clip(float(state.gripper_pos), 0.0, self.config.gripper_width_m))
            eef_pose_6d = self._eef_pose_6d_from_joint_pos(controller, arm_pos64)
            if np.all(np.isfinite(eef_pose_6d)):
                eef_wxyz = np.concatenate(
                    [eef_pose_6d[:3], rpy_to_quaternion_wxyz(eef_pose_6d[3], eef_pose_6d[4], eef_pose_6d[5])]
                ).astype(np.float32)
            else:
                eef_wxyz = np.full(7, np.nan, dtype=np.float32)
            return {
                "qpos": np.concatenate([arm_pos, [gripper_opening]]).astype(np.float32),
                "qvel": np.concatenate([arm_vel, [_optional_float(getattr(state, "gripper_vel", None))]]).astype(
                    np.float32
                ),
                "effort": np.concatenate(
                    [arm_effort, [_optional_float(getattr(state, "gripper_torque", None))]]
                ).astype(np.float32),
                "eef_pose_6d": eef_pose_6d,
                "eef_wxyz": eef_wxyz,
            }

        return self._run_sides(read_side_state)

    def _eef_action_from_side_states(self, side_states: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
        values = []
        for side in self.sides:
            values.append(
                np.concatenate(
                    [side_states[side]["eef_pose_6d"], [side_states[side]["qpos"][6]]],
                ).astype(np.float32)
            )
        return np.concatenate(values).astype(np.float32)

    def _eef_pose_6d_from_joint_pos(self, controller: Any, joint_pos: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(
                controller.forward_kinematics(np.asarray(joint_pos, dtype=np.float64).reshape(self.arm_dof)),
                dtype=np.float32,
            ).reshape(6)
        except Exception:
            return np.full(6, np.nan, dtype=np.float32)

    def _clip_gripper_command(self, gripper_pos: float) -> float:
        safe_max = max(0.0, min(self.config.gripper_width_m, self.config.gripper_width_m - GRIPPER_COMMAND_MARGIN_M))
        return float(np.clip(gripper_pos, 0.0, safe_max))

    def _run_sides(self, fn: Callable[[str], Any]) -> dict[str, Any]:
        executor = self._side_executor
        if executor is None:
            return {side: fn(side) for side in self.sides}
        futures = {side: executor.submit(fn, side) for side in self.sides}
        results: dict[str, Any] = {}
        first_error: Exception | None = None
        for side in self.sides:
            try:
                results[side] = futures[side].result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return results


def controller_type_for_action_space(action_space: str) -> str:
    if action_space in QPOS_ACTION_SPACES:
        return "joint_controller"
    if action_space in EEF_ACTION_SPACES:
        return "cartesian_controller"
    raise ValueError(f"unknown action_space {action_space!r}; expected {ARX5_ACTION_SPACES}")


def command_kind_for_action_space(action_space: str) -> str:
    if action_space in QPOS_ACTION_SPACES:
        return "qpos"
    if action_space in EEF_ACTION_SPACES:
        return "eef"
    raise ValueError(f"unknown action_space {action_space!r}; expected {ARX5_ACTION_SPACES}")


def pose_topics_for_arx5(*, robot_id: str, topic_root: str = "/prometheus/robots") -> tuple[dict[str, Any], ...]:
    """把 ARX5 的左右 eef slice 映射成 ROS PoseStamped topic。"""

    root = topic_root.rstrip("/")
    return (
        {
            "topic": f"{root}/{robot_id}/left_eef_pose",
            "frame_id": f"{robot_id}/left_eef",
            "start": 0,
            "metadata": {"part": "left"},
        },
        {
            "topic": f"{root}/{robot_id}/right_eef_pose",
            "frame_id": f"{robot_id}/right_eef",
            "start": 7,
            "metadata": {"part": "right"},
        },
    )


def side_vector(values: np.ndarray, side: str) -> np.ndarray:
    start = SIDE_VECTOR_START[str(side)]
    return values[start : start + 7]


def check_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or inf values")


def _import_arx5() -> Any:
    try:
        import arx5_interface as arx5
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "arx5_interface is required for ARX5 hardware. Install the ARX5 SDK "
            "and verify with `python -c \"import arx5_interface\"`."
        ) from exc
    except ImportError as exc:
        raise ImportError(f"arx5_interface was found but a native dependency failed to load: {exc}") from exc
    return arx5


def _optional_float(value: Any) -> float:
    return np.nan if value is None else float(value)


def _side_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"ARX5 config requires {name}.left and {name}.right")
    missing = {"left", "right"}.difference(value)
    if missing:
        raise ValueError(f"ARX5 config {name} missing sides: {sorted(missing)}")
    return {"left": value["left"], "right": value["right"]}
