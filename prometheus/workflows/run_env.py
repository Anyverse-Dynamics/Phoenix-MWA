from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import instantiate
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from prometheus.core.action import ActionTrace
from prometheus.core.metadata import KIND_SENSOR_IMAGE, META_FIELD, META_KIND, META_SENSOR, META_STREAM_NAME
from prometheus.core.sample import StreamSample
from prometheus.hardware import RobotStatePublisher, RosIngest, RosTopicSpec, robot_observation_topic_specs
from prometheus.recording import (
    EefTraceCapture,
    EpisodeCapture,
    camera_streams_from_topics,
    robot_recording_parts,
    tactile_fields_by_sensor,
)
from prometheus.runtime import ActionScheduler, ObsStore, replace_pending
from prometheus.runtime.preview import FfplayImagePreview, FfplayMosaicPreview, RosMosaicPreviewProcess
from prometheus.transport.zmq_env import ZmqEnvServer
from prometheus.workflows.common import (
    StreamReadiness,
    ignore_sigint_during_cleanup,
    instantiate_optional_node,
    required_stream_names,
    robot_pose_topics,
    robot_state_topic,
)


class LockedRobot:
    """Serialize SDK access when observation publishing and action scheduling run together."""

    def __init__(self, robot: Any):
        self.robot = robot
        self._lock = threading.RLock()

    @property
    def robot_id(self) -> str:
        return str(self.robot.robot_id)

    @property
    def action_space(self) -> str:
        return str(self.robot.action_space)

    @property
    def action_dim(self) -> int:
        return int(self.robot.action_dim)

    @property
    def control_hz(self) -> float:
        return float(self.robot.control_hz)

    @property
    def recording_parts(self) -> tuple[Any, ...]:
        return tuple(getattr(self.robot, "recording_parts", ()))

    def pose_topics(self, *, topic_root: str) -> tuple[dict[str, Any], ...]:
        pose_topics = getattr(self.robot, "pose_topics", None)
        if not callable(pose_topics):
            return ()
        with self._lock:
            return tuple(pose_topics(topic_root=topic_root))

    def connect(self) -> None:
        with self._lock:
            self.robot.connect()

    def read_state(self) -> Any:
        with self._lock:
            return self.robot.read_state()

    def start_control(self) -> None:
        with self._lock:
            self.robot.start_control()

    def send_action(self, action: np.ndarray) -> None:
        with self._lock:
            self.robot.send_action(action)

    def stop_control(self) -> None:
        with self._lock:
            self.robot.stop_control()

    def close(self) -> None:
        with self._lock:
            self.robot.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.robot.status())


@dataclass(frozen=True, slots=True)
class TerminalRequest:
    accept: bool
    reason: str


DISCARD_TERMINAL_REASONS = frozenset({"keyboard_abort", "client_keyboard_discard"})


def rollout_outcome_metadata(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    outcome = str(value).strip().lower()
    if outcome not in ("success", "failure"):
        raise ValueError("terminal outcome must be 'success' or 'failure'")
    return {
        "rollout_outcome": outcome,
        "rollout_success": outcome == "success",
    }


class ActiveSessionSlot:
    """Bridge server-side event callbacks to the currently attached client session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: RunEnvSession | None = None

    def attach(self, session: "RunEnvSession") -> None:
        with self._lock:
            self._session = session

    def detach(self, session: "RunEnvSession") -> None:
        with self._lock:
            if self._session is session:
                self._session = None

    def request_terminal(self, *, accept: bool, reason: str) -> None:
        with self._lock:
            session = self._session
        if session is not None:
            session.request_terminal(accept=accept, reason=reason)


def handle_keyboard_terminal_key(key: str, session_slot: ActiveSessionSlot) -> None:
    if key == "s":
        session_slot.request_terminal(accept=True, reason="keyboard_stop_save")
    elif key == "q":
        session_slot.request_terminal(accept=False, reason="keyboard_abort")


class KeyboardTerminalInput:
    """Direct run_env keyboard control: s=save/terminate, q=abort/truncate."""

    def __init__(self, *, session_slot: ActiveSessionSlot, enabled: bool):
        self.session_slot = session_slot
        self.enabled = bool(enabled)
        self._listener: Any | None = None

    def start(self) -> None:
        if not self.enabled or self._listener is not None:
            return
        from pynput import keyboard

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()
        print("[run_env] keyboard: s=save episode, q=abort episode, Ctrl-C=shutdown", flush=True)

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key: Any) -> None:
        handle_keyboard_terminal_key(str(getattr(key, "char", "")), self.session_slot)


@dataclass(frozen=True, slots=True)
class ResetRuntimeOptions:
    recording_enabled: bool = True
    eef_trace_enabled: bool = True
    recording_root: str | None = None
    recording_episode_prefix: str | None = None
    eef_trace_root: str | None = None
    eef_trace_episode_prefix: str | None = None


def reset_runtime_options(options: Any | None) -> ResetRuntimeOptions:
    data = dict(options or {})
    data.pop("expect", None)
    return ResetRuntimeOptions(
        recording_enabled=bool(data.get("recording_enabled", True)),
        eef_trace_enabled=bool(data.get("eef_trace_enabled", True)),
        recording_root=optional_reset_text(data.get("recording_root")),
        recording_episode_prefix=optional_reset_text(data.get("recording_episode_prefix")),
        eef_trace_root=optional_reset_text(data.get("eef_trace_root")),
        eef_trace_episode_prefix=optional_reset_text(data.get("eef_trace_episode_prefix")),
    )


def optional_reset_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_reset_path(path: str) -> Path:
    return Path(to_absolute_path(str(path))).expanduser()


class ActiveRecorderSlot:
    """Current run_env episode recorder seen by the ROS ingest callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recorder: EpisodeCapture | None = None

    def attach(self, recorder: EpisodeCapture) -> None:
        with self._lock:
            if self._recorder is not None:
                raise RuntimeError("run_env recorder is already attached")
            self._recorder = recorder

    def detach(self, recorder: EpisodeCapture) -> None:
        with self._lock:
            if self._recorder is recorder:
                self._recorder = None

    def record_sample(self, sample: StreamSample) -> None:
        with self._lock:
            recorder = self._recorder
        if recorder is not None:
            recorder.record_sample(sample)


class ActiveEefTraceSlot:
    """Current run_env EEF trace seen by the ROS ingest callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trace: EefTraceCapture | None = None

    def attach(self, trace: EefTraceCapture) -> None:
        with self._lock:
            if self._trace is not None:
                raise RuntimeError("run_env eef trace is already attached")
            self._trace = trace

    def detach(self, trace: EefTraceCapture) -> None:
        with self._lock:
            if self._trace is trace:
                self._trace = None

    def record_sample(self, sample: StreamSample) -> None:
        with self._lock:
            trace = self._trace
        if trace is not None:
            trace.record_sample(sample)


class RunEnvSession:
    """One attached policy client session.

    `reset()` starts robot control and the scheduler. `close()` detaches the
    client and returns the robot to teach/safe mode without stopping the server.
    """

    def __init__(
        self,
        *,
        robot: Any,
        obs_store: ObsStore,
        default_obs_keys: Sequence[str],
        action_hz: float,
        max_chunk_horizon: int,
        policy_low_watermark: int,
        reset_timeout_s: float,
        step_timeout_s: float | None,
        action_rewriter: Callable[..., np.ndarray] = replace_pending,
        fixed_chunk_switch: bool = False,
        dry_run: bool = False,
        action_log_interval: int = 0,
        recorder_factory: Callable[[ResetRuntimeOptions], EpisodeCapture | None] | None = None,
        recorder_slot: ActiveRecorderSlot | None = None,
        eef_trace_factory: Callable[[ResetRuntimeOptions], EefTraceCapture | None] | None = None,
        eef_trace_slot: ActiveEefTraceSlot | None = None,
    ):
        self.robot = robot
        self.obs_store = obs_store
        self.default_obs_keys = tuple(str(key) for key in default_obs_keys)
        self.reset_timeout_s = float(reset_timeout_s)
        self.step_timeout_s = None if step_timeout_s is None else float(step_timeout_s)
        self.recorder_factory = recorder_factory
        self.recorder_slot = recorder_slot
        self.recorder: EpisodeCapture | None = None
        self.eef_trace_factory = eef_trace_factory
        self.eef_trace_slot = eef_trace_slot
        self.eef_trace: EefTraceCapture | None = None
        self.scheduler = ActionScheduler(
            action_target=robot,
            action_hz=float(action_hz),
            max_chunk_horizon=int(max_chunk_horizon),
            policy_low_watermark=int(policy_low_watermark),
            rewriter=action_rewriter,
            fixed_chunk_switch=bool(fixed_chunk_switch),
            dry_run=bool(dry_run),
            action_log_interval=int(action_log_interval),
        )
        self._started = False
        self._terminal_lock = threading.Lock()
        self._terminal: TerminalRequest | None = None

    def reset(self, *, obs_keys: Sequence[str], options: Any | None = None) -> dict[str, Any]:
        if self._started:
            raise RuntimeError("run_env session is already reset")
        self._consume_terminal()
        keys = select_obs_keys(obs_keys, default_obs_keys=self.default_obs_keys)
        validate_reset_options(
            options,
            action_space=self.scheduler.action_space,
            action_dim=self.scheduler.action_dim,
            action_hz=self.scheduler.action_hz,
            max_chunk_horizon=self.scheduler.max_chunk_horizon,
            policy_low_watermark=self.scheduler.policy_low_watermark,
            fixed_chunk_switch=self.scheduler.fixed_chunk_switch,
        )
        runtime_options = reset_runtime_options(options)
        recorder = (
            self.recorder_factory(runtime_options)
            if runtime_options.recording_enabled and self.recorder_factory is not None
            else None
        )
        eef_trace = (
            self.eef_trace_factory(runtime_options)
            if runtime_options.eef_trace_enabled and self.eef_trace_factory is not None
            else None
        )
        obs: dict[str, np.ndarray] | None = None
        try:
            trace_hooks: list[Callable[[ActionTrace], None]] = []
            if recorder is not None:
                episode_id = recorder.start()
                if self.recorder_slot is not None:
                    self.recorder_slot.attach(recorder)
                self.recorder = recorder
                trace_hooks.append(recorder.record_action_trace)
                print(f"[run_env] recording {episode_id}", flush=True)
            if eef_trace is not None:
                trace_id = eef_trace.start()
                if self.eef_trace_slot is not None:
                    self.eef_trace_slot.attach(eef_trace)
                self.eef_trace = eef_trace
                trace_hooks.append(eef_trace.record_action_trace)
                print(f"[run_env] eef trace {trace_id}", flush=True)
            self.scheduler.trace_hook = compose_trace_hooks(trace_hooks)
            if recorder is not None or eef_trace is not None:
                # 调试/录制开启时，reset 返回的第一帧 obs 必须来自 attach 之后。
                obs = self.obs_store.next(keys, timeout_s=self.reset_timeout_s)
            else:
                obs = self.obs_store.latest(keys, timeout_s=self.reset_timeout_s)
            self.robot.start_control()
            try:
                self.scheduler.start()
            except BaseException:
                self.robot.stop_control()
                raise
        except BaseException:
            if recorder is not None:
                if self.recorder_slot is not None:
                    self.recorder_slot.detach(recorder)
                recorder.close(accept=False)
                self.recorder = None
            if eef_trace is not None:
                if self.eef_trace_slot is not None:
                    self.eef_trace_slot.detach(eef_trace)
                eef_trace.close(accept=False)
                self.eef_trace = None
            self.scheduler.trace_hook = None
            raise
        self._started = True
        return {
            "obs": obs,
            "info": {"available_obs": self.obs_store.available(),
                     "sent_action_count": self.scheduler.sent_action_count},
        }

    def observe(self, *, obs_keys: Sequence[str]) -> dict[str, Any]:
        """Return the latest policy observation without touching robot control."""
        keys = select_obs_keys(obs_keys, default_obs_keys=self.default_obs_keys)
        obs = self.obs_store.latest(keys, timeout_s=self.reset_timeout_s)
        return {
            "obs": obs,
            "info": {
                "available_obs": self.obs_store.available(),
                "session_active": self._started,
            },
        }

    def request_terminal(self, *, accept: bool, reason: str) -> None:
        with self._terminal_lock:
            if self._terminal is None:
                self._terminal = TerminalRequest(accept=bool(accept), reason=str(reason))
        self.prepare_terminal()

    def prepare_terminal(self) -> None:
        if self._started:
            self.scheduler.clear_pending()

    def step(self, actions: Any, *, obs_keys: Sequence[str]) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("run_env session has not been reset")
        step_start = time.monotonic()
        if self.recorder is not None:
            self.recorder.raise_if_error()
        keys = select_obs_keys(obs_keys, default_obs_keys=self.default_obs_keys)
        terminal = self._consume_terminal()
        if terminal is not None:
            return self._close_with_terminal(keys, terminal, info={})
        scheduler_start = time.monotonic()
        info: dict[str, Any] = dict(self.scheduler.submit_and_wait(actions, timeout_s=self.step_timeout_s))
        info["scheduler_wait_ms"] = (time.monotonic() - scheduler_start) * 1000.0
        terminal = self._consume_terminal()
        if terminal is not None:
            info["server_step_ms"] = (time.monotonic() - step_start) * 1000.0
            return self._close_with_terminal(keys, terminal, info=info)
        obs_start = time.monotonic()
        obs = self.obs_store.next(keys, timeout_s=self.step_timeout_s)
        info["obs_wait_ms"] = (time.monotonic() - obs_start) * 1000.0
        if self.recorder is not None:
            self.recorder.raise_if_error()
        terminal = self._consume_terminal()
        if terminal is not None:
            info["server_step_ms"] = (time.monotonic() - step_start) * 1000.0
            return self._close_with_terminal(keys, terminal, info=info)
        info["server_step_ms"] = (time.monotonic() - step_start) * 1000.0
        return {
            "obs": obs,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": info,
        }

    @property
    def closed(self) -> bool:
        return not self._started and self.recorder is None and self.eef_trace is None

    def _consume_terminal(self) -> TerminalRequest | None:
        with self._terminal_lock:
            terminal = self._terminal
            self._terminal = None
        return terminal

    def _close_with_terminal(
        self,
        keys: Sequence[str],
        terminal: TerminalRequest,
        *,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "obs": self.obs_store.latest(tuple(keys)),
            "reward": 0.0,
            "terminated": bool(terminal.accept),
            "truncated": not bool(terminal.accept),
            "info": dict(info) | {"terminal_reason": terminal.reason},
        }
        self.close(
            accept=terminal.accept,
            discard=(not terminal.accept and terminal.reason in DISCARD_TERMINAL_REASONS),
        )
        return payload

    def close(
        self,
        *,
        accept: bool = True,
        discard: bool = False,
        episode_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if accept and discard:
            raise ValueError("cannot accept and discard the same session")
        if not self._started and self.recorder is None and self.eef_trace is None:
            return
        error: BaseException | None = None
        recorder = self.recorder
        eef_trace = self.eef_trace
        if recorder is not None and self.recorder_slot is not None:
            self.recorder_slot.detach(recorder)
        if eef_trace is not None and self.eef_trace_slot is not None:
            self.eef_trace_slot.detach(eef_trace)
        try:
            if self._started:
                self.scheduler.stop()
        except BaseException as exc:
            accept = False
            error = exc
        try:
            if self._started:
                self.robot.stop_control()
        except BaseException as exc:
            accept = False
            if error is None:
                error = exc
        try:
            if recorder is not None:
                recorder.close(
                    accept=accept and error is None,
                    discard=discard,
                    episode_metadata=episode_metadata,
                )
                status = (
                    "discarded"
                    if discard
                    else ("saved" if accept and error is None else "rejected")
                )
                print(f"[run_env] {status} {recorder.episode_id}", flush=True)
        except BaseException as exc:
            if error is None:
                error = exc
        try:
            if eef_trace is not None:
                eef_trace.close(accept=accept and error is None, discard=discard)
                status = (
                    "discarded"
                    if discard
                    else ("saved" if accept and error is None else "rejected")
                )
                print(f"[run_env] eef trace {status} {eef_trace.episode_id}", flush=True)
        except BaseException as exc:
            if error is None:
                error = exc
        finally:
            self.recorder = None
            self.eef_trace = None
            self.scheduler.trace_hook = None
            self._started = False
        if error is not None:
            raise error


def serve_forever(
    server: ZmqEnvServer,
    session_factory: Callable[[], RunEnvSession],
    *,
    session_slot: ActiveSessionSlot | None = None,
    health_check: Callable[[], None] | None = None,
) -> None:
    session: RunEnvSession | None = None
    try:
        while True:
            request = server.recv()
            request_id = str(request.get("request_id", ""))
            request_type = str(request.get("type", ""))
            try:
                if health_check is not None:
                    health_check()
                if request_type == "reset":
                    if session is not None:
                        raise RuntimeError("client is already attached; call close before reset")
                    new_session = session_factory()
                    try:
                        payload = new_session.reset(
                            obs_keys=tuple(request.get("obs_keys", ())),
                            options=request.get("options"),
                        )
                    except Exception:
                        new_session.close(accept=False)
                        raise
                    session = new_session
                    if session_slot is not None:
                        session_slot.attach(session)
                    server.reply(request_id, **payload)
                elif request_type == "step":
                    if session is None:
                        raise RuntimeError("no attached client; call reset first")
                    payload = session.step(request.get("actions"), obs_keys=tuple(request.get("obs_keys", ())))
                    closed = session.closed
                    server.reply(request_id, **payload)
                    if closed:
                        if session_slot is not None:
                            session_slot.detach(session)
                        session = None
                elif request_type == "observe":
                    # A pre-reset observe is intentionally read-only: constructing a
                    # session does not start robot control or the action scheduler.
                    reader = session if session is not None else session_factory()
                    payload = reader.observe(obs_keys=tuple(request.get("obs_keys", ())))
                    server.reply(request_id, **payload)
                elif request_type == "prepare_terminal":
                    reason = str(request.get("reason", "client_prepare_terminal"))
                    if session is None:
                        server.reply(
                            request_id,
                            info={"active": False, "terminal_reason": reason},
                        )
                    else:
                        session.prepare_terminal()
                        server.reply(
                            request_id,
                            info={"active": True, "terminal_reason": reason},
                        )
                elif request_type == "terminal":
                    accept = bool(request.get("accept", True))
                    reason = str(request.get("reason", "client_terminal"))
                    episode_metadata = rollout_outcome_metadata(request.get("outcome"))
                    if episode_metadata and not accept:
                        raise ValueError("terminal outcome is only valid for an accepted episode")
                    discard = not accept and (
                        bool(request.get("discard", False))
                        or reason in DISCARD_TERMINAL_REASONS
                    )
                    if session is None:
                        info = {"active": False, "terminal_reason": reason}
                        info.update(episode_metadata)
                        server.reply(
                            request_id,
                            terminated=False,
                            truncated=False,
                            info=info,
                        )
                    else:
                        session.request_terminal(accept=accept, reason=reason)
                        if session_slot is not None:
                            session_slot.detach(session)
                        session.close(
                            accept=accept,
                            discard=discard,
                            episode_metadata=episode_metadata,
                        )
                        session = None
                        info = {"active": True, "terminal_reason": reason}
                        info.update(episode_metadata)
                        server.reply(
                            request_id,
                            terminated=accept,
                            truncated=not accept,
                            info=info,
                        )
                elif request_type == "close":
                    if session is not None:
                        if session_slot is not None:
                            session_slot.detach(session)
                        session.close()
                        session = None
                    server.reply(request_id, info={})
                elif request_type == "shutdown":
                    accept = bool(request.get("accept", False))
                    reason = str(request.get("reason", "client_shutdown"))
                    if session is not None:
                        if session_slot is not None:
                            session_slot.detach(session)
                        session.close(accept=accept)
                        session = None
                    server.reply(request_id, info={"shutdown": True, "terminal_reason": reason})
                    return
                else:
                    raise ValueError(f"unknown run_env request type {request_type!r}")
            except Exception as exc:
                server.reject(request_id, str(exc))
    finally:
        if session is not None:
            if session_slot is not None:
                session_slot.detach(session)
            session.close(accept=False)


def compose_trace_hooks(hooks: Sequence[Callable[[ActionTrace], None]]) -> Callable[[ActionTrace], None] | None:
    callbacks = tuple(hooks)
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]

    def combined(trace: ActionTrace) -> None:
        for callback in callbacks:
            callback(trace)

    return combined


EXPOSURE_CLI_OVERRIDES = {
    "auto-exposure": "realsense.exposure_mode=auto",
    "--auto-exposure": "realsense.exposure_mode=auto",
    "fix-exposure": "realsense.exposure_mode=fixed",
    "--fix-exposure": "realsense.exposure_mode=fixed",
    "fixed-exposure": "realsense.exposure_mode=fixed",
    "--fixed-exposure": "realsense.exposure_mode=fixed",
}

PREVIEW_CLI_OVERRIDES = {
    "head-preview": ("preview.enabled=true",),
    "--head-preview": ("preview.enabled=true",),
    "sensor-preview": ("preview.enabled=true", "preview.mode=ros_mosaic"),
    "--sensor-preview": ("preview.enabled=true", "preview.mode=ros_mosaic"),
}


def rewrite_exposure_cli_args(args: Sequence[str]) -> list[str]:
    """Translate friendly exposure selectors into Hydra overrides."""

    selectors = [arg for arg in args if arg in EXPOSURE_CLI_OVERRIDES]
    explicit = [arg for arg in args if arg.startswith("realsense.exposure_mode=")]
    if len(selectors) > 1:
        raise SystemExit("choose exactly one exposure selector: auto-exposure or fix-exposure")
    if selectors and explicit:
        raise SystemExit("do not combine an exposure selector with realsense.exposure_mode=...")
    return [EXPOSURE_CLI_OVERRIDES.get(arg, arg) for arg in args]


def rewrite_preview_cli_args(args: Sequence[str]) -> list[str]:
    """Translate the friendly head-camera preview selector into a Hydra override."""

    selectors = [arg for arg in args if arg in PREVIEW_CLI_OVERRIDES]
    explicit_enabled = [arg for arg in args if arg.startswith("preview.enabled=")]
    explicit_mode = [arg for arg in args if arg.startswith("preview.mode=")]
    if len(selectors) > 1:
        raise SystemExit("specify at most one preview selector")
    if selectors and explicit_enabled:
        raise SystemExit("do not combine head-preview with preview.enabled=...")
    if selectors and selectors[0] in {"sensor-preview", "--sensor-preview"} and explicit_mode:
        raise SystemExit("do not combine sensor-preview with preview.mode=...")
    rewritten: list[str] = []
    for arg in args:
        replacement = PREVIEW_CLI_OVERRIDES.get(arg)
        if replacement is None:
            rewritten.append(arg)
        else:
            rewritten.extend(replacement)
    return rewritten


@hydra.main(version_base=None, config_path="../configs", config_name="run_env/phoenix")
def main(cfg: DictConfig) -> None:
    run(cfg)


def run(cfg: DictConfig) -> None:
    env_cfg = cfg.env
    robot = LockedRobot(instantiate(cfg.robot.sdk, _convert_="all"))
    action_rewriter = instantiate(env_cfg.action_rewriter)
    realsense = instantiate_optional_node(cfg.realsense, ros_domain_id=env_cfg.ros_domain_id)
    xense = instantiate_optional_node(cfg.xense, ros_domain_id=env_cfg.ros_domain_id)
    publishers = [node for node in (realsense, xense) if node is not None]

    state_topic = robot_state_topic(robot, topic_root=str(env_cfg.robot_topic_root))
    pose_topics = robot_pose_topics(robot, topic_root=str(env_cfg.robot_topic_root))
    state_publisher = RobotStatePublisher(
        robot=robot,
        state_topic=state_topic,
        pose_topics=pose_topics,
        ros_domain_id=env_cfg.ros_domain_id,
        hz=float(robot.control_hz),
        qos_depth=int(env_cfg.qos_depth),
    )
    robot_topics = list(robot_observation_topic_specs(state_topic=state_topic, pose_topics=pose_topics))
    topics = list(robot_topics)
    for publisher in publishers:
        topics.extend(publisher.ros_topic_specs())
    tactile_fields = tactile_fields_by_sensor(publishers)
    camera_streams = camera_streams_from_topics(topics, tactile_fields)
    preview_camera_topics, preview_tactile_image_topics = sensor_preview_topic_maps(topics, tactile_fields)
    readiness = StreamReadiness(required_stream_names(topics))

    obs_store = ObsStore(history=int(env_cfg.obs_history))
    recording_cfg = getattr(cfg, "recording", None)
    recording_enabled = bool(recording_cfg is not None and recording_cfg.enabled)
    tactile_media_fps = float(getattr(cfg.xense, "hz", recording_cfg.media_fps)) if recording_enabled else 0.0
    tactile_field_sets = {sensor: set(fields) for sensor, fields in tactile_fields.items()}
    tactile_media_fps_by_stream: dict[str, float] = {}
    if recording_enabled:
        for topic in topics:
            metadata = dict(getattr(topic, "metadata", {}))
            if metadata.get(META_KIND) != KIND_SENSOR_IMAGE:
                continue
            sensor = str(metadata.get(META_SENSOR, ""))
            field = str(metadata.get(META_FIELD, ""))
            if field not in tactile_field_sets.get(sensor, set()):
                continue
            stream_name = str(metadata.get(META_STREAM_NAME, getattr(topic, "key", "")))
            if stream_name:
                tactile_media_fps_by_stream[stream_name] = tactile_media_fps
    recorder_slot = ActiveRecorderSlot() if recording_enabled else None
    eef_trace_cfg = getattr(cfg, "eef_trace", None)
    eef_trace_enabled = bool(eef_trace_cfg is not None and eef_trace_cfg.enabled)
    eef_trace_slot = ActiveEefTraceSlot() if eef_trace_enabled else None
    session_slot = ActiveSessionSlot()
    keyboard_cfg = getattr(env_cfg, "keyboard", None)
    keyboard = KeyboardTerminalInput(
        session_slot=session_slot,
        enabled=bool(False if keyboard_cfg is None else keyboard_cfg.enabled),
    )
    preview_cfg = getattr(cfg, "preview", None)
    preview_mode = str("single" if preview_cfg is None else getattr(preview_cfg, "mode", "single")).lower()
    if preview_mode == "mosaic":
        preview = FfplayMosaicPreview(
            enabled=bool(preview_cfg.enabled),
            stream_rows=preview_cfg.stream_rows,
            fps=float(preview_cfg.fps),
            width=int(preview_cfg.width),
            height=int(preview_cfg.height),
            stale_timeout_s=float(getattr(preview_cfg, "stale_timeout_s", 1.0)),
            ffplay=str(preview_cfg.ffplay),
            window_title=str(preview_cfg.window_title),
        )
    elif preview_mode == "ros_mosaic":
        window_title = str("Prometheus sensor preview" if preview_cfg is None else preview_cfg.window_title)
        if window_title == "Prometheus head camera":
            window_title = "Prometheus sensor preview"
        preview_python = None if preview_cfg is None else optional_text(getattr(preview_cfg, "python_executable", None))
        preview = RosMosaicPreviewProcess(
            enabled=bool(False if preview_cfg is None else preview_cfg.enabled),
            camera_topics=preview_camera_topics,
            tactile_image_topics=preview_tactile_image_topics,
            fps=float(20.0 if preview_cfg is None else getattr(preview_cfg, "fps", 20.0)),
            width=int(1280 if preview_cfg is None else getattr(preview_cfg, "width", 1280)),
            height=int(720 if preview_cfg is None else getattr(preview_cfg, "height", 720)),
            stale_timeout_s=float(1.0 if preview_cfg is None else getattr(preview_cfg, "stale_timeout_s", 1.0)),
            window_title=window_title,
            ros_domain_id=int(env_cfg.ros_domain_id),
            python_executable=preview_python,
        )
    elif preview_mode == "single":
        preview = FfplayImagePreview(
            enabled=bool(False if preview_cfg is None else preview_cfg.enabled),
            stream_key=str("base_0_rgb" if preview_cfg is None else preview_cfg.stream_key),
            fps=float(30.0 if preview_cfg is None else preview_cfg.fps),
            ffplay=str("ffplay" if preview_cfg is None else preview_cfg.ffplay),
            window_title=str("Prometheus head camera" if preview_cfg is None else preview_cfg.window_title),
        )
    else:
        raise ValueError(f"unknown preview mode: {preview_mode!r}")

    def on_sample(sample: StreamSample) -> None:
        readiness.mark(sample)
        if recorder_slot is not None:
            recorder_slot.record_sample(sample)
        if eef_trace_slot is not None:
            eef_trace_slot.record_sample(sample)
        preview.submit(sample)
        obs_store.put(sample)

    ingest = RosIngest(
        topics=topics,
        callback=on_sample,
        ros_domain_id=env_cfg.ros_domain_id,
        qos_depth=int(env_cfg.qos_depth),
    )

    def check_publishers() -> None:
        for publisher in publishers:
            raise_if_error = getattr(publisher, "raise_if_error", None)
            if callable(raise_if_error):
                raise_if_error()

    def make_session() -> RunEnvSession:
        def make_recorder(runtime_options: ResetRuntimeOptions) -> EpisodeCapture | None:
            if not recording_enabled:
                return None
            recording_root = runtime_options.recording_root or str(recording_cfg.root)
            recording_episode_prefix = runtime_options.recording_episode_prefix or str(recording_cfg.episode_prefix)
            return EpisodeCapture(
                root=resolve_reset_path(recording_root),
                episode_prefix=recording_episode_prefix,
                start_index=int(recording_cfg.start_index),
                queue_size=int(recording_cfg.recorder_queue),
                robot_parts=robot_recording_parts(robot),
                camera_streams=camera_streams,
                tactile_fields_by_sensor=tactile_fields,
                tactile_bundle_timeout_ns=int(recording_cfg.tactile_bundle_timeout_ns),
                media_fps=float(recording_cfg.media_fps),
                ffmpeg=str(recording_cfg.ffmpeg),
                video_codec=str(getattr(recording_cfg, "video_codec", "libx264")),
                video_crf=int(getattr(recording_cfg, "video_crf", 14)),
                video_preset=str(getattr(recording_cfg, "video_preset", "veryfast")),
                write_visualizations=bool(getattr(recording_cfg, "write_visualizations", True)),
                visualization_dpi=int(getattr(recording_cfg, "visualization_dpi", 220)),
                media_fps_by_stream=tactile_media_fps_by_stream,
                metadata={
                    "robot_id": robot.robot_id,
                    "action_space": robot.action_space,
                    "action_hz": float(env_cfg.action_hz),
                    "tactile_media_fps": tactile_media_fps,
                },
                workflow="run_env",
            )

        def make_eef_trace(runtime_options: ResetRuntimeOptions) -> EefTraceCapture | None:
            if not eef_trace_enabled:
                return None
            eef_trace_root = runtime_options.eef_trace_root or str(eef_trace_cfg.root)
            eef_trace_episode_prefix = runtime_options.eef_trace_episode_prefix or str(eef_trace_cfg.episode_prefix)
            return EefTraceCapture(
                root=resolve_reset_path(eef_trace_root),
                episode_prefix=eef_trace_episode_prefix,
                start_index=int(eef_trace_cfg.start_index),
                part=str(eef_trace_cfg.part),
                write_csv=bool(eef_trace_cfg.write_csv),
                write_png=bool(getattr(eef_trace_cfg, "write_png", True)),
                png_dpi=int(getattr(eef_trace_cfg, "png_dpi", 180)),
                write_svg=bool(getattr(eef_trace_cfg, "write_svg", False)),
            )

        return RunEnvSession(
            robot=robot,
            obs_store=obs_store,
            default_obs_keys=tuple(env_cfg.default_obs_keys),
            action_hz=float(env_cfg.action_hz),
            max_chunk_horizon=int(env_cfg.max_chunk_horizon),
            policy_low_watermark=int(env_cfg.policy_low_watermark),
            action_rewriter=action_rewriter,
            fixed_chunk_switch=bool(getattr(env_cfg, "fixed_chunk_switch", False)),
            dry_run=bool(getattr(env_cfg, "dry_run", False)),
            action_log_interval=int(getattr(env_cfg, "action_log_interval", 0)),
            reset_timeout_s=float(env_cfg.reset_timeout_s),
            step_timeout_s=none_or_float(env_cfg.step_timeout_s),
            recorder_factory=make_recorder,
            recorder_slot=recorder_slot,
            eef_trace_factory=make_eef_trace,
            eef_trace_slot=eef_trace_slot,
        )

    try:
        robot.connect()
        preview.start()
        ingest.start()
        state_publisher.start()
        for publisher in publishers:
            publisher.start()
        for publisher in publishers:
            wait_ready = getattr(publisher, "wait_ready", None)
            if callable(wait_ready):
                wait_ready(float(env_cfg.wait_ready_timeout_s))
        keyboard.start()
        readiness.wait(timeout_s=float(env_cfg.wait_ready_timeout_s))
        obs_store.latest(tuple(env_cfg.ready_obs_keys), timeout_s=float(env_cfg.reset_timeout_s))
        print(f"[run_env] serving {env_cfg.addr}", flush=True)
        with ZmqEnvServer(addr=str(env_cfg.addr)) as server:
            serve_forever(server, make_session, session_slot=session_slot, health_check=check_publishers)
    except KeyboardInterrupt:
        print("[run_env] interrupted", flush=True)
    finally:
        with ignore_sigint_during_cleanup():
            keyboard.close()
            ingest.close()
            preview.close()
            for publisher in reversed(publishers):
                publisher.close()
            state_publisher.close()
            robot.close()


def select_obs_keys(obs_keys: Sequence[str], *, default_obs_keys: Sequence[str]) -> tuple[str, ...]:
    keys = tuple(str(key) for key in obs_keys) or tuple(str(key) for key in default_obs_keys)
    if not keys:
        raise ValueError("obs_keys is empty and env.default_obs_keys is empty")
    return keys


def optional_text(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def sensor_preview_topic_maps(
    topics: Sequence[RosTopicSpec | Mapping[str, Any]],
    tactile_fields: Mapping[str, Sequence[str]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Split image topics into V2-style camera and tactile preview maps."""

    camera_topics: dict[str, str] = {}
    tactile_image_topics: dict[str, str] = {}
    tactile_fields_by_name = {
        str(sensor): {str(field) for field in fields}
        for sensor, fields in dict(tactile_fields).items()
    }
    for raw_topic in topics:
        topic = RosTopicSpec.from_value(raw_topic)
        metadata = dict(topic.metadata)
        if metadata.get(META_KIND) != KIND_SENSOR_IMAGE:
            continue
        sensor = str(metadata.get(META_SENSOR, ""))
        field = str(metadata.get(META_FIELD, ""))
        stream_name = str(metadata.get(META_STREAM_NAME, topic.key))
        if not stream_name:
            raise ValueError(f"image topic {topic.topic!r} requires stream_name or key")
        if field in tactile_fields_by_name.get(sensor, set()):
            target = tactile_image_topics
        else:
            target = camera_topics
        if stream_name in target:
            raise ValueError(f"duplicate preview stream name {stream_name!r}")
        target[stream_name] = topic.topic
    return camera_topics, tactile_image_topics


def validate_reset_options(
    options: Any | None,
    *,
    action_space: str,
    action_dim: int,
    action_hz: float,
    max_chunk_horizon: int,
    policy_low_watermark: int,
    fixed_chunk_switch: bool = False,
) -> None:
    """Validate client-declared runtime assumptions before robot control starts."""

    data = dict(options or {})
    expect = dict(data.pop("expect", {}))
    data.pop("recording_enabled", None)
    data.pop("eef_trace_enabled", None)
    data.pop("recording_root", None)
    data.pop("recording_episode_prefix", None)
    data.pop("eef_trace_root", None)
    data.pop("eef_trace_episode_prefix", None)
    if data:
        raise ValueError(f"unsupported reset options: {sorted(data)}")
    supported = {"fixed_chunk_switch", "action_space", "action_dim", "action_hz", "policy_low_watermark", "action_chunk_horizon"}
    unknown = set(expect).difference(supported)
    if unknown:
        raise ValueError(f"unsupported reset expect keys: {sorted(unknown)}")

    if "action_space" in expect and str(expect["action_space"]) != str(action_space):
        raise ValueError(f"action_space mismatch: expected {expect['action_space']!r}, server has {action_space!r}")
    if "action_dim" in expect and int(expect["action_dim"]) != int(action_dim):
        raise ValueError(f"action_dim mismatch: expected {int(expect['action_dim'])}, server has {int(action_dim)}")
    if "action_hz" in expect and not float_equal(float(expect["action_hz"]), float(action_hz)):
        raise ValueError(f"action_hz mismatch: expected {float(expect['action_hz'])}, server has {float(action_hz)}")
    if "policy_low_watermark" in expect and int(expect["policy_low_watermark"]) != int(policy_low_watermark):
        raise ValueError(
            "policy_low_watermark mismatch: "
            f"expected {int(expect['policy_low_watermark'])}, server has {int(policy_low_watermark)}"
        )
    if "fixed_chunk_switch" in expect and expect["fixed_chunk_switch"] != bool(fixed_chunk_switch):
        raise ValueError(f"fixed_chunk_switch mismatch: expected {expect['fixed_chunk_switch']}, server has {fixed_chunk_switch}")
    if "action_chunk_horizon" in expect:
        horizon = int(expect["action_chunk_horizon"])
        if not 1 <= horizon <= int(max_chunk_horizon):
            raise ValueError(f"action_chunk_horizon must be in [1,{int(max_chunk_horizon)}], got {horizon}")
        if horizon <= int(policy_low_watermark):
            raise ValueError(
                "action_chunk_horizon must be greater than policy_low_watermark: "
                f"{horizon} <= {int(policy_low_watermark)}"
            )


def float_equal(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-6


def none_or_float(value: Any) -> float | None:
    return None if value is None else float(value)


if __name__ == "__main__":
    sys.argv[1:] = rewrite_preview_cli_args(rewrite_exposure_cli_args(sys.argv[1:]))
    main()
