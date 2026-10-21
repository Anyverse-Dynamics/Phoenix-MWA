from __future__ import annotations

import os
import queue
import select
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from prometheus.core.metadata import KIND_INPUT, META_KIND, META_SENSOR, META_STREAM_NAME
from prometheus.hardware.ros import RosTopicSpec


PEDAL_START = 11
PEDAL_STOP_SAVE = 22
PEDAL_EXIT = 33


@dataclass(frozen=True, slots=True)
class PedalMapping:
    start_key: str = "a"
    stop_key: str = "b"
    exit_key: str = "c"
    start_cmd: int = PEDAL_START
    stop_cmd: int = PEDAL_STOP_SAVE
    exit_cmd: int = PEDAL_EXIT

    def command_for_key(self, key: str) -> int | None:
        key = str(key).lower()
        if key == self.start_key:
            return self.start_cmd
        if key == self.stop_key:
            return self.stop_cmd
        if key == self.exit_key:
            return self.exit_cmd
        return None


class PedalKeyFilter:
    """Debounce/confirm physical pedal key events before publishing commands."""

    def __init__(
        self,
        mapping: PedalMapping,
        *,
        confirm_presses: int = 2,
        confirm_window_s: float = 1.0,
        debounce_s: float = 0.25,
    ):
        self.mapping = mapping
        self.confirm_presses = max(1, int(confirm_presses))
        self.confirm_window_s = max(0.0, float(confirm_window_s))
        self.debounce_s = max(0.0, float(debounce_s))
        self._locked_key = ""
        self._candidate_key = ""
        self._candidate_count = 0
        self._candidate_started_at = 0.0
        self._last_publish_time: dict[str, float] = {}

    def process(self, key: str, event_type: str, *, now: float | None = None) -> int | None:
        key = str(key).lower()
        if self.mapping.command_for_key(key) is None:
            return None
        if event_type == "up":
            if self._locked_key == key:
                self._locked_key = ""
            return None
        if event_type != "down" or self._locked_key == key:
            return None

        timestamp = time.monotonic() if now is None else float(now)
        previous = self._last_publish_time.get(key)
        if previous is not None and timestamp - previous < self.debounce_s:
            return None

        expired = self._candidate_key and timestamp - self._candidate_started_at > self.confirm_window_s
        if expired or self._candidate_key != key:
            self._candidate_key = key
            self._candidate_count = 1
            self._candidate_started_at = timestamp
        else:
            self._candidate_count += 1
        if self._candidate_count < self.confirm_presses:
            return None

        self._candidate_key = ""
        self._candidate_count = 0
        self._candidate_started_at = 0.0
        self._locked_key = key
        self._last_publish_time[key] = timestamp
        return self.mapping.command_for_key(key)


class PedalPublisher:
    """Physical pedal input -> ROS2 Int32.

    Pedal remains a sensor. Top-level programs decide how to interpret commands.
    """

    def __init__(
        self,
        *,
        topic: str = "/prometheus/input/pedal_cmd",
        backend: str = "evdev",
        device: str = "/dev/input/prometheus-pedal",
        ros_domain_id: int | None = None,
        rate_hz: float = 50.0,
        confirm_presses: int = 2,
        confirm_window_s: float = 1.0,
        debounce_s: float = 0.25,
    ):
        if backend not in {"evdev", "keyboard", "stdin"}:
            raise ValueError(f"unsupported pedal backend: {backend!r}")
        self.topic = str(topic)
        self.backend = str(backend)
        self.device = str(device)
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        self.rate_hz = float(rate_hz)
        self.confirm_presses = int(confirm_presses)
        self.confirm_window_s = float(confirm_window_s)
        self.debounce_s = float(debounce_s)
        self.mapping = PedalMapping()
        self._context: Any | None = None
        self._node: Any | None = None
        self._pub: Any | None = None
        self._rclpy: Any | None = None
        self._Int32: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None

    def ros_topic_specs(self) -> tuple[RosTopicSpec, ...]:
        return (
            RosTopicSpec(
                key="input/pedal_cmd",
                topic=self.topic,
                msg_type="std_msgs.msg.Int32",
                converter="int",
                metadata={
                    META_KIND: KIND_INPUT,
                    META_SENSOR: "pedal",
                    META_STREAM_NAME: "input/pedal_cmd",
                },
            ),
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Int32

        class _Node(Node):
            pass

        self._rclpy = rclpy
        self._Int32 = Int32
        self._context = rclpy.Context()
        rclpy.init(args=None, context=self._context, domain_id=self.ros_domain_id)
        self._node = _Node("prometheus_pedal_publisher", context=self._context)
        self._pub = self._node.create_publisher(Int32, self.topic, 10)
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="PedalPublisher", daemon=True)
        self._thread.start()

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._error is not None:
                raise RuntimeError("pedal publisher failed") from self._error
            if self._thread is not None and self._thread.is_alive():
                return
            time.sleep(0.01)
        raise TimeoutError("timed out waiting for pedal publisher")

    def raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("pedal publisher failed") from self._error

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

    def publish_command(self, command: int) -> None:
        assert self._pub is not None and self._Int32 is not None
        msg = self._Int32()
        msg.data = int(command)
        self._pub.publish(msg)

    def _run(self) -> None:
        try:
            if self.backend == "evdev":
                self._run_evdev()
            elif self.backend == "keyboard":
                self._run_keyboard()
            else:
                self._run_stdin()
        except BaseException as exc:
            self._error = exc

    def _run_evdev(self) -> None:
        fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        key_filter = self._key_filter()
        event_struct = struct.Struct("llHHI")
        key_by_code = {30: "a", 48: "b", 46: "c"}
        pending = b""
        period = 1.0 / self.rate_hz
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([fd], [], [], period)
                if not readable:
                    continue
                try:
                    pending += os.read(fd, event_struct.size * 64)
                except BlockingIOError:
                    pass
                while len(pending) >= event_struct.size:
                    raw = pending[: event_struct.size]
                    pending = pending[event_struct.size :]
                    _sec, _usec, event_type, code, value = event_struct.unpack(raw)
                    if event_type != 1:
                        continue
                    command = key_filter.process(key_by_code.get(code, ""), _evdev_event_type(value))
                    if command is not None:
                        self.publish_command(command)
        finally:
            os.close(fd)

    def _run_keyboard(self) -> None:
        try:
            import keyboard
        except ModuleNotFoundError as exc:
            raise RuntimeError("keyboard backend requires the Python 'keyboard' package") from exc

        key_events: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
        key_filter = self._key_filter()

        def on_event(event: Any) -> None:
            name = str(getattr(event, "name", "")).lower()
            event_type = str(getattr(event, "event_type", "")).lower()
            if self.mapping.command_for_key(name) is not None:
                key_events.put((name, event_type))

        hook = keyboard.hook(on_event, suppress=False)
        try:
            while not self._stop.wait(1.0 / self.rate_hz):
                while True:
                    try:
                        key, event_type = key_events.get_nowait()
                    except queue.Empty:
                        break
                    command = key_filter.process(key, event_type)
                    if command is not None:
                        self.publish_command(command)
        finally:
            keyboard.unhook(hook)

    def _run_stdin(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 1.0 / self.rate_hz)
            if not readable:
                continue
            line = sys.stdin.readline()
            if line == "":
                return
            command = _stdin_command(self.mapping, line.strip().lower())
            if command is not None:
                self.publish_command(command)

    def _key_filter(self) -> PedalKeyFilter:
        return PedalKeyFilter(
            self.mapping,
            confirm_presses=self.confirm_presses,
            confirm_window_s=self.confirm_window_s,
            debounce_s=self.debounce_s,
        )


def _evdev_event_type(keystate: int) -> str:
    if int(keystate) == 0:
        return "up"
    if int(keystate) in {1, 2}:
        return "down"
    return ""


def _stdin_command(mapping: PedalMapping, token: str) -> int | None:
    if token in {"11", "22", "33"}:
        return int(token)
    return mapping.command_for_key(token)
