from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from prometheus.core.metadata import (
    KIND_SENSOR_IMAGE,
    KIND_SENSOR_INFO,
    META_FIELD,
    META_KIND,
    META_LOSSLESS,
    META_SENSOR,
    META_STREAM_NAME,
)
from prometheus.hardware.ros import RosTopicSpec, safe_ros_name


CAMERA_TOPIC_PREFIX = "/prometheus/sensors/cameras"
REALSENSE_EXPOSURE_US = 20_000.0
EXPOSURE_MODE_AUTO = "auto"
EXPOSURE_MODE_FIXED = "fixed"


class RealsenseLauncher:
    """pyrealsense2-backed ROS2 publisher for RealSense image streams."""

    def __init__(
        self,
        *,
        cameras: Mapping[str, str],
        name: str = "realsense",
        camera_namespace: str = CAMERA_TOPIC_PREFIX,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        color_format: str = "RGB8",
        exposure_mode: str = EXPOSURE_MODE_FIXED,
        exposure_us: float = REALSENSE_EXPOSURE_US,
        enable_infra: bool = True,
        color_companion_infra_cameras: Sequence[str] | None = None,
        ir_width: int | None = None,
        ir_height: int | None = None,
        qos_depth: int = 10,
        ros_domain_id: int | None = None,
    ):
        self.name = str(name)
        self.cameras = normalize_camera_map(cameras)
        if not self.cameras:
            raise ValueError("RealsenseLauncher requires at least one camera")
        self.camera_namespace = str(camera_namespace).rstrip("/")
        self.width = _positive_int(width, "width")
        self.height = _positive_int(height, "height")
        self.fps = _positive_int(fps, "fps")
        self.color_format = normalize_color_format(color_format)
        self.exposure_mode = normalize_exposure_mode(exposure_mode)
        self.exposure_us = _positive_float(exposure_us, "exposure_us")
        self.enable_infra = bool(enable_infra)
        self.color_companion_infra_cameras = normalize_companion_infra_cameras(
            color_companion_infra_cameras,
            cameras=self.cameras,
        )
        self.ir_width = self.width if ir_width is None else _positive_int(ir_width, "ir_width")
        self.ir_height = self.height if ir_height is None else _positive_int(ir_height, "ir_height")
        self.qos_depth = _positive_int(qos_depth, "qos_depth")
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        self._ctx = mp.get_context("spawn")
        self._stop = self._ctx.Event()
        self._ready_queue: Any | None = None
        self._ready_names: set[str] = set()
        self._procs: dict[str, mp.Process] = {}
        self._started = False
        self._closed = False

    def ros_topic_specs(self) -> tuple[RosTopicSpec, ...]:
        specs = []
        for camera_name in self.cameras:
            specs.extend(
                camera_topic_specs(
                    camera_name,
                    camera_namespace=self.camera_namespace,
                    enable_infra=self.enable_infra,
                )
            )
        return tuple(specs)

    def start(self) -> None:
        if self._started:
            return
        validate_realsense_devices(
            self.cameras,
            width=self.width,
            height=self.height,
            ir_width=self.ir_width,
            ir_height=self.ir_height,
            fps=self.fps,
            color_format=self.color_format,
            enable_infra=self.enable_infra,
            color_companion_infra_cameras=self.color_companion_infra_cameras,
        )
        self._stop.clear()
        self._ready_queue = self._ctx.Queue()
        self._ready_names.clear()
        try:
            for camera_name, serial in self.cameras.items():
                proc = self._ctx.Process(
                    target=realsense_camera_worker,
                    name=f"realsense_{camera_name}",
                    args=(camera_name, serial),
                    kwargs={
                        "stop": self._stop,
                        "ready_queue": self._ready_queue,
                        "camera_namespace": self.camera_namespace,
                        "width": self.width,
                        "height": self.height,
                        "ir_width": self.ir_width,
                        "ir_height": self.ir_height,
                        "fps": self.fps,
                        "color_format": self.color_format,
                        "exposure_mode": self.exposure_mode,
                        "exposure_us": self.exposure_us,
                        "enable_infra": self.enable_infra,
                        "enable_color_companion_infra": (
                            camera_name in self.color_companion_infra_cameras
                        ),
                        "qos_depth": self.qos_depth,
                        "ros_domain_id": self.ros_domain_id,
                        "parent_pid": os.getpid(),
                    },
                )
                proc.start()
                self._procs[camera_name] = proc
        except Exception:
            self.stop()
            raise
        self._started = True
        self._closed = False

    def wait_ready(self, timeout_s: float) -> None:
        if not self._started:
            raise RuntimeError("RealSense launcher has not been started")
        deadline = time.monotonic() + float(timeout_s)
        expected = set(self.cameras)
        while time.monotonic() < deadline:
            self._drain_ready_queue()
            self._raise_on_worker_error()
            if self._ready_names == expected:
                return
            time.sleep(0.05)
        raise TimeoutError(
            "timed out waiting for RealSense cameras: "
            f"ready={sorted(self._ready_names)} expected={sorted(expected)} processes={self._process_status()}"
        )

    def raise_if_error(self) -> None:
        self._raise_on_worker_error()

    def stop(self) -> None:
        self._stop.set()
        for proc in self._procs.values():
            proc.join(timeout=2.0)
        for proc in self._procs.values():
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs.values():
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        self._procs.clear()
        self._ready_names.clear()
        if self._ready_queue is not None:
            close = getattr(self._ready_queue, "close", None)
            if callable(close):
                close()
            self._ready_queue = None
        self._started = False

    def close(self) -> None:
        self.stop()
        self._closed = True

    def status(self) -> dict[str, Any]:
        self._drain_ready_queue()
        return {
            "type": type(self).__name__,
            "name": self.name,
            "started": self._started,
            "closed": self._closed,
            "ready": set(self._ready_names) == set(self.cameras),
            "ready_names": sorted(self._ready_names),
            "cameras": dict(self.cameras),
            "camera_namespace": self.camera_namespace,
            "ros_domain_id": self.ros_domain_id,
            "width": self.width,
            "height": self.height,
            "ir_width": self.ir_width,
            "ir_height": self.ir_height,
            "fps": self.fps,
            "color_format": self.color_format,
            "exposure_mode": self.exposure_mode,
            "exposure_us": self.exposure_us,
            "enable_infra": self.enable_infra,
            "color_companion_infra_cameras": list(self.color_companion_infra_cameras),
            "qos_depth": self.qos_depth,
            "processes": self._process_status(),
        }

    def _drain_ready_queue(self) -> None:
        if self._ready_queue is None:
            return
        while True:
            try:
                self._ready_names.add(str(self._ready_queue.get_nowait()))
            except queue.Empty:
                return

    def _raise_on_worker_error(self) -> None:
        failed = {
            name: {"pid": proc.pid, "exitcode": proc.exitcode}
            for name, proc in self._procs.items()
            if proc.exitcode is not None
        }
        if failed:
            raise RuntimeError(f"RealSense worker exited: {failed}")

    def _process_status(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "pid": proc.pid,
                "alive": proc.is_alive(),
                "exitcode": proc.exitcode,
                "name": proc.name,
            }
            for name, proc in self._procs.items()
        }


def realsense_camera_worker(
    camera_name: str,
    serial: str,
    *,
    parent_pid: int,
    stop: Any,
    ready_queue: Any | None,
    camera_namespace: str,
    width: int,
    height: int,
    ir_width: int,
    ir_height: int,
    fps: int,
    color_format: str,
    exposure_mode: str,
    exposure_us: float,
    enable_infra: bool,
    enable_color_companion_infra: bool,
    qos_depth: int,
    ros_domain_id: int | None,
) -> None:
    _register_parent_death_signal(parent_pid)
    try:
        realsense_camera_loop(
            camera_name,
            serial,
            stop=stop,
            ready_queue=ready_queue,
            camera_namespace=camera_namespace,
            width=width,
            height=height,
            ir_width=ir_width,
            ir_height=ir_height,
            fps=fps,
            color_format=color_format,
            exposure_mode=exposure_mode,
            exposure_us=exposure_us,
            enable_infra=enable_infra,
            enable_color_companion_infra=enable_color_companion_infra,
            qos_depth=qos_depth,
            ros_domain_id=ros_domain_id,
        )
    except BaseException:
        traceback.print_exc()
        raise


def _register_parent_death_signal(parent_pid: int) -> None:
    """Release camera devices when the run_env process aborts."""

    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != int(parent_pid):
        os.kill(os.getpid(), signal.SIGTERM)
        raise RuntimeError("run_env exited before RealSense worker initialization")


def realsense_camera_loop(
    camera_name: str,
    serial: str,
    *,
    stop: Any,
    ready_queue: Any | None,
    camera_namespace: str,
    width: int,
    height: int,
    ir_width: int,
    ir_height: int,
    fps: int,
    color_format: str,
    exposure_mode: str,
    exposure_us: float,
    enable_infra: bool,
    enable_color_companion_infra: bool,
    qos_depth: int,
    ros_domain_id: int | None,
) -> None:
    import rclpy
    import pyrealsense2 as rs
    CvBridge = import_cv_bridge()
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    cfg = rs.config()
    configure_realsense_streams(
        cfg,
        rs,
        serial=serial,
        width=width,
        height=height,
        ir_width=ir_width,
        ir_height=ir_height,
        fps=fps,
        color_format=color_format,
        enable_infra=enable_infra,
        enable_color_companion_infra=enable_color_companion_infra,
    )

    context = rclpy.Context()
    rclpy.init(args=None, context=context, domain_id=ros_domain_id)
    node = Node(f"prometheus_realsense_{safe_ros_name(camera_name)}", context=context)
    bridge = CvBridge()
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    pipeline = None
    try:
        pipeline = start_realsense_pipeline(rs.pipeline, cfg)
        profile = pipeline.get_active_profile()
        if enable_color_companion_infra and not enable_infra:
            node.get_logger().info(
                f"RealSense {camera_name} using an unpublished IR1 companion stream for RGB capture"
            )
        exposure_settings = apply_exposure(
            profile,
            rs,
            mode=exposure_mode,
            exposure_us=exposure_us,
        )
        if exposure_settings:
            node.get_logger().info(
                f"RealSense {camera_name} exposure mode={exposure_mode}: {exposure_settings}"
            )
        names = camera_stream_names(camera_name, enable_infra=enable_infra)
        infos = make_camera_info_values(profile, rs, names, enable_infra=enable_infra)
        expected_streams = image_stream_fields(enable_infra=enable_infra)
        publishers = make_stream_publishers(
            node,
            camera_name,
            camera_namespace=camera_namespace,
            enable_infra=enable_infra,
            qos_depth=qos_depth,
        )
        ready_reported = False
        consecutive_timeouts = 0

        while context.ok() and not stop.is_set():
            try:
                frames = pipeline.wait_for_frames(1000)
            except RuntimeError as exc:
                if stop.is_set():
                    break
                consecutive_timeouts += 1
                if consecutive_timeouts >= 5:
                    raise RuntimeError(f"no RealSense frames from {camera_name} for 5s") from exc
                continue
            consecutive_timeouts = 0
            stamp = node.get_clock().now().to_msg()
            published: set[str] = set()

            color = frames.get_color_frame()
            if color:
                publish_image_and_info(
                    publishers["color"],
                    camera_name,
                    "color",
                    np.asanyarray(color.get_data()),
                    infos["color"],
                    bridge=bridge,
                    stamp=stamp,
                    info_cls=CameraInfo,
                    encoding=color_format.lower(),
                )
                published.add("color")

            if enable_infra:
                left_frame = frames.get_infrared_frame(1)
                right_frame = frames.get_infrared_frame(2)
                if left_frame:
                    publish_image_and_info(
                        publishers["infra1"],
                        camera_name,
                        "infra1",
                        np.asanyarray(left_frame.get_data()),
                        infos["infra1"],
                        bridge=bridge,
                        stamp=stamp,
                        info_cls=CameraInfo,
                        encoding="mono8",
                    )
                    published.add("infra1")
                if right_frame:
                    publish_image_and_info(
                        publishers["infra2"],
                        camera_name,
                        "infra2",
                        np.asanyarray(right_frame.get_data()),
                        infos["infra2"],
                        bridge=bridge,
                        stamp=stamp,
                        info_cls=CameraInfo,
                        encoding="mono8",
                    )
                    published.add("infra2")

            if not ready_reported and published == expected_streams:
                if ready_queue is not None:
                    ready_queue.put(camera_name)
                ready_reported = True
            executor.spin_once(timeout_sec=0.0)
    finally:
        if pipeline is not None:
            pipeline.stop()
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)


def make_stream_publishers(
    node: Any,
    camera_name: str,
    *,
    camera_namespace: str,
    enable_infra: bool,
    qos_depth: int,
) -> dict[str, StreamPublishers]:
    from sensor_msgs.msg import CameraInfo, Image

    publishers = {}
    for field in sorted(image_stream_fields(enable_infra=enable_infra)):
        image_topic, info_topic = stream_topics(camera_name, field, camera_namespace=camera_namespace)
        publishers[field] = StreamPublishers(
            image=node.create_publisher(Image, image_topic, int(qos_depth)),
            info=node.create_publisher(CameraInfo, info_topic, int(qos_depth)),
        )
    return publishers


def publish_image_and_info(
    publishers: StreamPublishers,
    camera_name: str,
    field: str,
    image: np.ndarray,
    info: Mapping[str, Any],
    *,
    bridge: Any,
    stamp: Any,
    info_cls: type,
    encoding: str,
) -> None:
    # Use cv_bridge for ROS image conversion to avoid Python-level pixel copies.
    image_msg = bridge.cv2_to_imgmsg(np.ascontiguousarray(image), encoding=encoding)
    image_msg.header.stamp = stamp
    image_msg.header.frame_id = frame_id(camera_name, field)
    publishers.image.publish(image_msg)
    publishers.info.publish(
        make_camera_info_msg(
            info,
            stamp=stamp,
            frame_id=frame_id(camera_name, field),
            msg_cls=info_cls,
        )
    )


def make_camera_info_msg(info: Mapping[str, Any], *, stamp: Any, frame_id: str, msg_cls: type) -> Any:
    msg = msg_cls()
    msg.header.stamp = stamp
    msg.header.frame_id = str(frame_id)
    msg.width = int(info["width"])
    msg.height = int(info["height"])
    msg.distortion_model = str(info["distortion_model"])
    msg.d = [float(value) for value in np.asarray(info["d"], dtype=np.float64).reshape(-1)]
    msg.k = [float(value) for value in np.asarray(info["k"], dtype=np.float64).reshape(9)]
    msg.r = [float(value) for value in np.asarray(info["r"], dtype=np.float64).reshape(9)]
    msg.p = [float(value) for value in np.asarray(info["p"], dtype=np.float64).reshape(12)]
    return msg


def validate_realsense_devices(
    cameras: Mapping[str, str],
    *,
    width: int,
    height: int,
    ir_width: int,
    ir_height: int,
    fps: int,
    color_format: str,
    enable_infra: bool,
    color_companion_infra_cameras: Sequence[str] = (),
) -> None:
    import pyrealsense2 as rs

    devices = {}
    for device in rs.context().devices:
        name = device.get_info(rs.camera_info.name)
        if name.lower() == "platform camera":
            continue
        serial = device.get_info(rs.camera_info.serial_number)
        devices[serial] = device

    missing_devices = {name: serial for name, serial in cameras.items() if serial not in devices}
    if missing_devices:
        raise RuntimeError(f"Missing RealSense devices: {missing_devices}. Connected serials: {sorted(devices)}")

    incompatible = {}
    for name, serial in cameras.items():
        missing_profiles = missing_realsense_profiles(
            devices[serial],
            rs,
            width=width,
            height=height,
            ir_width=ir_width,
            ir_height=ir_height,
            fps=fps,
            color_format=color_format,
            enable_infra=enable_infra,
            enable_color_companion_infra=name in color_companion_infra_cameras,
        )
        if missing_profiles:
            incompatible[name] = {"serial": serial, "missing_profiles": missing_profiles}
    if incompatible:
        raise RuntimeError(f"RealSense devices do not support the configured streams: {incompatible}")


def missing_realsense_profiles(
    device: Any,
    rs: Any,
    *,
    width: int,
    height: int,
    ir_width: int,
    ir_height: int,
    fps: int,
    color_format: str,
    enable_infra: bool,
    enable_color_companion_infra: bool = False,
) -> list[str]:
    available: set[tuple[Any, int, Any, int, int, int]] = set()
    for sensor in device.sensors:
        for profile in sensor.get_stream_profiles():
            try:
                video = profile.as_video_stream_profile()
            except RuntimeError:
                continue
            available.add(
                (
                    profile.stream_type(),
                    int(profile.stream_index()),
                    profile.format(),
                    int(video.width()),
                    int(video.height()),
                    int(profile.fps()),
                )
            )

    required = [
        ("rgb", (rs.stream.color, 0, _rs_color_format(rs, color_format), int(width), int(height), int(fps)))
    ]
    if enable_infra or enable_color_companion_infra:
        required.append(
            ("ir_left", (rs.stream.infrared, 1, rs.format.y8, int(ir_width), int(ir_height), int(fps)))
        )
    if enable_infra:
        required.append(
            ("ir_right", (rs.stream.infrared, 2, rs.format.y8, int(ir_width), int(ir_height), int(fps)))
        )
    return [f"{name} {profile[3]}x{profile[4]}@{profile[5]}" for name, profile in required if profile not in available]


def configure_realsense_streams(
    config: Any,
    rs: Any,
    *,
    serial: str,
    width: int,
    height: int,
    ir_width: int,
    ir_height: int,
    fps: int,
    color_format: str,
    enable_infra: bool,
    enable_color_companion_infra: bool,
) -> None:
    """Configure published streams plus an optional unpublished IR companion.

    An optional stereo companion stream can improve RGB capture on supported
    RealSense profiles. The companion participates in the frameset without
    being published through ROS or included in recordings.
    """

    config.enable_device(str(serial))
    config.enable_stream(
        rs.stream.color,
        int(width),
        int(height),
        _rs_color_format(rs, color_format),
        int(fps),
    )
    if enable_infra or enable_color_companion_infra:
        config.enable_stream(
            rs.stream.infrared,
            1,
            int(ir_width),
            int(ir_height),
            rs.format.y8,
            int(fps),
        )
    if enable_infra:
        config.enable_stream(
            rs.stream.infrared,
            2,
            int(ir_width),
            int(ir_height),
            rs.format.y8,
            int(fps),
        )


def start_realsense_pipeline(
    pipeline_factory: Any,
    config: Any,
    *,
    attempts: int = 5,
    retry_delay_s: float = 1.0,
) -> Any:
    attempts = max(1, int(attempts))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        pipeline = pipeline_factory()
        try:
            pipeline.start(config)
            return pipeline
        except RuntimeError as exc:
            last_error = exc
            try:
                pipeline.stop()
            except Exception:
                pass
            if attempt < attempts:
                print(
                    f"[WARN] RealSense pipeline start attempt {attempt}/{attempts} failed: "
                    f"{exc}; retrying in {retry_delay_s:.1f}s",
                    flush=True,
                )
                time.sleep(max(0.0, float(retry_delay_s)))
    raise RuntimeError(f"RealSense pipeline failed to start after {attempts} attempts: {last_error}") from last_error


def apply_exposure(
    profile: Any,
    rs: Any,
    *,
    mode: str,
    exposure_us: float = REALSENSE_EXPOSURE_US,
) -> dict[str, dict[str, float]]:
    """Apply automatic or fixed exposure to every sensor that supports it.

    Exposure duration can limit the effective capture rate independently of
    the declared stream FPS. Fixed mode sets exposure explicitly; automatic
    mode leaves sensor auto-exposure enabled.
    """

    mode = normalize_exposure_mode(mode)
    exposure_us = _positive_float(exposure_us, "exposure_us")
    applied: dict[str, dict[str, float]] = {}
    for sensor in profile.get_device().sensors:
        if not sensor.supports(rs.option.exposure):
            continue
        sensor_name = (
            sensor.get_info(rs.camera_info.name)
            if sensor.supports(rs.camera_info.name)
            else type(sensor).__name__
        )
        supports_auto = sensor.supports(rs.option.enable_auto_exposure)
        if mode == EXPOSURE_MODE_AUTO:
            if supports_auto:
                sensor.set_option(rs.option.enable_auto_exposure, 1.0)
        else:
            if supports_auto:
                sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            sensor.set_option(rs.option.exposure, exposure_us)
        applied[str(sensor_name)] = {
            "auto_exposure": float(sensor.get_option(rs.option.enable_auto_exposure))
            if supports_auto
            else -1.0,
            "exposure_us": float(sensor.get_option(rs.option.exposure)),
        }
    return applied


def apply_fixed_exposure(
    profile: Any,
    rs: Any,
    *,
    exposure_us: float = REALSENSE_EXPOSURE_US,
) -> dict[str, dict[str, float]]:
    """Backward-compatible fixed-exposure helper."""

    return apply_exposure(profile, rs, mode=EXPOSURE_MODE_FIXED, exposure_us=exposure_us)


def import_cv_bridge() -> type:
    try:
        from cv_bridge import CvBridge
        ensure_cv2_importable()
        return CvBridge
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            # ROS/cv_bridge may use system packages while the runtime uses a venv.
            # conda 默认看不到 apt 的 cv2，但系统路径里已有可用 OpenCV。只补这一条
            # 标准 Ubuntu dist-packages 路径；不存在或仍不可用时继续 fail loudly。
            add_system_cv2_path()
            from cv_bridge import CvBridge

            ensure_cv2_importable()
            return CvBridge
        raise RuntimeError(
            "RealSense ROS publisher requires cv_bridge and cv2. "
            "Install ROS cv_bridge and opencv in the active Prometheus runtime environment."
        ) from exc


def ensure_cv2_importable() -> None:
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "cv2":
            raise
        add_system_cv2_path()
        import cv2  # noqa: F401


def add_system_cv2_path() -> None:
    system_dist = Path("/usr/lib/python3/dist-packages")
    if system_dist.is_dir() and str(system_dist) not in sys.path:
        sys.path.append(str(system_dist))


def make_camera_info_values(profile: Any, rs: Any, names: dict[str, str], *, enable_infra: bool) -> dict[str, Any]:
    infos: dict[str, Any] = {}
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    infos["color"] = camera_info_value(color_profile.get_intrinsics())

    if not enable_infra:
        return infos

    left_stream = profile.get_stream(rs.stream.infrared, 1)
    right_stream = profile.get_stream(rs.stream.infrared, 2)
    left_intrinsics = left_stream.as_video_stream_profile().get_intrinsics()
    right_intrinsics = right_stream.as_video_stream_profile().get_intrinsics()
    extrinsics = left_stream.get_extrinsics_to(right_stream)
    baseline_m = abs(float(extrinsics.translation[0]))
    if baseline_m <= 0.0:
        raise RuntimeError(f"invalid RealSense IR stereo baseline from extrinsics: {extrinsics.translation}")

    infos["infra1"] = camera_info_value(left_intrinsics)
    infos["infra2"] = camera_info_value(right_intrinsics, tx=-float(right_intrinsics.fx) * baseline_m)
    return infos


def camera_info_value(intrinsics: Any, *, tx: float = 0.0) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "distortion_model": "plumb_bob",
        "d": np.asarray(intrinsics.coeffs, dtype=np.float32).copy(),
        "k": np.asarray(
            [
                float(intrinsics.fx),
                0.0,
                float(intrinsics.ppx),
                0.0,
                float(intrinsics.fy),
                float(intrinsics.ppy),
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        ),
        "r": np.eye(3, dtype=np.float32).reshape(9),
        "p": np.asarray(
            [
                float(intrinsics.fx),
                0.0,
                float(intrinsics.ppx),
                float(tx),
                0.0,
                float(intrinsics.fy),
                float(intrinsics.ppy),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    }


@dataclass(frozen=True, slots=True)
class StreamPublishers:
    image: Any
    info: Any


def parse_camera_map(value: str) -> dict[str, str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("camera map must be a JSON object: name -> RealSense serial")
    return normalize_camera_map(parsed)


def normalize_camera_map(value: Mapping[str, Any]) -> dict[str, str]:
    cameras = {}
    serials: dict[str, str] = {}
    for key, serial in value.items():
        name = str(key).strip()
        if not name:
            raise ValueError("RealSense camera name must be non-empty")
        if name != safe_ros_name(name):
            raise ValueError(f"RealSense camera name {name!r} must be a ROS-safe name")
        serial = str(serial).strip()
        if not serial:
            raise ValueError(f"RealSense camera {name!r} serial must be non-empty")
        if serial in serials:
            raise ValueError(f"RealSense serial {serial!r} is used by {serials[serial]!r} and {name!r}")
        cameras[name] = serial
        serials[serial] = name
    return cameras


def normalize_companion_infra_cameras(
    value: Sequence[str] | None,
    *,
    cameras: Mapping[str, str],
) -> tuple[str, ...]:
    if value is None:
        return ()
    requested_values = [value] if isinstance(value, str) else value
    requested = {str(name).strip() for name in requested_values}
    if "" in requested:
        raise ValueError("RealSense color companion camera name must be non-empty")
    unknown = requested.difference(cameras)
    if unknown:
        raise ValueError(f"unknown RealSense camera(s) for color companion IR: {sorted(unknown)}")
    return tuple(name for name in cameras if name in requested)


def normalize_color_format(value: str) -> str:
    color_format = str(value).upper()
    if color_format not in {"RGB8", "BGR8"}:
        raise ValueError("RealSense color_format must be RGB8 or BGR8")
    return color_format


def normalize_exposure_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("_", "-")
    aliases = {
        "auto": EXPOSURE_MODE_AUTO,
        "auto-exposure": EXPOSURE_MODE_AUTO,
        "fix": EXPOSURE_MODE_FIXED,
        "fixed": EXPOSURE_MODE_FIXED,
        "fix-exposure": EXPOSURE_MODE_FIXED,
        "fixed-exposure": EXPOSURE_MODE_FIXED,
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError("RealSense exposure_mode must be auto or fixed") from exc


def camera_stream_names(camera_name: str, *, enable_infra: bool = True) -> dict[str, str]:
    base = str(camera_name)
    names = {"color": f"{base}_rgb", "color_info": f"{base}_rgb_info"}
    if enable_infra:
        names.update(
            {
                "infra1": f"{base}_ir_left",
                "infra1_info": f"{base}_ir_left_info",
                "infra2": f"{base}_ir_right",
                "infra2_info": f"{base}_ir_right_info",
            }
        )
    return names


def camera_topic_specs(
    camera_name: str,
    *,
    camera_namespace: str = CAMERA_TOPIC_PREFIX,
    enable_infra: bool = True,
) -> tuple[RosTopicSpec, ...]:
    names = camera_stream_names(camera_name, enable_infra=enable_infra)
    specs = [
        RosTopicSpec(
            key=names["color"],
            topic=stream_topics(camera_name, "color", camera_namespace=camera_namespace)[0],
            msg_type="sensor_msgs.msg.Image",
            converter="image",
            metadata=_sensor_metadata(
                kind=KIND_SENSOR_IMAGE,
                sensor=camera_name,
                field="rgb",
                stream_name=names["color"],
                lossless=False,
            ),
        ),
        RosTopicSpec(
            key=names["color_info"],
            topic=stream_topics(camera_name, "color", camera_namespace=camera_namespace)[1],
            msg_type="sensor_msgs.msg.CameraInfo",
            converter="camera_info",
            metadata=_sensor_metadata(
                kind=KIND_SENSOR_INFO,
                sensor=camera_name,
                field="rgb_info",
                stream_name=names["color_info"],
            ),
        ),
    ]
    if enable_infra:
        specs.extend(
            [
                RosTopicSpec(
                    key=names["infra1"],
                    topic=stream_topics(camera_name, "infra1", camera_namespace=camera_namespace)[0],
                    msg_type="sensor_msgs.msg.Image",
                    converter="image",
                    metadata=_sensor_metadata(
                        kind=KIND_SENSOR_IMAGE,
                        sensor=camera_name,
                        field="ir_left",
                        stream_name=names["infra1"],
                        lossless=True,
                    ),
                ),
                RosTopicSpec(
                    key=names["infra1_info"],
                    topic=stream_topics(camera_name, "infra1", camera_namespace=camera_namespace)[1],
                    msg_type="sensor_msgs.msg.CameraInfo",
                    converter="camera_info",
                    metadata=_sensor_metadata(
                        kind=KIND_SENSOR_INFO,
                        sensor=camera_name,
                        field="ir_left_info",
                        stream_name=names["infra1_info"],
                    ),
                ),
                RosTopicSpec(
                    key=names["infra2"],
                    topic=stream_topics(camera_name, "infra2", camera_namespace=camera_namespace)[0],
                    msg_type="sensor_msgs.msg.Image",
                    converter="image",
                    metadata=_sensor_metadata(
                        kind=KIND_SENSOR_IMAGE,
                        sensor=camera_name,
                        field="ir_right",
                        stream_name=names["infra2"],
                        lossless=True,
                    ),
                ),
                RosTopicSpec(
                    key=names["infra2_info"],
                    topic=stream_topics(camera_name, "infra2", camera_namespace=camera_namespace)[1],
                    msg_type="sensor_msgs.msg.CameraInfo",
                    converter="camera_info",
                    metadata=_sensor_metadata(
                        kind=KIND_SENSOR_INFO,
                        sensor=camera_name,
                        field="ir_right_info",
                        stream_name=names["infra2_info"],
                    ),
                ),
            ]
        )
    return tuple(specs)


def stream_topics(camera_name: str, field: str, *, camera_namespace: str) -> tuple[str, str]:
    if field == "color":
        stream_name = camera_stream_names(camera_name, enable_infra=False)["color"]
        base = f"{camera_namespace.rstrip('/')}/{stream_name}"
        return f"{base}/image_raw", f"{base}/camera_info"
    if field == "infra1":
        stream_name = camera_stream_names(camera_name, enable_infra=True)["infra1"]
        base = f"{camera_namespace.rstrip('/')}/{stream_name}"
        return f"{base}/image_raw", f"{base}/camera_info"
    if field == "infra2":
        stream_name = camera_stream_names(camera_name, enable_infra=True)["infra2"]
        base = f"{camera_namespace.rstrip('/')}/{stream_name}"
        return f"{base}/image_raw", f"{base}/camera_info"
    raise ValueError(f"unknown RealSense stream field: {field}")


def image_stream_fields(*, enable_infra: bool) -> set[str]:
    fields = {"color"}
    if enable_infra:
        fields.update({"infra1", "infra2"})
    return fields


def frame_id(camera_name: str, field: str) -> str:
    if field == "color":
        return f"{camera_name}_color_optical_frame"
    if field == "infra1":
        return f"{camera_name}_infra1_optical_frame"
    if field == "infra2":
        return f"{camera_name}_infra2_optical_frame"
    raise ValueError(f"unknown RealSense stream field: {field}")


def _sensor_metadata(
    *,
    kind: str,
    sensor: str,
    field: str,
    stream_name: str,
    lossless: bool | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        META_KIND: kind,
        META_SENSOR: str(sensor),
        META_FIELD: str(field),
        META_STREAM_NAME: str(stream_name),
    }
    if lossless is not None:
        metadata[META_LOSSLESS] = bool(lossless)
    return metadata


def _rs_color_format(rs: Any, color_format: str) -> Any:
    color_format = normalize_color_format(color_format)
    return rs.format.rgb8 if color_format == "RGB8" else rs.format.bgr8


def _positive_int(value: Any, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(value: Any, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
