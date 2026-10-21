from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from prometheus.core.metadata import (
    KIND_SENSOR_IMAGE,
    KIND_SENSOR_POINTCLOUD,
    META_FIELD,
    META_KIND,
    META_LOSSLESS,
    META_SENSOR,
    META_STREAM_NAME,
)
from prometheus.hardware.ros import RosTopicSpec, safe_ros_name


TACTILE_POINT_FIELDS = ("x", "y", "z", "dx", "dy", "dz")
XENSE_TOPIC_PREFIX = "/prometheus/sensors/tactile"
XENSE_FRAME_FIELDS = ("rectify", "difference", "mesh", "flow", "sdk_timestamp")
XENSE_GPU_LIBRARIES = (
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libnvrtc.so.12",
    "libcudnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
)


class XensePublisher:
    """Xense tactile SDK worker -> ROS2 streams.

    Xense inference is heavy and may own GPU/runtime state, so each physical
    sensor runs in a child process. This object only owns process lifecycle and
    topic schema; it does not interpret pedal commands or downstream data semantics.
    """

    def __init__(
        self,
        *,
        sensors: Mapping[str, str],
        topic_prefix: str = XENSE_TOPIC_PREFIX,
        hz: float = 30.0,
        image_width: int = 700,
        image_height: int = 400,
        flow_width: int = 35,
        flow_height: int = 20,
        publish_rectify: bool = False,
        publish_difference: bool = True,
        publish_meshflow: bool = True,
        require_gpu: bool = True,
        gpu_library_dir: str | Path | None = None,
        qos_depth: int = 20,
        ros_domain_id: int | None = None,
    ):
        self.sensors = normalize_sensor_map(sensors)
        if not self.sensors:
            raise ValueError("XensePublisher requires at least one sensor")
        if not (publish_rectify or publish_difference or publish_meshflow):
            raise ValueError("XensePublisher requires at least one enabled stream")
        self.topic_prefix = str(topic_prefix).rstrip("/")
        self.hz = _positive_float(hz, "hz")
        self.image_width = _positive_int(image_width, "image_width")
        self.image_height = _positive_int(image_height, "image_height")
        self.flow_width = _positive_int(flow_width, "flow_width")
        self.flow_height = _positive_int(flow_height, "flow_height")
        self.publish_rectify = bool(publish_rectify)
        self.publish_difference = bool(publish_difference)
        self.publish_meshflow = bool(publish_meshflow)
        self.require_gpu = bool(require_gpu)
        self.gpu_library_dir = (
            None if gpu_library_dir is None else str(Path(gpu_library_dir).expanduser())
        )
        self.qos_depth = _positive_int(qos_depth, "qos_depth")
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        self._ctx = mp.get_context("spawn")
        self._ready_queue: Any | None = None
        self._procs: dict[str, mp.Process] = {}
        self._ready_names: set[str] = set()
        self._started = False
        self._closed = False

    def ros_topic_specs(self) -> tuple[RosTopicSpec, ...]:
        specs = []
        for sensor_name in self.sensors:
            specs.extend(
                xense_topic_specs(
                    sensor_name,
                    topic_prefix=self.topic_prefix,
                    publish_rectify=self.publish_rectify,
                    publish_difference=self.publish_difference,
                    publish_meshflow=self.publish_meshflow,
                )
            )
        return tuple(specs)

    def recording_tactile_fields_by_sensor(self) -> dict[str, tuple[str, ...]]:
        """Xense 完整 tactile bundle 需要等待哪些字段，由设备 profile 决定。"""

        fields = []
        if self.publish_rectify:
            fields.append("rectify")
        if self.publish_difference:
            fields.append("difference")
        if self.publish_meshflow:
            fields.append("meshflow")
        return {sensor_name: tuple(fields) for sensor_name in self.sensors}

    def start(self) -> None:
        if self._started:
            return
        self._ready_queue = self._ctx.Queue()
        self._ready_names.clear()
        try:
            for sensor_name, serial in self.sensors.items():
                proc = self._ctx.Process(
                    target=xense_worker,
                    name=f"xense_{sensor_name}",
                    args=(sensor_name, serial),
                    kwargs={
                        "topic_prefix": self.topic_prefix,
                        "hz": self.hz,
                        "image_width": self.image_width,
                        "image_height": self.image_height,
                        "flow_width": self.flow_width,
                        "flow_height": self.flow_height,
                        "publish_rectify": self.publish_rectify,
                        "publish_difference": self.publish_difference,
                        "publish_meshflow": self.publish_meshflow,
                        "require_gpu": self.require_gpu,
                        "gpu_library_dir": self.gpu_library_dir,
                        "qos_depth": self.qos_depth,
                        "ros_domain_id": self.ros_domain_id,
                        "ready_queue": self._ready_queue,
                    },
                )
                proc.start()
                self._procs[sensor_name] = proc
        except Exception:
            self.stop()
            raise
        self._started = True
        self._closed = False

    def wait_ready(self, timeout_s: float) -> None:
        if not self._started:
            raise RuntimeError("XensePublisher has not been started")
        deadline = time.monotonic() + float(timeout_s)
        expected = set(self.sensors)
        while time.monotonic() < deadline:
            self._drain_ready_queue()
            self._raise_on_worker_exit()
            if self._ready_names == expected:
                return
            time.sleep(0.05)
        raise TimeoutError(f"timed out waiting for Xense sensors: ready={sorted(self._ready_names)}")

    def raise_if_error(self) -> None:
        """Surface a worker crash through the run_env server health check."""

        self._raise_on_worker_exit()

    def stop(self) -> None:
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
        self._started = False

    def close(self) -> None:
        self.stop()
        self._closed = True

    def status(self) -> dict[str, Any]:
        self._drain_ready_queue()
        return {
            "type": type(self).__name__,
            "started": self._started,
            "closed": self._closed,
            "ready": self._ready_names == set(self.sensors),
            "ready_names": sorted(self._ready_names),
            "sensors": dict(self.sensors),
            "topic_prefix": self.topic_prefix,
            "hz": self.hz,
            "ros_domain_id": self.ros_domain_id,
            "publish_rectify": self.publish_rectify,
            "publish_difference": self.publish_difference,
            "publish_meshflow": self.publish_meshflow,
            "gpu_library_dir": self.gpu_library_dir,
            "processes": {name: {"pid": proc.pid, "exitcode": proc.exitcode} for name, proc in self._procs.items()},
        }

    def _drain_ready_queue(self) -> None:
        if self._ready_queue is None:
            return
        while True:
            try:
                self._ready_names.add(str(self._ready_queue.get_nowait()))
            except queue.Empty:
                return

    def _raise_on_worker_exit(self) -> None:
        dead = {name: proc.exitcode for name, proc in self._procs.items() if proc.exitcode is not None}
        if dead:
            raise RuntimeError(f"Xense worker exited: {dead}")


def xense_topic_specs(
    sensor_name: str,
    *,
    topic_prefix: str = XENSE_TOPIC_PREFIX,
    publish_rectify: bool = False,
    publish_difference: bool = True,
    publish_meshflow: bool = True,
) -> tuple[RosTopicSpec, ...]:
    root = f"{topic_prefix.rstrip('/')}/{sensor_name}"
    names = xense_stream_names(sensor_name)
    specs: list[RosTopicSpec] = []
    for field, enabled, lossless in (
        ("rectify", publish_rectify, True),
        ("difference", publish_difference, True),
    ):
        if enabled:
            specs.append(
                RosTopicSpec(
                    key=names[field],
                    topic=f"{root}/{field}",
                    msg_type="sensor_msgs.msg.Image",
                    converter="image",
                    metadata=_sensor_metadata(
                        kind=KIND_SENSOR_IMAGE,
                        sensor=sensor_name,
                        field=field,
                        stream_name=names[field],
                        lossless=lossless,
                    ),
                )
            )
    if publish_meshflow:
        specs.append(
            RosTopicSpec(
                key=names["pointcloud"],
                topic=f"{root}/pointcloud",
                msg_type="sensor_msgs.msg.PointCloud2",
                converter="pointcloud2",
                metadata=_sensor_metadata(
                    kind=KIND_SENSOR_POINTCLOUD,
                    sensor=sensor_name,
                    field="meshflow",
                    stream_name=names["pointcloud"],
                    lossless=True,
                )
                | {"fields": TACTILE_POINT_FIELDS},
            )
        )
    return tuple(specs)


def xense_stream_names(sensor_name: str) -> dict[str, str]:
    base = str(sensor_name)
    return {
        "rectify": f"{base}_rectify",
        "difference": f"{base}_difference",
        "pointcloud": f"{base}_pointcloud",
    }


def normalize_sensor_map(value: Mapping[str, Any]) -> dict[str, str]:
    sensors: dict[str, str] = {}
    serials: dict[str, str] = {}
    for key, serial in value.items():
        name = str(key).strip()
        if not name:
            raise ValueError("Xense sensor name must be non-empty")
        if name != safe_ros_name(name):
            raise ValueError(f"Xense sensor name {name!r} must be a ROS-safe name")
        serial = str(serial).strip()
        if not serial:
            raise ValueError(f"Xense sensor {name!r} serial must be non-empty")
        if serial in serials:
            raise ValueError(f"Xense serial {serial!r} is used by {serials[serial]!r} and {name!r}")
        sensors[name] = serial
        serials[serial] = name
    return sensors


def parse_sensor_map(value: str) -> dict[str, str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("sensor map must be a JSON object: name -> Xense serial")
    return normalize_sensor_map(parsed)


def preload_xense_gpu_libraries(*, library_dir: str | Path | None = None, required: bool = True) -> tuple[str, ...]:
    configured_dir = os.environ.get("XENSE_GPU_LIBRARY_DIR")
    root = Path(library_dir or configured_dir) if library_dir is not None or configured_dir else Path(sys.prefix) / "lib"
    loaded: list[str] = []
    errors: list[str] = []
    for library_name in XENSE_GPU_LIBRARIES:
        direct_path = root / library_name
        nested_paths = sorted(root.glob(f"*/lib/{library_name}"))
        path = direct_path if direct_path.exists() or not nested_paths else nested_paths[0]
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            errors.append(f"{library_name}: {exc}")
        else:
            loaded.append(library_name)
    if required and errors:
        raise RuntimeError(f"Xense GPU runtime preload failed under {root}: " + "; ".join(errors))
    return tuple(loaded)


def read_xense_frame(
    sensor: Any,
    output_type: Any,
    *,
    publish_rectify: bool,
    publish_difference: bool,
    publish_meshflow: bool,
) -> dict[str, Any]:
    requested = []
    fields = []
    if publish_rectify:
        requested.append(output_type.Rectify)
        fields.append("rectify")
    if publish_difference:
        requested.append(output_type.Difference)
        fields.append("difference")
    if publish_meshflow:
        requested.extend((output_type.Mesh3D, output_type.Mesh3DFlow))
        fields.extend(("mesh", "flow"))
    requested.append(output_type.TimeStamp)
    fields.append("sdk_timestamp")

    values = tuple(sensor.selectSensorInfo(*requested))
    if len(values) != len(fields):
        raise RuntimeError(f"Xense selectSensorInfo returned {len(values)} values; expected {len(fields)}")
    return dict(zip(fields, values, strict=True))


def xense_worker(
    sensor_name: str,
    serial: str,
    *,
    topic_prefix: str,
    hz: float,
    image_width: int,
    image_height: int,
    flow_width: int,
    flow_height: int,
    publish_rectify: bool,
    publish_difference: bool,
    publish_meshflow: bool,
    require_gpu: bool,
    gpu_library_dir: str | None,
    qos_depth: int,
    ros_domain_id: int | None,
    ready_queue: Any | None,
) -> None:
    preload_xense_gpu_libraries(
        library_dir=gpu_library_dir,
        required=require_gpu,
    )

    import numpy as np
    import rclpy
    import sensor_msgs_py.point_cloud2 as pc2
    from cv_bridge import CvBridge
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from std_msgs.msg import Header
    from xensesdk import Sensor

    shutdown_requested = False

    def request_shutdown(_signum: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    sensor = Sensor.create(serial)
    using_gpu = bool(getattr(sensor, "infer_engine_using_gpu", False))
    if require_gpu and not using_gpu:
        _release_sensor(sensor)
        raise RuntimeError(f"Xense {sensor_name} serial={serial} fell back to CPU inference while require_gpu=true")

    context = rclpy.Context()
    rclpy.init(args=None, context=context, domain_id=ros_domain_id)
    node = Node(f"prometheus_xense_{safe_ros_name(sensor_name)}", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    bridge = CvBridge()
    topic_root = topic_prefix.rstrip("/")
    image_pubs: dict[str, Any] = {}
    pointcloud_pub = None
    if publish_meshflow:
        fields = [
            PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
            for index, name in enumerate(TACTILE_POINT_FIELDS)
        ]
        pointcloud_pub = node.create_publisher(PointCloud2, f"{topic_root}/{sensor_name}/pointcloud", int(qos_depth))
    else:
        fields = []
    for stream, enabled in (("rectify", publish_rectify), ("difference", publish_difference)):
        if enabled:
            image_pubs[stream] = node.create_publisher(Image, f"{topic_root}/{sensor_name}/{stream}", int(qos_depth))

    period_s = 1.0 / float(hz)
    next_time = time.monotonic()
    ready_reported = False
    try:
        while context.ok() and not shutdown_requested:
            outputs = read_xense_frame(
                sensor,
                Sensor.OutputType,
                publish_rectify=publish_rectify,
                publish_difference=publish_difference,
                publish_meshflow=publish_meshflow,
            )
            header = Header()
            header.stamp = node.get_clock().now().to_msg()
            header.frame_id = f"{sensor_name}_tactile_frame"
            if pointcloud_pub is not None:
                mesh = np.asarray(outputs["mesh"], dtype=np.float32).reshape(-1, 3)
                flow = np.asarray(outputs["flow"], dtype=np.float32).reshape(-1, 3)
                expected_points = int(flow_width) * int(flow_height)
                if mesh.shape != (expected_points, 3):
                    raise ValueError(f"Xense {sensor_name} Mesh3D got {mesh.shape}; expected {(expected_points, 3)}")
                if flow.shape != (expected_points, 3):
                    raise ValueError(f"Xense {sensor_name} Mesh3DFlow got {flow.shape}; expected {(expected_points, 3)}")
                points = np.concatenate((mesh, flow), axis=1).astype(np.float32, copy=False)
                pointcloud_pub.publish(pc2.create_cloud(header, fields, points))
            for stream, enabled in (("rectify", publish_rectify), ("difference", publish_difference)):
                if enabled:
                    publish_rgb_image(
                        bridge,
                        image_pubs[stream],
                        sensor_name,
                        stream,
                        outputs[stream],
                        header,
                        width=image_width,
                        height=image_height,
                    )
            if not ready_reported:
                if ready_queue is not None:
                    ready_queue.put(sensor_name)
                ready_reported = True
            executor.spin_once(timeout_sec=0.0)
            next_time += period_s
            time.sleep(max(0.0, next_time - time.monotonic()))
    finally:
        _release_sensor(sensor)
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)


def publish_rgb_image(
    bridge: Any,
    publisher: Any,
    sensor_name: str,
    stream: str,
    image: Any,
    header: Any,
    *,
    width: int,
    height: int,
) -> None:
    import numpy as np

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"unsupported Xense {sensor_name} {stream} image shape: {arr.shape}")
    if arr.shape[:2] == (int(width), int(height)) and int(width) != int(height):
        arr = np.swapaxes(arr, 0, 1)
    if arr.shape[:2] != (int(height), int(width)):
        raise ValueError(f"Xense {sensor_name} {stream} must be {width}x{height}, got {arr.shape[1]}x{arr.shape[0]}")
    msg = bridge.cv2_to_imgmsg(arr, encoding="rgb8")
    msg.header.stamp = header.stamp
    msg.header.frame_id = f"{sensor_name}_{stream}_frame"
    publisher.publish(msg)


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


def _release_sensor(sensor: Any) -> None:
    release = getattr(sensor, "release", None)
    if callable(release):
        release()


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
