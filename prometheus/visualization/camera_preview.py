from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np


CAMERA_ROW = (
    ("left_wrist_0_rgb", "left0 rgb"),
    ("right_wrist_0_rgb", "right0 rgb"),
    ("base_0_rgb", "base0 rgb"),
)

TACTILE_DIFF_ROW = (
    ("left_wrist_0_difference", "left0 tactile diff"),
    ("left_wrist_1_difference", "left1 tactile diff"),
    ("right_wrist_0_difference", "right0 tactile diff"),
    ("right_wrist_1_difference", "right1 tactile diff"),
)


class CameraPreview:
    """Read-only ROS image preview matching the V2 collection sidecar style."""

    def __init__(self, args: argparse.Namespace, *, context: Any):
        import cv2
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        class _Node(Node):
            pass

        self.cv2 = cv2
        self.rclpy = rclpy
        self.bridge = CvBridge()
        self.context = context
        self.node = _Node("prometheus_camera_preview", context=context)
        self.executor = SingleThreadedExecutor(context=context)
        self.executor.add_node(self.node)
        self.window = str(args.window)
        self.width = int(args.width)
        self.height = int(args.height)
        self.fps = float(args.fps)
        self.stale_timeout_s = float(args.stale_timeout_s)
        self.started_at = time.monotonic()
        all_camera_topics = parse_topic_map(args.camera_topics, "camera_topics")
        all_tactile_image_topics = parse_topic_map(args.tactile_image_topics, "tactile_image_topics")
        self.stream_rows = [
            row
            for row in (
                ordered_preview_streams(all_camera_topics, CAMERA_ROW, row_name="rgb"),
                ordered_preview_streams(all_tactile_image_topics, TACTILE_DIFF_ROW, row_name="tactile_diff"),
            )
            if row
        ]
        self.streams = [stream for row in self.stream_rows for stream in row]
        self.latest: dict[str, np.ndarray] = {}
        self.received_at: dict[str, float] = {}
        self.frame_counts: dict[str, int] = {stream["key"]: 0 for stream in self.streams}

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for stream in self.streams:
            topic = stream["topic"]
            if topic:
                self.node.create_subscription(Image, topic, self._image_cb(stream["key"]), image_qos)
        if args.status_topic:
            self.node.create_subscription(String, str(args.status_topic), self._status_cb, 10)

        self.status = "IDLE"
        self.status_detail = ""
        self.status_at = 0.0

    def _image_cb(self, key: str):
        def callback(msg: Any) -> None:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest[key] = self._as_bgr(image, str(msg.encoding))
            self.received_at[key] = time.monotonic()
            self.frame_counts[key] += 1

        return callback

    def _status_cb(self, msg: Any) -> None:
        try:
            payload = json.loads(str(msg.data))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        status = payload.get("status")
        if isinstance(status, str):
            self.status = status.upper()
            self.status_detail = str(payload.get("detail", ""))
            self.status_at = time.monotonic()

    def _as_bgr(self, image: np.ndarray, encoding: str) -> np.ndarray:
        cv2 = self.cv2
        encoding = encoding.lower()
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image[:, :, :3] if image.ndim == 3 and image.shape[2] > 3 else image

    def run(self) -> None:
        cv2 = self.cv2
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if has_display:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, self.width, self.height)
        else:
            self.node.get_logger().warn("DISPLAY/WAYLAND_DISPLAY is not set; preview window will not be shown.")

        draw_period = 1.0 / max(1.0, self.fps)
        next_draw = 0.0
        while self.context.ok():
            self.executor.spin_once(timeout_sec=0.001)
            if not has_display:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if now >= next_draw:
                cv2.imshow(self.window, self._canvas(now))
                next_draw = now + draw_period
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if has_display:
            cv2.destroyAllWindows()

    def _canvas(self, now: float) -> np.ndarray:
        cv2 = self.cv2
        header_h = max(150, self.height // 5)
        canvas = np.full((self.height, self.width, 3), status_color(self.status), dtype=np.uint8)
        title = "RECORDING" if self.status == "RUNNING" else "PROMETHEUS PREVIEW"
        cv2.putText(canvas, title, (34, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        age = now - self.status_at if self.status_at else 0.0
        lines = [
            f"status: {self.status}  age: {age:.1f}s",
            "top: RGB cameras    bottom: tactile difference images",
        ]
        if self.status_detail:
            lines.append(f"detail: {self.status_detail[:90]}")
        y = 112
        for line in lines:
            cv2.putText(canvas, line, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28
        area = canvas[header_h:, :]
        area[:] = 18
        for tile, x, y in self._tiles(area.shape[1], area.shape[0], now):
            h, w = tile.shape[:2]
            area[y : y + h, x : x + w] = tile
        return canvas

    def _tiles(self, area_w: int, area_h: int, now: float):
        cv2 = self.cv2
        cols = max((len(row) for row in self.stream_rows), default=1)
        rows = max(1, len(self.stream_rows))
        gap = 12
        tile_w = max(1, (area_w - gap * (cols + 1)) // cols)
        tile_h = max(1, (area_h - gap * (rows + 1)) // rows)
        tiles = []
        for row_idx, row in enumerate(self.stream_rows):
            for col, stream in enumerate(row):
                key = stream["key"]
                image = self.latest.get(key)
                tile = np.full((tile_h, tile_w, 3), 26, dtype=np.uint8)
                stream_age = now - self.received_at.get(key, self.started_at)
                unhealthy = False
                if image is None:
                    unhealthy = stream_age >= self.stale_timeout_s
                    self._draw_center_text(
                        tile,
                        "NO SIGNAL" if unhealthy else "waiting for frames",
                        color=(80, 80, 255) if unhealthy else (180, 180, 180),
                    )
                else:
                    tile = self._letterbox(image, tile_w, tile_h)
                    unhealthy = stream_age >= self.stale_timeout_s
                    if unhealthy:
                        overlay = tile.copy()
                        overlay[:] = (25, 25, 120)
                        tile = cv2.addWeighted(tile, 0.55, overlay, 0.45, 0.0)
                        self._draw_center_text(tile, f"STALE {stream_age:.1f}", color=(255, 255, 255))
                label = f"{stream['label']}  frames:{self.frame_counts.get(key, 0)} age:{stream_age:.2f}s"
                cv2.rectangle(tile, (0, 0), (tile.shape[1], 42), (0, 0, 0), -1)
                cv2.putText(tile, label[:80], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                if unhealthy:
                    cv2.rectangle(tile, (1, 1), (tile.shape[1] - 2, tile.shape[0] - 2), (40, 40, 255), 5)
                tiles.append((tile, gap + col * (tile_w + gap), gap + row_idx * (tile_h + gap)))
        return tiles

    def _draw_center_text(
        self,
        tile: np.ndarray,
        text: str,
        *,
        color: tuple[int, int, int] = (180, 180, 180),
    ) -> None:
        cv2 = self.cv2
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
        x = max(12, (tile.shape[1] - text_w) // 2)
        y = max(48, (tile.shape[0] + text_h) // 2)
        cv2.putText(tile, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)

    def _letterbox(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        cv2 = self.cv2
        scale = min(width / image.shape[1], height / image.shape[0])
        resized_w = max(1, int(round(image.shape[1] * scale)))
        resized_h = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        tile = np.zeros((height, width, 3), dtype=np.uint8)
        x = (width - resized_w) // 2
        y = (height - resized_h) // 2
        tile[y : y + resized_h, x : x + resized_w] = resized
        return tile

    def destroy(self) -> None:
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        self.node.destroy_node()


def status_color(status: str) -> tuple[int, int, int]:
    status = status.upper()
    if status == "RUNNING":
        return (45, 155, 55)
    if status == "ERROR":
        return (30, 35, 190)
    if status == "SAVED":
        return (170, 105, 30)
    return (205, 90, 25)


def ordered_preview_streams(
    topics: dict[str, str],
    order: tuple[tuple[str, str], ...],
    *,
    row_name: str,
) -> list[dict[str, str]]:
    streams = []
    for name, label in order:
        if name not in topics:
            continue
        streams.append(
            {
                "key": f"{row_name}:{name}",
                "name": name,
                "label": label,
                "topic": str(topics[name]),
                "row": row_name,
            }
        )
    return streams


def parse_topic_map(value: str, name: str) -> dict[str, str]:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {str(key): str(topic) for key, topic in parsed.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prometheus read-only RGB camera and tactile difference preview.")
    parser.add_argument("--camera_topics", required=True)
    parser.add_argument("--tactile_image_topics", default="{}")
    parser.add_argument("--status_topic", default="")
    parser.add_argument("--window", default="Prometheus Camera Preview")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--stale_timeout_s", type=float, default=1.0)
    parser.add_argument("--ros_domain_id", type=int, default=None)
    args = parser.parse_args()
    for name in ("width", "height", "fps", "stale_timeout_s"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    return args


def main() -> None:
    args = parse_args()
    import rclpy
    from rclpy.executors import ExternalShutdownException

    context = rclpy.Context()
    rclpy.init(args=None, context=context, domain_id=args.ros_domain_id)
    preview: CameraPreview | None = None
    try:
        preview = CameraPreview(args, context=context)
        preview.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if preview is not None:
            preview.destroy()
        if context.ok():
            rclpy.shutdown(context=context)


if __name__ == "__main__":
    main()
