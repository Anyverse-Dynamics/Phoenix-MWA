from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from prometheus.core.sample import StreamSample


class FfplayImagePreview:
    """Display the latest image sample without blocking its producer."""

    def __init__(
        self,
        *,
        enabled: bool,
        stream_key: str,
        fps: float = 30.0,
        ffplay: str = "ffplay",
        window_title: str = "Prometheus camera preview",
    ):
        self.enabled = bool(enabled)
        self.stream_key = str(stream_key)
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError("preview fps must be positive")
        self.ffplay = str(ffplay)
        self.window_title = str(window_title)

        self._condition = threading.Condition()
        self._latest: tuple[np.ndarray, str] | None = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._warned_bad_frame = False

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print(
                "[run_env] head preview disabled: DISPLAY/WAYLAND_DISPLAY is not set; "
                "start run_env in a graphical session or use SSH X11 forwarding",
                flush=True,
            )
            self.enabled = False
            return
        if shutil.which(self.ffplay) is None:
            print(f"[run_env] head preview disabled: ffplay not found: {self.ffplay}", flush=True)
            self.enabled = False
            return

        with self._condition:
            self._stop = False
        self._thread = threading.Thread(target=self._run, name="head-camera-preview", daemon=True)
        self._thread.start()
        print(f"[run_env] head preview waiting for {self.stream_key}", flush=True)

    def submit(self, sample: StreamSample) -> None:
        if not self.enabled or sample.key != self.stream_key:
            return
        try:
            frame, pixel_format = prepare_preview_frame(
                sample.value,
                encoding=sample.metadata.get("encoding"),
            )
        except (TypeError, ValueError) as exc:
            if not self._warned_bad_frame:
                print(f"[run_env] head preview ignored invalid frame: {exc}", flush=True)
                self._warned_bad_frame = True
            return

        with self._condition:
            if self._stop:
                return
            self._latest = (frame, pixel_format)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._latest = None
            self._condition.notify_all()

        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        process = self._process
        if thread is not None and thread.is_alive() and process is not None and process.poll() is None:
            process.terminate()
            thread.join(timeout=1.0)
        self._thread = None
        self._close_process()

    def _run(self) -> None:
        expected: tuple[int, int, str] | None = None
        try:
            while True:
                with self._condition:
                    while self._latest is None and not self._stop:
                        self._condition.wait()
                    if self._stop:
                        return
                    frame, pixel_format = self._latest
                    self._latest = None

                height, width = frame.shape[:2]
                frame_spec = (height, width, pixel_format)
                if expected is None:
                    expected = frame_spec
                    self._process = self._start_process(width=width, height=height, pixel_format=pixel_format)
                    print(f"[run_env] head preview showing {width}x{height} {pixel_format}", flush=True)
                elif frame_spec != expected:
                    if not self._warned_bad_frame:
                        print(
                            f"[run_env] head preview ignored frame spec {frame_spec}; expected {expected}",
                            flush=True,
                        )
                        self._warned_bad_frame = True
                    continue

                process = self._process
                if process is None or process.poll() is not None or process.stdin is None:
                    print("[run_env] head preview window closed; rollout continues", flush=True)
                    return
                write_all(process.stdin, frame)
        except (BrokenPipeError, OSError) as exc:
            print(f"[run_env] head preview stopped ({exc}); rollout continues", flush=True)
        finally:
            self.enabled = False
            self._close_process()

    def _start_process(self, *, width: int, height: int, pixel_format: str) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [
                self.ffplay,
                "-loglevel",
                "error",
                "-nostats",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-framedrop",
                "-f",
                "rawvideo",
                "-pixel_format",
                pixel_format,
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                f"{self.fps:g}",
                "-window_title",
                self.window_title,
                "-i",
                "pipe:0",
            ],
            stdin=subprocess.PIPE,
            bufsize=0,
        )

    def _close_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)


class FfplayMosaicPreview:
    """Display several latest image streams in one non-blocking ffplay window."""

    def __init__(
        self,
        *,
        enabled: bool,
        stream_rows: Sequence[Sequence[Mapping[str, Any] | str]],
        fps: float = 20.0,
        width: int = 1280,
        height: int = 720,
        stale_timeout_s: float = 1.0,
        ffplay: str = "ffplay",
        window_title: str = "Prometheus sensor preview",
    ):
        self.enabled = bool(enabled)
        self.stream_rows = normalize_mosaic_stream_rows(stream_rows)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.stale_timeout_s = float(stale_timeout_s)
        if self.fps <= 0:
            raise ValueError("preview fps must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("preview width and height must be positive")
        if self.stale_timeout_s <= 0:
            raise ValueError("preview stale_timeout_s must be positive")
        self.ffplay = str(ffplay)
        self.window_title = str(window_title)

        self._stream_keys = {key for row in self.stream_rows for key, _label in row}
        self._condition = threading.Condition()
        self._latest: dict[str, tuple[np.ndarray, str, float]] = {}
        self._frame_counts: dict[str, int] = {key: 0 for key in self._stream_keys}
        self._frame_times: dict[str, deque[float]] = {
            key: deque(maxlen=240) for key in self._stream_keys
        }
        self._stop = False
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._cv2: Any | None = None
        self._warned_bad_streams: set[str] = set()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print(
                "[run_env] sensor preview disabled: DISPLAY/WAYLAND_DISPLAY is not set; "
                "start run_env in a graphical session or use SSH X11 forwarding",
                flush=True,
            )
            self.enabled = False
            return
        if shutil.which(self.ffplay) is None:
            print(f"[run_env] sensor preview disabled: ffplay not found: {self.ffplay}", flush=True)
            self.enabled = False
            return
        try:
            import cv2
        except ImportError as exc:
            print(f"[run_env] sensor preview disabled: OpenCV is unavailable ({exc})", flush=True)
            self.enabled = False
            return

        self._cv2 = cv2
        with self._condition:
            self._stop = False
        self._thread = threading.Thread(target=self._run, name="sensor-mosaic-preview", daemon=True)
        self._thread.start()
        print(
            f"[run_env] sensor preview waiting for {sorted(self._stream_keys)}",
            flush=True,
        )

    def submit(self, sample: StreamSample) -> None:
        if not self.enabled or sample.key not in self._stream_keys:
            return
        try:
            frame, pixel_format = prepare_preview_frame(
                sample.value,
                encoding=sample.metadata.get("encoding"),
            )
        except (TypeError, ValueError) as exc:
            if sample.key not in self._warned_bad_streams:
                print(f"[run_env] sensor preview ignored {sample.key}: {exc}", flush=True)
                self._warned_bad_streams.add(sample.key)
            return

        with self._condition:
            if self._stop:
                return
            received_at = time.monotonic()
            self._latest[sample.key] = (frame, pixel_format, received_at)
            self._frame_counts[sample.key] += 1
            self._frame_times[sample.key].append(received_at)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._latest.clear()
            self._condition.notify_all()

        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        process = self._process
        if thread is not None and thread.is_alive() and process is not None and process.poll() is None:
            process.terminate()
            thread.join(timeout=1.0)
        self._thread = None
        self._close_process()

    def _run(self) -> None:
        period_s = 1.0 / self.fps
        try:
            with self._condition:
                while not self._latest and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return

            self._process = self._start_process()
            print(
                f"[run_env] sensor preview showing {self.width}x{self.height} "
                f"rows={len(self.stream_rows)}",
                flush=True,
            )
            next_draw = time.monotonic()
            while True:
                with self._condition:
                    if self._stop:
                        return
                    now = time.monotonic()
                    if now < next_draw:
                        self._condition.wait(timeout=next_draw - now)
                        continue
                    latest = dict(self._latest)
                    counts = dict(self._frame_counts)
                    rates = {
                        key: preview_stream_fps(timestamps, now=now)
                        for key, timestamps in self._frame_times.items()
                    }

                process = self._process
                if process is None or process.poll() is not None or process.stdin is None:
                    print("[run_env] sensor preview window closed; rollout continues", flush=True)
                    return
                canvas = self._canvas(latest, counts, rates=rates, now=time.monotonic())
                encoded_ok, encoded = self._cv2.imencode(
                    ".jpg",
                    canvas,
                    [self._cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                if not encoded_ok:
                    raise RuntimeError("OpenCV failed to encode preview mosaic")
                write_all(process.stdin, encoded)
                next_draw = max(next_draw + period_s, time.monotonic())
        except (BrokenPipeError, OSError) as exc:
            print(f"[run_env] sensor preview stopped ({exc}); rollout continues", flush=True)
        finally:
            self.enabled = False
            self._close_process()

    def _canvas(
        self,
        latest: Mapping[str, tuple[np.ndarray, str, float]],
        counts: Mapping[str, int],
        *,
        rates: Mapping[str, float] | None = None,
        now: float,
    ) -> np.ndarray:
        cv2 = self._cv2
        assert cv2 is not None

        header_h = max(72, self.height // 9)
        canvas = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
        cv2.putText(
            canvas,
            "PROMETHEUS SENSOR PREVIEW",
            (24, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "top: RGB cameras    bottom: tactile difference",
            (26, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

        rows = len(self.stream_rows)
        cols = max(len(row) for row in self.stream_rows)
        gap = 10
        area_h = self.height - header_h
        tile_w = max(1, (self.width - gap * (cols + 1)) // cols)
        tile_h = max(1, (area_h - gap * (rows + 1)) // rows)
        for row_index, row in enumerate(self.stream_rows):
            for column, (key, label) in enumerate(row):
                tile = np.full((tile_h, tile_w, 3), 26, dtype=np.uint8)
                sample = latest.get(key)
                age_s = float("inf")
                if sample is None:
                    draw_center_text(cv2, tile, "waiting for frames")
                else:
                    frame, pixel_format, received_at = sample
                    age_s = max(0.0, now - received_at)
                    bgr = (
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if pixel_format == "rgb24"
                        else frame
                    )
                    tile = letterbox_image(cv2, bgr, tile_w, tile_h)
                    if age_s > self.stale_timeout_s:
                        stale_layer = np.full_like(tile, (25, 25, 115))
                        tile = cv2.addWeighted(tile, 0.35, stale_layer, 0.65, 0.0)
                        draw_center_text(
                            cv2,
                            tile,
                            f"STALE {age_s:.1f}s",
                            color=(255, 255, 255),
                            thickness=2,
                        )
                        cv2.rectangle(
                            tile,
                            (1, 1),
                            (tile.shape[1] - 2, tile.shape[0] - 2),
                            (40, 40, 255),
                            3,
                        )

                label_text = f"{label}  frames:{counts.get(key, 0)}"
                if rates is not None:
                    label_text += f" fps:{rates.get(key, 0.0):.1f}"
                if age_s != float("inf"):
                    label_text += f" age:{age_s:.2f}s"
                label_color = (
                    (80, 80, 255)
                    if age_s > self.stale_timeout_s
                    else (255, 255, 255)
                )
                cv2.rectangle(tile, (0, 0), (tile.shape[1], 36), (0, 0, 0), -1)
                cv2.putText(
                    tile,
                    label_text[:72],
                    (9, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    label_color,
                    1,
                    cv2.LINE_AA,
                )
                x = gap + column * (tile_w + gap)
                y = header_h + gap + row_index * (tile_h + gap)
                canvas[y : y + tile_h, x : x + tile_w] = tile
        return np.ascontiguousarray(canvas)

    def _start_process(self) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [
                self.ffplay,
                "-loglevel",
                "error",
                "-nostats",
                "-flags",
                "low_delay",
                "-framedrop",
                "-f",
                "mjpeg",
                "-framerate",
                f"{self.fps:g}",
                "-window_title",
                self.window_title,
                "-i",
                "pipe:0",
            ],
            stdin=subprocess.PIPE,
            bufsize=0,
        )

    def _close_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)


class RosMosaicPreviewProcess:
    """Launch a V2-style ROS/OpenCV preview sidecar.

    This preview subscribes to ROS image topics directly in its own process, so
    run_env inference, recording, and observation fanout stay on the V3 path.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        camera_topics: Mapping[str, str],
        tactile_image_topics: Mapping[str, str],
        fps: float = 20.0,
        width: int = 1280,
        height: int = 720,
        stale_timeout_s: float = 1.0,
        window_title: str = "Prometheus sensor preview",
        ros_domain_id: int | None = None,
        python_executable: str | None = None,
    ):
        self.enabled = bool(enabled)
        self.camera_topics = {str(key): str(value) for key, value in dict(camera_topics).items()}
        self.tactile_image_topics = {str(key): str(value) for key, value in dict(tactile_image_topics).items()}
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.stale_timeout_s = float(stale_timeout_s)
        if self.fps <= 0:
            raise ValueError("preview fps must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("preview width and height must be positive")
        if self.stale_timeout_s <= 0:
            raise ValueError("preview stale_timeout_s must be positive")
        self.window_title = str(window_title)
        self.ros_domain_id = None if ros_domain_id is None else int(ros_domain_id)
        self.python_executable = str(python_executable or sys.executable)
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if not self.enabled or self._process is not None:
            return
        if not self.camera_topics and not self.tactile_image_topics:
            print("[run_env] ROS sensor preview disabled: no image topics configured", flush=True)
            self.enabled = False
            return
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print(
                "[run_env] ROS sensor preview disabled: DISPLAY/WAYLAND_DISPLAY is not set; "
                "start run_env in a graphical session or use SSH X11 forwarding",
                flush=True,
            )
            self.enabled = False
            return

        env = os.environ.copy()
        if self.ros_domain_id is not None:
            env["ROS_DOMAIN_ID"] = str(self.ros_domain_id)
        cmd = [
            self.python_executable,
            "-m",
            "prometheus.visualization.camera_preview",
            "--camera_topics",
            json.dumps(self.camera_topics, separators=(",", ":"), ensure_ascii=True),
            "--tactile_image_topics",
            json.dumps(self.tactile_image_topics, separators=(",", ":"), ensure_ascii=True),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--fps",
            f"{self.fps:g}",
            "--stale_timeout_s",
            f"{self.stale_timeout_s:g}",
            "--window",
            self.window_title,
        ]
        if self.ros_domain_id is not None:
            cmd.extend(["--ros_domain_id", str(self.ros_domain_id)])
        self._process = subprocess.Popen(cmd, env=env)
        print(
            "[run_env] ROS sensor preview sidecar started "
            f"pid={self._process.pid} camera={sorted(self.camera_topics)} "
            f"tactile={sorted(self.tactile_image_topics)}",
            flush=True,
        )

    def submit(self, _sample: StreamSample) -> None:
        return

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)


def normalize_mosaic_stream_rows(
    value: Sequence[Sequence[Mapping[str, Any] | str]],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    rows: list[tuple[tuple[str, str], ...]] = []
    seen: set[str] = set()
    for row in value:
        normalized_row: list[tuple[str, str]] = []
        for item in row:
            if isinstance(item, Mapping):
                key = str(item.get("key", "")).strip()
                label = str(item.get("label", key)).strip()
            else:
                key = str(item).strip()
                label = key
            if not key:
                raise ValueError("mosaic preview stream key must be non-empty")
            if key in seen:
                raise ValueError(f"duplicate mosaic preview stream key {key!r}")
            seen.add(key)
            normalized_row.append((key, label or key))
        if not normalized_row:
            raise ValueError("mosaic preview rows must be non-empty")
        rows.append(tuple(normalized_row))
    if not rows:
        raise ValueError("mosaic preview requires at least one row")
    return tuple(rows)


def preview_stream_fps(
    timestamps: Sequence[float],
    *,
    now: float,
    window_s: float = 2.0,
) -> float:
    """Estimate receive FPS from recent monotonic timestamps."""

    if window_s <= 0:
        raise ValueError("preview FPS window must be positive")
    cutoff = float(now) - float(window_s)
    recent = [float(value) for value in timestamps if cutoff <= float(value) <= now]
    if len(recent) < 2:
        return 0.0
    elapsed = recent[-1] - recent[0]
    if elapsed <= 0:
        return 0.0
    return float(len(recent) - 1) / elapsed


def draw_center_text(
    cv2: Any,
    tile: np.ndarray,
    text: str,
    *,
    color: tuple[int, int, int] = (180, 180, 180),
    thickness: int = 1,
) -> None:
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    x = max(8, (tile.shape[1] - text_w) // 2)
    y = max(40, (tile.shape[0] + text_h) // 2)
    cv2.putText(
        tile,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        thickness,
        cv2.LINE_AA,
    )


def letterbox_image(cv2: Any, image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_w = max(1, int(round(image.shape[1] * scale)))
    resized_h = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    tile[y : y + resized_h, x : x + resized_w] = resized
    return tile


def prepare_preview_frame(value: Any, *, encoding: Any) -> tuple[np.ndarray, str]:
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {frame.dtype}")

    normalized_encoding = str(encoding or "rgb8").lower()
    if normalized_encoding == "rgb8":
        pixel_format = "rgb24"
    elif normalized_encoding == "bgr8":
        pixel_format = "bgr24"
    else:
        raise ValueError(f"unsupported encoding {encoding!r}; expected rgb8 or bgr8")
    return np.ascontiguousarray(frame), pixel_format


def write_all(stream: Any, value: Any) -> None:
    remaining = memoryview(value).cast("B")
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise BrokenPipeError("preview pipe accepted no frame data")
        remaining = remaining[written:]
