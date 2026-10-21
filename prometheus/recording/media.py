from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prometheus.core.metadata import META_LOSSLESS


@dataclass(slots=True)
class _VideoStream:
    name: str
    video: str
    path: Path
    proc: subprocess.Popen[bytes]
    shape: tuple[int, ...]
    dtype: str
    pix_fmt: str
    container: str
    codec: str
    lossless: bool
    fps: float
    crf: int | None
    preset: str | None
    frames: int = 0


class FFmpegImageWriter:
    """Write image samples to per-stream video files with ffmpeg.

    这是 media 编码边界：输入是 numpy image，输出是 EpisodeWriter metadata。
    它不理解 robot、tactile bundle、episode 状态机。
    """

    def __init__(
        self,
        *,
        episode_dir: str | Path,
        fps: float = 30.0,
        ffmpeg: str = "ffmpeg",
        overwrite: bool = False,
        video_codec: str = "libx264",
        video_crf: int = 14,
        video_preset: str = "veryfast",
        fps_by_stream: Mapping[str, float] | None = None,
    ):
        self.episode_dir = Path(episode_dir)
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError("FFmpegImageWriter fps must be positive")
        self.ffmpeg = str(ffmpeg)
        self.overwrite = bool(overwrite)
        self.video_codec = str(video_codec).strip()
        if not self.video_codec:
            raise ValueError("FFmpegImageWriter video_codec must not be empty")
        self.video_crf = int(video_crf)
        if not 0 <= self.video_crf <= 51:
            raise ValueError("FFmpegImageWriter video_crf must be in [0, 51]")
        self.video_preset = str(video_preset).strip()
        if not self.video_preset:
            raise ValueError("FFmpegImageWriter video_preset must not be empty")
        self.fps_by_stream = {str(name): float(value) for name, value in dict(fps_by_stream or {}).items()}
        invalid = {name: value for name, value in self.fps_by_stream.items() if value <= 0}
        if invalid:
            raise ValueError(f"FFmpegImageWriter fps_by_stream values must be positive: {invalid}")
        self._streams: dict[str, _VideoStream] = {}
        self._closed = False

    def write_frame(
        self,
        *,
        name: str,
        image: np.ndarray,
        stamp_ns: int,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("FFmpegImageWriter is already closed")
        stream_name = str(name)
        frame, pix_fmt = _normalize_frame(image, encoding=str(metadata.get("encoding", "")))
        lossless = bool(metadata.get(META_LOSSLESS, False))
        stream = self._streams.get(stream_name)
        if stream is None:
            stream = self._open_stream(stream_name, frame=frame, pix_fmt=pix_fmt, lossless=lossless)
            self._streams[stream_name] = stream
        _check_stream_shape(stream, frame=frame, pix_fmt=pix_fmt)
        assert stream.proc.stdin is not None
        stream.proc.stdin.write(frame.tobytes(order="C"))
        stream.frames += 1
        media = {
            "video": stream.video,
            "container": stream.container,
            "codec": stream.codec,
            "pixel_format": stream.pix_fmt,
            "lossless": stream.lossless,
            "fps": stream.fps,
        }
        if not stream.lossless:
            media.update({"crf": stream.crf, "preset": stream.preset})
        return media

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for stream in self._streams.values():
            try:
                _close_stream(stream)
            except BaseException as exc:
                errors.append(exc)
        self._closed = True
        if errors:
            raise RuntimeError("FFmpegImageWriter close failed") from errors[0]

    def __enter__(self) -> "FFmpegImageWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _open_stream(self, name: str, *, frame: np.ndarray, pix_fmt: str, lossless: bool) -> _VideoStream:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".mkv" if lossless else ".mp4"
        video = f"{name}{suffix}"
        path = self.episode_dir / video
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"media file already exists: {path}")
        height, width = frame.shape[:2]
        container = "matroska" if lossless else "mp4"
        codec = "ffv1" if lossless else self.video_codec
        fps = float(self.fps_by_stream.get(name, self.fps))
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if self.overwrite else "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            codec,
        ]
        crf = None
        preset = None
        if not lossless:
            crf = self.video_crf
            preset = self.video_preset
            command.extend(
                [
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        command.append(str(path))
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return _VideoStream(
            name=name,
            video=video,
            path=path,
            proc=proc,
            shape=tuple(frame.shape),
            dtype=str(frame.dtype),
            pix_fmt=pix_fmt,
            container=container,
            codec=codec,
            lossless=lossless,
            fps=fps,
            crf=crf,
            preset=preset,
        )


def _normalize_frame(image: np.ndarray, *, encoding: str) -> tuple[np.ndarray, str]:
    frame = np.asarray(image)
    if frame.ndim == 2:
        if frame.dtype == np.uint8:
            return np.ascontiguousarray(frame), "gray"
        if frame.dtype == np.uint16:
            return np.ascontiguousarray(frame.astype("<u2", copy=False)), "gray16le"
        raise ValueError(f"2D image dtype must be uint8 or uint16, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
        raise ValueError(f"image must be HxW, HxWx3, or HxWx4, got shape={frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"multi-channel image dtype must be uint8, got {frame.dtype}")
    encoding = encoding.lower()
    if frame.shape[2] == 3:
        pix_fmt = "rgb24" if encoding == "rgb8" else "bgr24"
    else:
        pix_fmt = "rgba" if encoding == "rgba8" else "bgra"
    return np.ascontiguousarray(frame), pix_fmt


def _check_stream_shape(stream: _VideoStream, *, frame: np.ndarray, pix_fmt: str) -> None:
    if tuple(frame.shape) != stream.shape or str(frame.dtype) != stream.dtype or pix_fmt != stream.pix_fmt:
        raise ValueError(
            f"stream {stream.name!r} changed format: "
            f"expected shape={stream.shape} dtype={stream.dtype} pix_fmt={stream.pix_fmt}, "
            f"got shape={tuple(frame.shape)} dtype={frame.dtype} pix_fmt={pix_fmt}"
        )


def _close_stream(stream: _VideoStream) -> None:
    assert stream.proc.stdin is not None
    stream.proc.stdin.close()
    try:
        stream.proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        stream.proc.kill()
        stream.proc.wait(timeout=1.0)
        raise TimeoutError(f"ffmpeg did not stop for {stream.path}")
    if stream.proc.returncode != 0:
        stderr = stream.proc.stderr.read() if stream.proc.stderr is not None else b""
        detail = stderr.decode("utf-8", errors="replace") if stderr else ""
        raise RuntimeError(f"ffmpeg failed for {stream.path}: {detail}")
