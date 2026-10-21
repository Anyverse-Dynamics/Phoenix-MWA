from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from prometheus.core.action import ActionTrace
from prometheus.core.sample import StreamSample
from prometheus.data import EpisodeWriter, RobotPart
from prometheus.recording.media import FFmpegImageWriter
from prometheus.recording.profile import next_episode_id
from prometheus.recording.recorder import EpisodeRecorder
from prometheus.runtime.sink import AsyncSampleSink


@dataclass(slots=True)
class EpisodeCapture:
    """One active episode writer behind a critical async queue.

    入口程序负责决定什么时候 start/close；这里只保证 sample/action trace
    不在 ROS callback 或 control path 里同步落盘。
    """

    root: Path
    episode_prefix: str
    start_index: int
    queue_size: int
    robot_parts: Sequence[RobotPart | Mapping[str, Any]]
    camera_streams: Sequence[str]
    tactile_fields_by_sensor: Mapping[str, Sequence[str]]
    tactile_bundle_timeout_ns: int
    media_fps: float
    ffmpeg: str
    metadata: Mapping[str, Any]
    workflow: str
    video_codec: str = "libx264"
    video_crf: int = 14
    video_preset: str = "veryfast"
    media_fps_by_stream: Mapping[str, float] | None = None
    record_sensors: bool | None = None
    write_visualizations: bool = True
    visualization_dpi: int = 220
    episode_id: str = ""
    writer: EpisodeWriter | None = None
    recorder: EpisodeRecorder | None = None
    sink: AsyncSampleSink | None = None

    def start(self) -> str:
        if self.sink is not None:
            raise RuntimeError("EpisodeCapture is already started")
        self.episode_id = next_episode_id(self.root, prefix=self.episode_prefix, start_index=self.start_index)
        record_sensors = (
            bool(self.camera_streams or self.tactile_fields_by_sensor)
            if self.record_sensors is None
            else bool(self.record_sensors)
        )
        writer = EpisodeWriter(
            root=self.root,
            record_robot=True,
            record_sensors=record_sensors,
            write_visualizations=self.write_visualizations,
            visualization_dpi=self.visualization_dpi,
        )
        writer.open(self.episode_id, metadata=dict(self.metadata) | {"workflow": self.workflow})
        assert writer.episode_dir is not None
        image_writer = (
            FFmpegImageWriter(
                episode_dir=writer.episode_dir,
                fps=self.media_fps,
                ffmpeg=self.ffmpeg,
                video_codec=self.video_codec,
                video_crf=self.video_crf,
                video_preset=self.video_preset,
                fps_by_stream=self.media_fps_by_stream,
            )
            if record_sensors
            else None
        )
        recorder = EpisodeRecorder(
            writer=writer,
            image_writer=image_writer,
            robot_parts=self.robot_parts,
            camera_streams=self.camera_streams,
            tactile_fields_by_sensor=self.tactile_fields_by_sensor,
            tactile_bundle_timeout_ns=self.tactile_bundle_timeout_ns,
        )
        sink = AsyncSampleSink(
            name=f"{self.workflow}:{self.episode_id}",
            consumer=self._write_payload,
            max_queue=self.queue_size,
            drop_old=False,
            close_timeout_s=None,
        )
        self.writer = writer
        self.recorder = recorder
        self.sink = sink
        sink.start()
        return self.episode_id

    def record_sample(self, sample: StreamSample) -> None:
        if self.sink is not None:
            self.sink.enqueue(sample)

    def record_action_trace(self, trace: ActionTrace) -> None:
        if self.sink is not None:
            self.sink.enqueue(trace)

    def raise_if_error(self) -> None:
        if self.sink is not None:
            self.sink.raise_if_error()

    def close(
        self,
        *,
        accept: bool,
        discard: bool = False,
        episode_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if accept and discard:
            raise ValueError("cannot accept and discard the same episode")
        sink, recorder, writer = self.sink, self.recorder, self.writer
        episode_dir = None if writer is None else writer.episode_dir
        error: BaseException | None = None
        try:
            if writer is not None and episode_metadata:
                writer.metadata.update(dict(episode_metadata))
            if sink is not None:
                try:
                    sink.close()
                except BaseException as exc:
                    accept = False
                    error = exc
            if recorder is not None:
                try:
                    recorder.close(accept=accept and error is None)
                except BaseException as exc:
                    if error is None:
                        error = exc
        finally:
            self.sink = None
            self.recorder = None
            self.writer = None
        if discard and episode_dir is not None:
            try:
                self._discard_episode_dir(episode_dir)
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def _discard_episode_dir(self, episode_dir: Path) -> None:
        root = Path(self.root).expanduser().resolve()
        target = Path(episode_dir).resolve()
        if target.parent != root or target.name != self.episode_id:
            raise RuntimeError(f"refusing to discard unexpected episode directory: {target}")
        if target.exists():
            shutil.rmtree(target)

    def _write_payload(self, payload: Any) -> None:
        recorder, writer = self.recorder, self.writer
        if recorder is None or writer is None:
            raise RuntimeError("EpisodeCapture is not open")
        if isinstance(payload, StreamSample):
            recorder.record_sample(payload)
            return
        if isinstance(payload, ActionTrace):
            writer.write_action_trace(payload)
            return
        raise TypeError(f"unsupported episode payload: {type(payload).__name__}")
