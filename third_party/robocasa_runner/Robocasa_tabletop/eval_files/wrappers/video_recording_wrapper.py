# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import uuid
import logging
import shutil
import subprocess
from pathlib import Path

import av
import gymnasium as gym
import numpy as np


def get_accumulate_timestamp_idxs(
    timestamps: list[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: int | None = 0,
    allow_negative: bool = False,
) -> tuple[list[int], list[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = np.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


class VideoRecorder:
    def __init__(
        self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs,
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0
        self.last_stop_ok = True

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile:v": "high"},
            **kwargs,
        )
        return obj

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1 / self.fps,
                next_global_idx=self.next_global_idx,
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)

        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h, w, c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            self.last_stop_ok = True
            return True

        try:
            # Flush stream
            for packet in self.stream.encode():
                self.container.mux(packet)

            # Close the file
            self.container.close()
            self.last_stop_ok = True
        except Exception as exc:
            logging.warning("Failed to finalize evaluation video; continuing without video. Error: %s", exc)
            self.last_stop_ok = False
        finally:
            # reset runtime parameters
            stop_ok = self.last_stop_ok
            self._reset_state()
            self.last_stop_ok = stop_ok
        return self.last_stop_ok


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_recorder: VideoRecorder,
        mode="rgb_array",
        video_dir: Path | None = None,
        steps_per_render=1,
        **kwargs,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)

        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.video_dir = video_dir
        self.video_recorder = video_recorder
        self.file_path = None

        self.step_count = 0
        self.frames = list()

        self.is_success = False

    def _probe_video(self, file_path: Path) -> bool:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,duration",
            "-of",
            "default=noprint_wrappers=1",
            str(file_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            logging.warning("Evaluation video is not decodable: %s. ffprobe: %s", file_path, stderr)
            return False
        return True

    def _write_fallback_video(self, file_path: Path, frames: list[np.ndarray]) -> bool:
        if not frames:
            logging.warning("Cannot recover evaluation video with fallback: no frames were cached.")
            return False

        tmp_dir = Path(os.environ.get("FASTWAM_VIDEO_FALLBACK_TMPDIR", "/tmp/fastwam_eval_video_fallback"))
        tmp_path = tmp_dir / f"{uuid.uuid4().hex}_{file_path.name}"
        copy_path = file_path.with_name(f"{file_path.stem}.copying{file_path.suffix}")
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            arrays = [np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
            h, w, c = arrays[0].shape
            if c != 3:
                raise ValueError(f"Expected RGB frames with 3 channels, got shape={arrays[0].shape}")
            if any(frame.shape != arrays[0].shape for frame in arrays):
                raise ValueError("Cannot recover evaluation video: cached frames have different shapes.")

            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(self.video_recorder.fps),
                "-i",
                "-",
                "-an",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_path),
            ]
            payload = b"".join(frame.tobytes() for frame in arrays)
            result = subprocess.run(cmd, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                stdout = result.stdout.decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg fallback failed with code {result.returncode}: {stderr or stdout}")
            if not self._probe_video(tmp_path):
                raise RuntimeError(f"ffmpeg fallback produced an undecodable temp video: {tmp_path}")

            shutil.copy2(tmp_path, copy_path)
            if not self._probe_video(copy_path):
                raise RuntimeError(f"copied fallback video is undecodable: {copy_path}")

            os.replace(copy_path, file_path)
            if not self._probe_video(file_path):
                raise RuntimeError(f"final fallback video is undecodable: {file_path}")
            logging.warning("Recovered evaluation video with ffmpeg fallback: %s", file_path)
            return True
        except Exception as exc:
            logging.warning("Failed to recover evaluation video with ffmpeg fallback: %s", exc)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            try:
                if copy_path.exists():
                    copy_path.unlink()
            except OSError:
                pass
            return False
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def reset(self, **kwargs):
        completed_frames = self.frames
        result = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        finalized_ok = self.video_recorder.stop()

        if self.video_dir is not None and self.file_path is not None:
            original_filestem = self.file_path.stem
            fallback_ok = False
            fallback_file_path = self.video_dir / f"{original_filestem}.ffmpeg_recovered.mp4"
            if not finalized_ok:
                fallback_ok = self._write_fallback_video(fallback_file_path, completed_frames)

            if finalized_ok or fallback_ok:
                # Rename the file to indicate success or failure.
                new_filestem = f"{original_filestem}_success{int(self.is_success)}"
            else:
                # A failed MP4 finalization leaves a file without a valid moov atom.
                # Keep it out of success/failure video counts and make it easy to clean up.
                new_filestem = f"{original_filestem}_broken"
            new_file_path = self.video_dir / f"{new_filestem}.mp4"
            if fallback_ok and fallback_file_path.exists():
                os.replace(fallback_file_path, new_file_path)
                if not self._probe_video(new_file_path):
                    broken_file_path = self.video_dir / f"{original_filestem}_broken.mp4"
                    os.replace(new_file_path, broken_file_path)
                    new_file_path = broken_file_path
                try:
                    if self.file_path.exists():
                        self.file_path.unlink()
                except OSError:
                    pass
            elif self.file_path.exists():
                os.rename(self.file_path, new_file_path)

        self.is_success = False
        if self.video_dir is not None:
            self.file_path = self.video_dir / f"{uuid.uuid4()}.mp4"
        return result

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        # Success may happen on a non-render step. Always accumulate it so
        # file suffix reflects episode outcome even when we early-terminate.
        step_success = bool(np.any(np.asarray(result[-1].get("success", False))))
        self.is_success = self.is_success or step_success
        if self.file_path is not None and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_path)

            frame = self.env.render()
            assert frame.dtype == np.uint8
            self.frames.append(frame.copy())
            self.video_recorder.write_frame(frame)
        return result

    def render(self, mode="rgb_array", **kwargs):
        if self.video_recorder.is_ready():
            self.video_recorder.stop()
        return self.file_path
