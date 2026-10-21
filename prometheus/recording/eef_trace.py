from __future__ import annotations

import csv
import html
import json
import math
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prometheus.core.action import ActionTrace
from prometheus.core.metadata import KIND_ROBOT_EEF, META_KIND, META_PART
from prometheus.core.sample import StreamSample
from prometheus.recording.profile import next_episode_id


@dataclass(slots=True)
class EefTraceCapture:
    """Debug trace for one EEF stream plus scheduler chunk boundaries."""

    root: Path
    episode_prefix: str = "eef_trace"
    start_index: int = 1
    part: str = "right"
    write_csv: bool = True
    write_png: bool = True
    png_dpi: int = 180
    write_svg: bool = False
    episode_id: str = ""
    episode_dir: Path | None = None
    _samples: list[dict[str, Any]] = field(default_factory=list)
    _actions: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _closed: bool = False

    def start(self) -> str:
        if self.episode_dir is not None:
            raise RuntimeError("EefTraceCapture is already started")
        self.root = Path(self.root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode_id = next_episode_id(self.root, prefix=self.episode_prefix, start_index=self.start_index)
        self.episode_dir = self.root / self.episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self._closed = False
        return self.episode_id

    def record_sample(self, sample: StreamSample) -> None:
        if self.episode_dir is None or self._closed or not self._is_target_sample(sample):
            return
        pose = np.asarray(sample.value, dtype=np.float32).reshape(-1)
        if pose.size != 7:
            raise ValueError(f"{self.part} eef pose must be 7D [x,y,z,qw,qx,qy,qz], got {pose.shape}")
        payload = {
            "stamp_ns": int(sample.stamp_ns),
            "recv_ns": int(sample.recv_ns),
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
            "qw": float(pose[3]),
            "qx": float(pose[4]),
            "qy": float(pose[5]),
            "qz": float(pose[6]),
        }
        with self._lock:
            self._samples.append(payload)

    def record_action_trace(self, trace: ActionTrace) -> None:
        if self.episode_dir is None or self._closed:
            return
        action = np.asarray(trace.action, dtype=np.float32).reshape(-1)
        payload = {
            "sent_action_count": int(trace.sent_action_count),
            "action_space": str(trace.action_space),
            "scheduler_stamp_ns": int(trace.scheduler_stamp_ns),
            "send_start_ns": int(trace.send_start_ns),
            "send_done_ns": int(trace.send_done_ns),
            "wall_time_ns": _optional_int(trace.metadata.get("wall_time_ns")),
            "queue_len_after_send": int(trace.queue_len_after_send),
            "action": action.astype(np.float32).copy(),
        }
        with self._lock:
            self._actions.append(payload)

    def close(self, *, accept: bool = True, discard: bool = False) -> None:
        if self.episode_dir is None or self._closed:
            return
        if accept and discard:
            raise ValueError("cannot accept and discard the same EEF trace")
        if discard:
            root = Path(self.root).expanduser().resolve()
            target = self.episode_dir.resolve()
            if target.parent != root or target.name != self.episode_id:
                raise RuntimeError(f"refusing to discard unexpected EEF trace directory: {target}")
            if target.exists():
                shutil.rmtree(target)
            self._closed = True
            return
        with self._lock:
            samples = [dict(item) for item in self._samples]
            actions = [dict(item) for item in self._actions]
        annotated_actions = _annotate_chunks(actions)
        metadata = {
            "episode_id": self.episode_id,
            "part": self.part,
            "accepted": bool(accept),
            "sample_count": len(samples),
            "action_trace_count": len(actions),
            "files": {
                "eef_csv": f"{self.part}_eef_trace.csv" if self.write_csv else "",
                "action_csv": "action_trace.csv" if self.write_csv else "",
                "png": f"{self.part}_eef_xyz.png" if self.write_png and samples else "",
                "svg": f"{self.part}_eef_xyz.svg" if self.write_svg else "",
            },
            "notes": "action wall_time_ns is used to align chunk boundaries with robot EEF stamps",
        }
        assert self.episode_dir is not None
        (self.episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if self.write_csv:
            _write_eef_csv(self.episode_dir / f"{self.part}_eef_trace.csv", samples)
            _write_action_csv(self.episode_dir / "action_trace.csv", annotated_actions)
        if self.write_png and samples:
            render_eef_xyz_png(
                samples,
                annotated_actions,
                self.episode_dir / f"{self.part}_eef_xyz.png",
                title=f"{self.episode_id} {self.part} EEF",
                dpi=int(self.png_dpi),
            )
        if self.write_svg and samples:
            svg = render_eef_xyz_svg(samples, annotated_actions, title=f"{self.episode_id} {self.part} EEF")
            (self.episode_dir / f"{self.part}_eef_xyz.svg").write_text(svg)
        self._closed = True

    def _is_target_sample(self, sample: StreamSample) -> bool:
        metadata = dict(sample.metadata)
        if sample.key == f"robot/{self.part}_eef_pose":
            return True
        return metadata.get(META_KIND) == KIND_ROBOT_EEF and str(metadata.get(META_PART, "")) == self.part


def render_eef_xyz_png(
    samples: list[Mapping[str, Any]],
    actions: list[Mapping[str, Any]],
    path: Path,
    *,
    title: str,
    dpi: int = 180,
) -> None:
    if not samples:
        raise ValueError("cannot render EEF PNG without samples")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/prometheus_matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    start_ns = int(samples[0]["stamp_ns"])
    times = np.asarray([(int(item["stamp_ns"]) - start_ns) / 1e9 for item in samples], dtype=np.float64)
    series_by_axis = {
        "x": np.asarray([float(item["x"]) for item in samples], dtype=np.float64),
        "y": np.asarray([float(item["y"]) for item in samples], dtype=np.float64),
        "z": np.asarray([float(item["z"]) for item in samples], dtype=np.float64),
    }
    chunk_times = [
        (int(action["wall_time_ns"]) - start_ns) / 1e9
        for action in actions
        if bool(action.get("chunk_start")) and action.get("wall_time_ns") is not None
    ]
    colors = {"x": "#2563eb", "y": "#059669", "z": "#dc2626"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight="bold")
    for axis, key in zip(axes, ("x", "y", "z")):
        values = series_by_axis[key]
        axis.plot(times, values, color=colors[key], linewidth=2.4, label=f"{key} position")
        for chunk_t in chunk_times:
            axis.axvline(chunk_t, color="#f97316", linestyle="--", linewidth=1.2, alpha=0.75)
        axis.set_ylabel(f"{key} (m)", fontsize=12)
        axis.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.7)
        axis.legend(loc="upper right", fontsize=10)
        finite = values[np.isfinite(values)]
        if finite.size:
            y_min = float(np.nanmin(finite))
            y_max = float(np.nanmax(finite))
            pad = max(1e-4, (y_max - y_min) * 0.12)
            axis.set_ylim(y_min - pad, y_max + pad)
    axes[-1].set_xlabel("time (s); orange dashed lines = inferred chunk starts", fontsize=12)
    fig.savefig(path, dpi=max(72, int(dpi)), facecolor="white")
    plt.close(fig)


def render_eef_xyz_svg(samples: list[Mapping[str, Any]], actions: list[Mapping[str, Any]], *, title: str) -> str:
    if not samples:
        raise ValueError("cannot render EEF SVG without samples")
    start_ns = int(samples[0]["stamp_ns"])
    times = np.asarray([(int(item["stamp_ns"]) - start_ns) / 1e9 for item in samples], dtype=np.float64)
    values = {
        "x": np.asarray([float(item["x"]) for item in samples], dtype=np.float64),
        "y": np.asarray([float(item["y"]) for item in samples], dtype=np.float64),
        "z": np.asarray([float(item["z"]) for item in samples], dtype=np.float64),
    }
    chunk_times = [
        (int(action["wall_time_ns"]) - start_ns) / 1e9
        for action in actions
        if bool(action.get("chunk_start")) and action.get("wall_time_ns") is not None
    ]
    width, panel_h = 1100, 210
    margin_l, margin_r, margin_t, margin_b = 74, 26, 42, 36
    gap = 28
    height = margin_t + 3 * panel_h + 2 * gap + margin_b
    t_min = float(np.nanmin(times))
    t_max = float(np.nanmax(times))
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        t_min, t_max = 0.0, max(1.0, t_min + 1.0 if math.isfinite(t_min) else 1.0)

    def sx(t: float) -> float:
        return margin_l + (float(t) - t_min) / (t_max - t_min) * (width - margin_l - margin_r)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:monospace;font-size:13px;fill:#1f2937}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.chunk{stroke:#f97316;stroke-width:1;stroke-dasharray:4 4}.curve{fill:none;stroke-width:2}</style>",
        f'<text x="{margin_l}" y="24">{html.escape(title)}</text>',
    ]
    colors = {"x": "#2563eb", "y": "#059669", "z": "#dc2626"}
    for idx, key in enumerate(("x", "y", "z")):
        top = margin_t + idx * (panel_h + gap)
        bottom = top + panel_h
        series = values[key]
        y_min = float(np.nanmin(series))
        y_max = float(np.nanmax(series))
        if not math.isfinite(y_min) or not math.isfinite(y_max):
            y_min, y_max = 0.0, 1.0
        if y_max <= y_min:
            pad = max(1e-4, abs(y_min) * 0.01)
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.08
            y_min -= pad
            y_max += pad

        def sy(v: float) -> float:
            return bottom - (float(v) - y_min) / (y_max - y_min) * panel_h

        parts.append(f'<line class="axis" x1="{margin_l}" y1="{bottom:.1f}" x2="{width - margin_r}" y2="{bottom:.1f}"/>')
        parts.append(f'<line class="axis" x1="{margin_l}" y1="{top:.1f}" x2="{margin_l}" y2="{bottom:.1f}"/>')
        for fraction in (0.25, 0.5, 0.75):
            gy = top + panel_h * fraction
            parts.append(f'<line class="grid" x1="{margin_l}" y1="{gy:.1f}" x2="{width - margin_r}" y2="{gy:.1f}"/>')
        for chunk_t in chunk_times:
            if t_min <= chunk_t <= t_max:
                x = sx(chunk_t)
                parts.append(f'<line class="chunk" x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}"/>')
        points = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(times, series))
        parts.append(f'<polyline class="curve" stroke="{colors[key]}" points="{points}"/>')
        parts.append(f'<text x="18" y="{top + 18:.1f}">{key} m</text>')
        parts.append(f'<text x="{margin_l}" y="{top + 16:.1f}">{y_max:.4f}</text>')
        parts.append(f'<text x="{margin_l}" y="{bottom - 6:.1f}">{y_min:.4f}</text>')
    parts.append(f'<text x="{margin_l}" y="{height - 12}">time {t_min:.2f}s to {t_max:.2f}s; orange dashed lines = inferred chunk starts</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def _write_eef_csv(path: Path, samples: list[Mapping[str, Any]]) -> None:
    start_ns = int(samples[0]["stamp_ns"]) if samples else 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "stamp_ns", "t_s", "recv_ns", "x", "y", "z", "qw", "qx", "qy", "qz"])
        for index, item in enumerate(samples):
            writer.writerow(
                [
                    index,
                    int(item["stamp_ns"]),
                    _rel_s(item["stamp_ns"], start_ns),
                    int(item["recv_ns"]),
                    item["x"],
                    item["y"],
                    item["z"],
                    item["qw"],
                    item["qx"],
                    item["qy"],
                    item["qz"],
                ]
            )


def _write_action_csv(path: Path, actions: list[Mapping[str, Any]]) -> None:
    first_wall = next((int(item["wall_time_ns"]) for item in actions if item.get("wall_time_ns") is not None), 0)
    max_dim = max((len(np.asarray(item["action"]).reshape(-1)) for item in actions), default=0)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trace_index",
                "sent_action_count",
                "chunk_index",
                "chunk_phase",
                "chunk_start",
                "wall_time_ns",
                "t_s",
                "queue_len_after_send",
                "scheduler_stamp_ns",
                "send_start_ns",
                "send_done_ns",
                *[f"action_{index}" for index in range(max_dim)],
            ]
        )
        for index, item in enumerate(actions):
            action = np.asarray(item["action"], dtype=np.float32).reshape(-1)
            wall_time_ns = item.get("wall_time_ns")
            writer.writerow(
                [
                    index,
                    int(item["sent_action_count"]),
                    int(item["chunk_index"]),
                    int(item["chunk_phase"]),
                    int(bool(item["chunk_start"])),
                    "" if wall_time_ns is None else int(wall_time_ns),
                    "" if wall_time_ns is None or first_wall == 0 else _rel_s(int(wall_time_ns), first_wall),
                    int(item["queue_len_after_send"]),
                    int(item["scheduler_stamp_ns"]),
                    int(item["send_start_ns"]),
                    int(item["send_done_ns"]),
                    *[float(value) for value in action],
                ]
            )


def _annotate_chunks(actions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    chunk_index = -1
    chunk_phase = 0
    previous_pending: int | None = None
    for item in actions:
        pending = int(item["queue_len_after_send"])
        chunk_start = previous_pending is None or pending > previous_pending
        if chunk_start:
            chunk_index += 1
            chunk_phase = 0
        else:
            chunk_phase += 1
        payload = dict(item)
        payload["chunk_index"] = chunk_index
        payload["chunk_phase"] = chunk_phase
        payload["chunk_start"] = chunk_start
        result.append(payload)
        previous_pending = pending
    return result


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _rel_s(stamp_ns: Any, start_ns: int) -> float:
    return (int(stamp_ns) - int(start_ns)) / 1e9
