from __future__ import annotations

import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_JOINT_PLOT_RANGES = (
    (-3.14, 2.618),
    (-0.05, 3.50),
    (-0.10, 3.20),
    (-1.60, 1.55),
    (-1.57, 1.57),
    (-2.00, 2.00),
    (-0.005, 0.09),
)
_PLOT_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class PointcloudDisplacementStats:
    """Per-frame signed xyz displacement statistics over corresponding points."""

    time_s: np.ndarray
    mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    valid_counts: np.ndarray
    point_count: int


def pointcloud_xyz_displacement_stats(
    frames: Sequence[Any],
    timestamps_ms: Sequence[int | float],
) -> PointcloudDisplacementStats:
    """Compare Mesh3D xyz in every frame with the sensor's earliest frame.

    Xense pointcloud rows are ``[x, y, z, dx, dy, dz]``. This diagnostic
    deliberately uses columns 0:3 because it visualizes the recorded xyz
    geometry requested by the user, not the SDK-provided flow columns.
    Correspondence is by the fixed Xense grid point index.
    """

    frame_list = [np.asarray(frame) for frame in frames]
    if not frame_list:
        raise ValueError("pointcloud stream has no frames")
    timestamps = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    if timestamps.size != len(frame_list):
        raise ValueError(
            "pointcloud frame/timestamp length mismatch: "
            f"frames={len(frame_list)} timestamps={timestamps.size}"
        )
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("pointcloud timestamps contain NaN or inf")

    order = np.argsort(timestamps, kind="stable")
    ordered_frames = [frame_list[int(index)] for index in order]
    ordered_timestamps = timestamps[order]
    reference = _pointcloud_xyz(ordered_frames[0], frame_index=int(order[0]))
    point_count = int(reference.shape[0])
    if point_count <= 0:
        raise ValueError("pointcloud frames must contain at least one point")

    shape = (len(ordered_frames), 3)
    mean = np.full(shape, np.nan, dtype=np.float64)
    minimum = np.full(shape, np.nan, dtype=np.float64)
    maximum = np.full(shape, np.nan, dtype=np.float64)
    valid_counts = np.zeros(shape, dtype=np.int64)

    for frame_index, frame in enumerate(ordered_frames):
        xyz = _pointcloud_xyz(frame, frame_index=int(order[frame_index]))
        if xyz.shape != reference.shape:
            raise ValueError(
                "pointcloud shape changed within episode: "
                f"reference={reference.shape} frame[{int(order[frame_index])}]={xyz.shape}"
            )
        delta = xyz - reference
        valid = np.isfinite(xyz) & np.isfinite(reference)
        for axis in range(3):
            values = delta[valid[:, axis], axis]
            valid_counts[frame_index, axis] = int(values.size)
            if values.size:
                mean[frame_index, axis] = float(np.mean(values))
                minimum[frame_index, axis] = float(np.min(values))
                maximum[frame_index, axis] = float(np.max(values))

    return PointcloudDisplacementStats(
        time_s=(ordered_timestamps - ordered_timestamps[0]) / 1000.0,
        mean=mean,
        minimum=minimum,
        maximum=maximum,
        valid_counts=valid_counts,
        point_count=point_count,
    )


def write_episode_visualizations(
    episode_dir: str | Path,
    *,
    robot_state: Mapping[str, Any],
    tactile_pointcloud: Mapping[str, Any],
    tactile_timestamps: Mapping[str, Sequence[int | float]],
    dpi: int = 220,
) -> dict[str, Any]:
    """Render best-effort episode diagnostics without affecting raw data."""

    with _PLOT_LOCK:
        return _write_episode_visualizations_unlocked(
            episode_dir,
            robot_state=robot_state,
            tactile_pointcloud=tactile_pointcloud,
            tactile_timestamps=tactile_timestamps,
            dpi=dpi,
        )


def _write_episode_visualizations_unlocked(
    episode_dir: str | Path,
    *,
    robot_state: Mapping[str, Any],
    tactile_pointcloud: Mapping[str, Any],
    tactile_timestamps: Mapping[str, Sequence[int | float]],
    dpi: int,
) -> dict[str, Any]:

    output_dir = Path(episode_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()
    outputs: list[str] = []
    errors: dict[str, str] = {}
    pointcloud_metadata: dict[str, Any] = {
        "reference": "earliest frame per sensor",
        "columns": ["x", "y", "z"],
        "statistics": ["minimum", "mean", "maximum"],
        "signed": True,
        "sensors": {},
    }

    def attempt(name: str, render: Any) -> Path | None:
        try:
            path = Path(render())
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            return None
        outputs.append(path.name)
        return path

    if _has_joint_data(robot_state):
        attempt(
            "robot_joint",
            lambda: plot_joint_state(
                output_dir,
                robot_state,
                plt,
                dpi=dpi,
                episode_name=output_dir.name,
            ),
        )
    if _has_eef_xyz_data(robot_state):
        attempt(
            "robot_eef_xyz",
            lambda: plot_eef_xyz(
                output_dir,
                robot_state,
                plt,
                dpi=dpi,
                episode_name=output_dir.name,
            ),
        )

    used_names: set[str] = set()
    for sensor_name, stream in sorted(tactile_pointcloud.items()):
        frames = list(dict(stream).get("data", [])) if isinstance(stream, Mapping) else []
        if not frames:
            continue
        component = f"pointcloud/{sensor_name}"
        safe_name = _unique_safe_name(str(sensor_name), used_names)
        try:
            stats = pointcloud_xyz_displacement_stats(
                frames,
                tactile_timestamps.get(sensor_name, ()),
            )
        except Exception as exc:
            errors[component] = f"{type(exc).__name__}: {exc}"
            continue
        invalid_frame_axes = int(np.count_nonzero(stats.valid_counts == 0))
        if invalid_frame_axes:
            errors[f"{component}/valid_points"] = (
                f"{invalid_frame_axes} frame-axis entries have zero valid corresponding points"
            )
        filename = f"{safe_name}_pointcloud_xyz_displacement_vis.png"
        path = attempt(
            component,
            lambda sensor=str(sensor_name), values=stats, name=filename: plot_pointcloud_xyz_displacement(
                output_dir / name,
                values,
                plt,
                sensor_name=sensor,
                episode_name=output_dir.name,
                dpi=dpi,
            ),
        )
        pointcloud_metadata["sensors"][str(sensor_name)] = {
            "frames": int(stats.time_s.size),
            "points": int(stats.point_count),
            "output": None if path is None else path.name,
            "minimum_valid_points": int(np.min(stats.valid_counts)),
        }

    status = "complete"
    if errors:
        status = "partial" if outputs else "error"
    elif not outputs:
        status = "empty"
    return {
        "status": status,
        "outputs": outputs,
        "errors": errors,
        "pointcloud_displacement": pointcloud_metadata,
    }


def plot_joint_state(
    output_dir: Path,
    robot_state: Mapping[str, Any],
    plt: Any,
    *,
    dpi: int,
    episode_name: str,
) -> Path:
    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for part_name, part_stream in robot_state.items():
        if not isinstance(part_stream, Mapping) or not part_stream.get("joint"):
            continue
        joint, time_s = series_to_arrays(part_stream, "joint")
        series.append((str(part_name), joint, time_s))
    if not series:
        raise ValueError("robot_state has no joint data")

    axis_count = sum(values.shape[1] for _, values, _ in series)
    fig_height = max(8.0, 1.25 * axis_count)
    fig, axes = plt.subplots(axis_count, 1, figsize=(18, fig_height), sharex=True)
    if axis_count == 1:
        axes = [axes]
    axis_index = 0
    sample_counts = []
    try:
        for part_name, values, time_s in series:
            sample_counts.append(values.shape[0])
            part_dim = values.shape[1]
            for joint_index in range(part_dim):
                axis = axes[axis_index]
                axis.plot(time_s, values[:, joint_index], color="#1f77b4", linewidth=0.85)
                if part_dim == 7:
                    axis.set_ylim(*DEFAULT_JOINT_PLOT_RANGES[joint_index])
                label = f"{part_name}_joint_{joint_index + 1}"
                if part_dim == 7 and joint_index == 6:
                    label = f"{part_name}_gripper"
                axis.set_ylabel(label, fontsize=8)
                axis.grid(True, alpha=0.35, linewidth=0.5)
                axis.tick_params(labelsize=8)
                axis_index += 1
        axes[-1].set_xlabel("time (s)", fontsize=10)
        fig.suptitle(
            f"Prometheus joint state - {episode_name}\n"
            f"parts={len(series)} samples={min(sample_counts)} dimensions={axis_count}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        output = output_dir / "state_joint_vis.png"
        _save_png(fig, output, dpi=dpi)
    finally:
        plt.close(fig)
    return output


def plot_eef_xyz(
    output_dir: Path,
    robot_state: Mapping[str, Any],
    plt: Any,
    *,
    dpi: int,
    episode_name: str,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    labels = ("x", "y", "z")
    palette = ("#d62728", "#1f77b4", "#059669", "#7c3aed")
    plotted = 0
    try:
        for part_index, (part_name, part_stream) in enumerate(robot_state.items()):
            if not isinstance(part_stream, Mapping) or not part_stream.get("eef"):
                continue
            pose, time_s = series_to_arrays(part_stream, "eef")
            if pose.shape[1] < 3 or not np.isfinite(pose[:, :3]).any():
                continue
            plotted += 1
            for axis_index, axis in enumerate(axes):
                axis.plot(
                    time_s,
                    pose[:, axis_index],
                    color=palette[part_index % len(palette)],
                    linewidth=0.9,
                    label=str(part_name),
                )
        if not plotted:
            raise ValueError("robot_state has no EEF xyz data")
        for axis_index, axis in enumerate(axes):
            axis.set_ylabel(f"eef {labels[axis_index]} (m)", fontsize=10)
            axis.grid(True, alpha=0.35, linewidth=0.5)
            axis.tick_params(labelsize=9)
            if axis_index == 0:
                axis.legend(loc="upper right")
        axes[-1].set_xlabel("time (s)", fontsize=10)
        fig.suptitle(f"Prometheus EEF xyz - {episode_name}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output = output_dir / "state_eef_xyz_vis.png"
        _save_png(fig, output, dpi=dpi)
    finally:
        plt.close(fig)
    return output


def plot_pointcloud_xyz_displacement(
    output: str | Path,
    stats: PointcloudDisplacementStats,
    plt: Any,
    *,
    sensor_name: str,
    episode_name: str,
    dpi: int,
) -> Path:
    output = Path(output)
    fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
    axis_names = ("x", "y", "z")
    try:
        for axis_index, axis in enumerate(axes):
            minimum = stats.minimum[:, axis_index]
            mean = stats.mean[:, axis_index]
            maximum = stats.maximum[:, axis_index]
            axis.fill_between(stats.time_s, minimum, maximum, color="#93c5fd", alpha=0.28)
            axis.plot(stats.time_s, minimum, color="#dc2626", linewidth=0.9, label="minimum")
            axis.plot(stats.time_s, mean, color="#1d4ed8", linewidth=1.6, label="mean")
            axis.plot(stats.time_s, maximum, color="#059669", linewidth=0.9, label="maximum")
            axis.axhline(0.0, color="#6b7280", linewidth=0.7, alpha=0.7)
            axis.set_ylabel(f"Δ{axis_names[axis_index]}\n(point-cloud units)", fontsize=10)
            axis.grid(True, alpha=0.35, linewidth=0.5)
            axis.tick_params(labelsize=9)
            if axis_index == 0:
                axis.legend(loc="upper right", ncol=3)
        axes[-1].set_xlabel("time from initial frame (s)", fontsize=10)
        fig.suptitle(
            f"Pointcloud xyz displacement - {episode_name} - {sensor_name}\n"
            f"current xyz - initial xyz; points={stats.point_count}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save_png(fig, output, dpi=dpi)
    finally:
        plt.close(fig)
    return output


def series_to_arrays(
    series: Mapping[str, Any],
    value_key: str,
    *,
    fallback_hz: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    values = list(series.get(value_key, []))
    if not values:
        raise ValueError(f"series has no {value_key} data")
    valid_rows = [np.asarray(row, dtype=np.float64).reshape(-1) for row in values if row is not None]
    if not valid_rows:
        raise ValueError(f"series has no finite-shape {value_key} data")
    dimension = int(valid_rows[0].size)
    if dimension <= 0:
        raise ValueError(f"series {value_key} rows must not be empty")
    rows = []
    for value in values:
        if value is None:
            rows.append(np.full(dimension, np.nan, dtype=np.float64))
            continue
        row = np.asarray(value, dtype=np.float64).reshape(-1)
        if row.size != dimension:
            raise ValueError(f"{value_key} row has dimension {row.size}, expected {dimension}")
        rows.append(row)
    data = np.stack(rows, axis=0)
    timestamps = np.asarray(series.get("timestamps", []), dtype=np.float64).reshape(-1)
    if timestamps.size == data.shape[0] and np.all(np.isfinite(timestamps)):
        order = np.argsort(timestamps, kind="stable")
        data = data[order]
        timestamps = timestamps[order]
        time_s = (timestamps - timestamps[0]) / 1000.0
    else:
        time_s = np.arange(data.shape[0], dtype=np.float64) / max(1.0, float(fallback_hz))
    return data, time_s


def _pointcloud_xyz(frame: np.ndarray, *, frame_index: int) -> np.ndarray:
    if frame.ndim != 2 or frame.shape[1] < 3:
        raise ValueError(
            f"pointcloud frame[{frame_index}] must have shape (N, >=3), got {tuple(frame.shape)}"
        )
    return np.asarray(frame[:, :3], dtype=np.float64)


def _load_pyplot() -> Any:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/prometheus_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/prometheus_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_png(fig: Any, output: Path, *, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        fig.savefig(
            temporary,
            format="png",
            dpi=max(72, int(dpi)),
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _has_joint_data(robot_state: Mapping[str, Any]) -> bool:
    return any(isinstance(stream, Mapping) and bool(stream.get("joint")) for stream in robot_state.values())


def _has_eef_xyz_data(robot_state: Mapping[str, Any]) -> bool:
    for stream in robot_state.values():
        if not isinstance(stream, Mapping):
            continue
        for value in stream.get("eef", []):
            if value is None:
                continue
            pose = np.asarray(value).reshape(-1)
            if pose.size >= 3 and np.isfinite(pose[:3]).any():
                return True
    return False


def _unique_safe_name(value: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sensor"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate
