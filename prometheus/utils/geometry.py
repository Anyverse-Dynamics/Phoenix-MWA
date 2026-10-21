from __future__ import annotations

import math

import numpy as np


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return a unit quaternion in wxyz order."""

    quat = np.asarray(quaternion, dtype=np.float32).reshape(4)
    if not np.all(np.isfinite(quat)):
        raise ValueError("quaternion contains NaN or inf")
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("quaternion norm must be nonzero")
    return (quat / norm).astype(np.float32)


def rpy_to_quaternion_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll/pitch/yaw radians to quaternion in wxyz order."""

    cr = math.cos(float(roll) * 0.5)
    sr = math.sin(float(roll) * 0.5)
    cp = math.cos(float(pitch) * 0.5)
    sp = math.sin(float(pitch) * 0.5)
    cy = math.cos(float(yaw) * 0.5)
    sy = math.sin(float(yaw) * 0.5)
    return normalize_quaternion_wxyz(
        np.asarray(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float32,
        )
    )


def quaternion_wxyz_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to roll/pitch/yaw radians."""

    w, x, y, z = [float(value) for value in normalize_quaternion_wxyz(quaternion)]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(pitch_arg, -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float32)
