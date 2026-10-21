from __future__ import annotations

import threading
from collections import deque

import numpy as np

from prometheus.core.sample import StreamSample


class ObsStore:
    """Policy-facing latest/history observation cache.

    原始连续录制应直接消费 StreamSample；ObsStore 只提供 policy 所需的
    latest obs 和 policy boundary 后的 next obs。
    """

    def __init__(self, *, history: int = 1):
        if int(history) <= 0:
            raise ValueError("history must be positive")
        self.history = int(history)
        self._lock = threading.RLock()
        self._ready = threading.Condition(self._lock)
        self._buffers: dict[str, deque[StreamSample]] = {}

    def put(self, sample: StreamSample) -> None:
        with self._ready:
            self._buffers.setdefault(sample.key, deque(maxlen=self.history)).append(sample)
            self._ready.notify_all()

    def available(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._buffers))

    def latest(self, keys: tuple[str, ...], *, timeout_s: float | None = None) -> dict[str, np.ndarray]:
        keys = tuple(str(key) for key in keys)
        timeout = None if timeout_s is None else max(0.0, float(timeout_s))

        def ready() -> bool:
            return all(key in self._buffers and self._buffers[key] for key in keys)

        with self._ready:
            if timeout_s is None:
                if not ready():
                    missing = [key for key in keys if key not in self._buffers or not self._buffers[key]]
                    raise KeyError(f"obs keys not ready: {missing}")
            elif not self._ready.wait_for(ready, timeout=timeout):
                missing = [key for key in keys if key not in self._buffers or not self._buffers[key]]
                raise TimeoutError(f"obs keys not ready: {missing}")
            return self._copy_latest_locked(keys)

    def next(self, keys: tuple[str, ...], *, timeout_s: float | None) -> dict[str, np.ndarray]:
        keys = tuple(str(key) for key in keys)
        with self._ready:
            baseline = {key: self._buffers[key][-1] if key in self._buffers and self._buffers[key] else None for key in keys}

            def ready() -> bool:
                return all(
                    key in self._buffers
                    and self._buffers[key]
                    and (baseline[key] is None or self._buffers[key][-1] is not baseline[key])
                    for key in keys
                )

            timeout = None if timeout_s is None else max(0.0, float(timeout_s))
            if not self._ready.wait_for(ready, timeout=timeout):
                stale = [
                    key
                    for key in keys
                    if key not in self._buffers
                    or not self._buffers[key]
                    or self._buffers[key][-1] is baseline[key]
                ]
                raise TimeoutError(f"obs keys did not receive next sample: {stale}")
            return self._copy_latest_locked(keys)

    def _copy_latest_locked(self, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
        return {key: self._buffers[key][-1].value.copy() for key in keys}
