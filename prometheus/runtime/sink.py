from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any


class AsyncSampleSink:
    """Bounded async boundary for slow consumers.

    `enqueue()` 是给 ROS callback / control-loop 调用的快接口。critical sink
    队列满时 fail loudly；best-effort sink 可配置 drop-old，不能拖慢控制路径。
    """

    def __init__(
        self,
        *,
        name: str,
        consumer: Callable[[Any], None],
        max_queue: int,
        drop_old: bool = False,
        drain_on_close: bool = True,
        close_timeout_s: float | None = 2.0,
    ):
        if int(max_queue) <= 0:
            raise ValueError("AsyncSampleSink max_queue must be positive")
        self.name = str(name)
        self.consumer = consumer
        self.max_queue = int(max_queue)
        self.drop_old = bool(drop_old)
        self.drain_on_close = bool(drain_on_close)
        self.close_timeout_s = None if close_timeout_s is None else float(close_timeout_s)
        if self.close_timeout_s is not None and self.close_timeout_s < 0:
            raise ValueError("AsyncSampleSink close_timeout_s must be non-negative or None")

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self.enqueued_count = 0
        self.consumed_count = 0
        self.dropped_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"AsyncSampleSink:{self.name}", daemon=True)
        self._thread.start()

    def enqueue(self, sample: Any) -> None:
        self.raise_if_error()
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            if not self.drop_old:
                self._set_error(RuntimeError(f"AsyncSampleSink {self.name!r} queue is full"))
                self.raise_if_error()
            self._drop_one_old_sample()
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                with self._lock:
                    self.dropped_count += 1
                return
        with self._lock:
            self.enqueued_count += 1

    def raise_if_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"AsyncSampleSink {self.name!r} failed") from error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.close_timeout_s)
            if self._thread.is_alive():
                self._set_error(TimeoutError(f"AsyncSampleSink {self.name!r} did not stop"))
                self.raise_if_error()
            self._thread = None
        self.raise_if_error()

    def stats(self) -> dict[str, int | str | bool]:
        with self._lock:
            return {
                "name": self.name,
                "queue_len": self._queue.qsize(),
                "max_queue": self.max_queue,
                "drop_old": self.drop_old,
                "enqueued": self.enqueued_count,
                "consumed": self.consumed_count,
                "dropped": self.dropped_count,
            }

    def _run(self) -> None:
        while True:
            if self._stop.is_set() and (not self.drain_on_close or self._queue.empty()):
                return
            try:
                sample = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.consumer(sample)
            except BaseException as exc:
                self._set_error(exc)
                return
            finally:
                self._queue.task_done()
            with self._lock:
                self.consumed_count += 1

    def _drop_one_old_sample(self) -> None:
        try:
            self._queue.get_nowait()
        except queue.Empty:
            return
        self._queue.task_done()
        with self._lock:
            self.dropped_count += 1

    def _set_error(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
        self._stop.set()
