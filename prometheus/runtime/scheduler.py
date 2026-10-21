from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from prometheus.core.action import ActionTrace


class ActionTarget(Protocol):
    action_space: str
    action_dim: int

    def send_action(self, action: np.ndarray) -> None: ...


class QueueRewriter(Protocol):
    """submit 边界的 future queue 重写接口。

    重写只发生在新 chunk 进入队列时，不在 action_hz tick 内做批量变形。
    """

    def __call__(
        self,
        *,
        pending: np.ndarray,
        incoming: np.ndarray,
        last_sent_action: np.ndarray | None,
        sent_action_count: int,
    ) -> np.ndarray: ...


def replace_pending(
    *,
    pending: np.ndarray,
    incoming: np.ndarray,
    last_sent_action: np.ndarray | None,
    sent_action_count: int,
) -> np.ndarray:
    # 默认闭环语义：新的 action chunk 来自最新 obs，直接覆盖旧 future queue。
    return incoming


def downsample_incoming(
    *,
    pending: np.ndarray,
    incoming: np.ndarray,
    last_sent_action: np.ndarray | None,
    sent_action_count: int,
    downsample_factor: int,
) -> np.ndarray:
    # 降采样，适用于 policy 侧训练数据频率高于 action_hz 的情况。
    if int(downsample_factor) <= 0:
        raise ValueError("downsample_factor must be positive")
    downsample_factor = int(downsample_factor)
    if downsample_factor == 1:
        return incoming.copy()
    return incoming[::downsample_factor].copy()


def ema_incoming(
    *,
    pending: np.ndarray,
    incoming: np.ndarray,
    last_sent_action: np.ndarray | None,
    sent_action_count: int,
    alpha: float,
) -> np.ndarray:
    # submit 边界平滑：第一帧从上一条已下发 action 接上，chunk 内部递推。
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if last_sent_action is None:
        return incoming.copy()
    smoothed = np.empty_like(incoming)
    smoothed[0] = alpha * incoming[0] + (1 - alpha) * last_sent_action
    for i in range(1, len(incoming)):
        smoothed[i] = alpha * incoming[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def interpolate_incoming(
    *,
    pending: np.ndarray,
    incoming: np.ndarray,
    last_sent_action: np.ndarray | None,
    sent_action_count: int,
    factor: int,
) -> np.ndarray:
    # 把轨迹按 action_hz 变慢 factor 倍：每个 incoming target 用 factor 个线性插值点到达。
    if int(factor) <= 0:
        raise ValueError("factor must be positive")
    factor = int(factor)
    if factor == 1:
        return incoming.copy()

    previous = incoming[0] if last_sent_action is None else last_sent_action
    interpolated = np.empty((len(incoming) * factor, incoming.shape[1]), dtype=np.float32)
    index = 0
    for target in incoming:
        for step in range(1, factor + 1):
            weight = step / factor
            interpolated[index] = (1.0 - weight) * previous + weight * target
            index += 1
        previous = target
    return interpolated


def parse_action_chunk(raw: Any, *, action_dim: int, max_chunk_horizon: int) -> np.ndarray:
    actions = np.asarray(raw, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != int(action_dim):
        raise ValueError(f"actions must be [T,{int(action_dim)}], got {actions.shape}")
    if not 1 <= len(actions) <= int(max_chunk_horizon):
        raise ValueError(f"chunk horizon must be in [1,{int(max_chunk_horizon)}]")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contains NaN or inf")
    return actions.copy()


class ActionScheduler:
    """唯一 action queue + 唯一 action_hz。

    它不懂 policy transport、不懂 ROS、不懂 data writer。真实控制副作用只发生在
    action_target.send_action(action) 这一点。
    """

    def __init__(
        self,
        *,
        action_target: ActionTarget,
        action_hz: float,
        max_chunk_horizon: int,
        policy_low_watermark: int,
        rewriter: QueueRewriter = replace_pending,
        fixed_chunk_switch: bool = False,
        dry_run: bool = False,
        action_log_interval: int = 0,
        trace_hook: Callable[[ActionTrace], None] | None = None,
        stop_timeout_s: float | None = 2.0,
    ):
        if float(action_hz) <= 0:
            raise ValueError("action_hz must be positive")
        if int(max_chunk_horizon) <= 0:
            raise ValueError("max_chunk_horizon must be positive")
        if not 0 <= int(policy_low_watermark) < int(max_chunk_horizon):
            raise ValueError("policy_low_watermark must be in [0, max_chunk_horizon)")

        self.action_target = action_target
        self.action_hz = float(action_hz)
        self.max_chunk_horizon = int(max_chunk_horizon)
        self.policy_low_watermark = int(policy_low_watermark)
        self.fixed_chunk_switch = bool(fixed_chunk_switch)
        if self.fixed_chunk_switch and rewriter is not replace_pending:
            # Hydra instantiates replace_pending as functools.partial.
            if getattr(rewriter, "func", None) is not replace_pending or getattr(rewriter, "keywords", {}):
                raise ValueError("Fixed chunk switching requires the identity replace_pending rewriter")
        self.rewriter = rewriter
        self.dry_run = bool(dry_run)
        if int(action_log_interval) < 0:
            raise ValueError("action_log_interval must be non-negative")
        self.action_log_interval = int(action_log_interval)
        self.trace_hook = trace_hook
        self.stop_timeout_s = None if stop_timeout_s is None else float(stop_timeout_s)
        if self.stop_timeout_s is not None and self.stop_timeout_s < 0:
            raise ValueError("stop_timeout_s must be non-negative or None")
        self.action_space = str(action_target.action_space)
        self.action_dim = int(action_target.action_dim)

        self._cv = threading.Condition()
        self._queue = np.empty((0, self.action_dim), dtype=np.float32)
        self._sending = False
        self._queue_epoch = 0
        self._last_sent_action: np.ndarray | None = None
        self._sent_action_count = 0
        self._submit_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def sent_action_count(self) -> int:
        with self._cv:
            return self._sent_action_count

    @property
    def pending_count(self) -> int:
        with self._cv:
            return int(len(self._queue))

    def clear_pending(self) -> None:
        # 由 server 侧终止 episode 时调用；清 future queue，让等待 policy boundary 的 step 尽快返回。
        with self._cv:
            self._queue = np.empty((0, self.action_dim), dtype=np.float32)
            self._queue_epoch += 1
            self._cv.notify_all()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ActionScheduler is already started")
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._loop, name="ActionScheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=self.stop_timeout_s)
        if thread.is_alive():
            raise TimeoutError("ActionScheduler did not stop")
        self._thread = None

    def submit(self, actions: Any) -> tuple[int, int]:
        submitted = self._submit(actions)
        return submitted["submit_sent_action_count"], submitted["queue_len_after_submit"]

    def submit_and_wait(self, actions: Any, *, timeout_s: float | None = None) -> dict[str, int]:
        submitted = self._submit(actions, timeout_s=timeout_s)
        sent_action_count = self.wait_until_policy_boundary(timeout_s=timeout_s)
        result = {
            "submit_sent_action_count": submitted["submit_sent_action_count"],
            "sent_action_count": sent_action_count,
            "queue_len_after_submit": submitted["queue_len_after_submit"],
            "pending_count": self.pending_count,
        }
        self._log_action_step(submitted, result)
        return result

    def _submit(self, actions: Any, *, timeout_s: float | None = None) -> dict[str, int]:
        incoming = parse_action_chunk(
            actions,
            action_dim=self.action_dim,
            max_chunk_horizon=self.max_chunk_horizon,
        )
        if len(incoming) <= self.policy_low_watermark:
            raise ValueError("chunk horizon must be greater than policy_low_watermark")

        with self._cv:
            if self.fixed_chunk_switch:
                # Stage an early reply here. Never replace any unexecuted old action.
                # Include the in-flight final send, which has already left _queue.
                queue_epoch = self._queue_epoch
                ready = self._cv.wait_for(
                    lambda: self._stop.is_set() or self._error is not None
                    or self._queue_epoch != queue_epoch
                    or (len(self._queue) == 0 and not self._sending),
                    timeout=timeout_s,
                )
                if not ready:
                    raise TimeoutError("timed out waiting for fixed chunk boundary")
                if self._stop.is_set():
                    raise RuntimeError("ActionScheduler stopped while waiting for chunk boundary")
                self._raise_if_error_locked()
                if self._queue_epoch != queue_epoch:
                    # An episode stop/prepare_terminal invalidates the staged reply.
                    return {
                        "submit_count": self._submit_count,
                        "incoming_horizon": int(len(incoming)),
                        "scheduled_horizon": 0,
                        "submit_sent_action_count": self._sent_action_count,
                        "queue_len_after_submit": 0,
                    }
            self._raise_if_error_locked()
            rewritten = self.rewriter(
                pending=self._queue.copy(),
                incoming=incoming.copy(),
                last_sent_action=None if self._last_sent_action is None else self._last_sent_action.copy(),
                sent_action_count=self._sent_action_count,
            )
            queue = parse_action_chunk(
                rewritten,
                action_dim=self.action_dim,
                max_chunk_horizon=self.max_chunk_horizon,
            )
            if len(queue) <= self.policy_low_watermark:
                raise ValueError("rewriter output horizon must be greater than policy_low_watermark")
            submit_sent_action_count = self._sent_action_count
            self._submit_count += 1
            submit_count = self._submit_count
            self._queue = queue
            self._cv.notify_all()
            return {
                "submit_count": submit_count,
                "incoming_horizon": int(len(incoming)),
                "scheduled_horizon": int(len(queue)),
                "submit_sent_action_count": submit_sent_action_count,
                "queue_len_after_submit": int(len(queue)),
            }

    def wait_until_policy_boundary(self, *, timeout_s: float | None = None) -> int:
        """Wait until queue length reaches the policy boundary.

        `policy_low_watermark=0` 是同步 chunk 执行完再推理；大于 0 是提前续 chunk。
        """

        def ready() -> bool:
            return self._error is not None or len(self._queue) <= self.policy_low_watermark

        with self._cv:
            ok = self._cv.wait_for(ready, timeout=None if timeout_s is None else max(0.0, float(timeout_s)))
            if not ok:
                raise TimeoutError("timed out waiting for policy boundary")
            self._raise_if_error_locked()
            return self._sent_action_count

    def _loop(self) -> None:
        period_ns = int(1_000_000_000 / self.action_hz)
        next_ns = time.perf_counter_ns()
        while True:
            delay_s = (next_ns - time.perf_counter_ns()) / 1e9
            if delay_s > 0 and self._stop.wait(delay_s):
                return

            with self._cv:
                self._cv.wait_for(lambda: self._stop.is_set() or len(self._queue) > 0)
                if self._stop.is_set():
                    return
                if next_ns < time.perf_counter_ns():
                    # queue 为空时没有 tick 债务；新 chunk 的第一条 action 作为新的匀速起点。
                    next_ns = time.perf_counter_ns()
                action = self._queue[0].copy()
                self._queue = self._queue[1:]
                self._sending = True

            scheduler_stamp_ns = time.perf_counter_ns()
            scheduler_wall_time_ns = time.time_ns()
            try:
                send_start_ns = time.perf_counter_ns()
                if not self.dry_run:
                    self.action_target.send_action(action)
                send_done_ns = time.perf_counter_ns()
            except BaseException as exc:
                with self._cv:
                    self._error = exc
                    self._sending = False
                    self._cv.notify_all()
                return

            with self._cv:
                self._sending = False
                self._last_sent_action = action.copy()
                self._sent_action_count += 1
                sent_count = self._sent_action_count
                pending_count = int(len(self._queue))
                self._cv.notify_all()

            if self.trace_hook is not None:
                try:
                    self.trace_hook(
                        ActionTrace(
                            sent_action_count=sent_count,
                            action_space=self.action_space,
                            action=action,
                            scheduler_stamp_ns=scheduler_stamp_ns,
                            send_start_ns=send_start_ns,
                            send_done_ns=send_done_ns,
                            queue_len_after_send=pending_count,
                            metadata={"wall_time_ns": scheduler_wall_time_ns},
                        )
                    )
                except BaseException as exc:
                    with self._cv:
                        self._error = exc
                        self._cv.notify_all()
                    return

            now_ns = time.perf_counter_ns()
            next_ns += period_ns
            if next_ns < now_ns:
                # send_action() 已经拖慢本 tick 时，下一条应立即接上；否则会变成 send 耗时 + period。
                next_ns = now_ns

    def _raise_if_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("ActionScheduler failed") from self._error

    def _log_action_step(self, submitted: dict[str, int], result: dict[str, int]) -> None:
        if not self.dry_run or self.action_log_interval == 0:
            return
        if submitted["submit_count"] % self.action_log_interval != 0:
            return
        sent_from_submit = result["sent_action_count"] - result["submit_sent_action_count"]
        print(
            "[ActionScheduler][dry_run] "
            f"submit={submitted['submit_count']} "
            f"incoming_horizon={submitted['incoming_horizon']} "
            f"scheduled_horizon={submitted['scheduled_horizon']} "
            f"sent={result['sent_action_count']} "
            f"sent_from_submit={sent_from_submit} "
            f"queue_len_after_submit={result['queue_len_after_submit']} "
            f"pending={result['pending_count']}",
            flush=True,
        )
