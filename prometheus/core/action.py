from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from prometheus.core.sample import freeze_mapping


@dataclass(frozen=True, slots=True)
class ActionTrace:
    """ActionScheduler 已经发生过的控制事实。

    它可以给下游 logging/UI 使用，但不能被当作真实控制命令重新下发。
    """

    sent_action_count: int
    action_space: str
    action: np.ndarray
    scheduler_stamp_ns: int
    send_start_ns: int
    send_done_ns: int
    queue_len_after_send: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=np.float32).copy()
        action.setflags(write=False)
        object.__setattr__(self, "sent_action_count", int(self.sent_action_count))
        object.__setattr__(self, "action_space", str(self.action_space))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "scheduler_stamp_ns", int(self.scheduler_stamp_ns))
        object.__setattr__(self, "send_start_ns", int(self.send_start_ns))
        object.__setattr__(self, "send_done_ns", int(self.send_done_ns))
        object.__setattr__(self, "queue_len_after_send", int(self.queue_len_after_send))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @property
    def send_latency_ms(self) -> float:
        return max(0.0, (self.send_done_ns - self.send_start_ns) / 1_000_000.0)
