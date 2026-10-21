from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from prometheus.core.metadata import KIND_INPUT, META_KIND, META_STREAM_NAME
from prometheus.core.sample import StreamSample


def instantiate_optional_node(node_cfg: DictConfig, *, ros_domain_id: Any) -> Any | None:
    """Instantiate one optional hardware publisher from a concrete Hydra node config."""

    data = OmegaConf.to_container(node_cfg, resolve=True)
    assert isinstance(data, dict)
    enabled = bool(data.pop("enabled", False))
    if not enabled:
        return None
    if data.get("ros_domain_id") is None:
        data["ros_domain_id"] = None if ros_domain_id is None else int(ros_domain_id)
    return instantiate(data, _convert_="all")


def robot_state_topic(robot: Any, *, topic_root: str) -> str:
    return f"{topic_root.rstrip('/')}/{robot.robot_id}/state"


def robot_pose_topics(robot: Any, *, topic_root: str) -> tuple[dict[str, Any], ...]:
    """Return robot-provided pose topics when the profile has eef observations."""

    pose_topics = getattr(robot, "pose_topics", None)
    if not callable(pose_topics):
        return ()
    return tuple(pose_topics(topic_root=topic_root))


def required_stream_names(topics: Any) -> tuple[str, ...]:
    """Stream names that must produce data before a workflow is ready.

    `KIND_INPUT` topics such as pedals are event sources, so they are not
    required to emit during startup.
    """

    result: list[str] = []
    for topic in topics:
        metadata = dict(getattr(topic, "metadata", {}))
        if metadata.get(META_KIND) == KIND_INPUT:
            continue
        name = str(metadata.get(META_STREAM_NAME, getattr(topic, "key", "")))
        if not name:
            raise ValueError(f"topic {getattr(topic, 'topic', '')!r} requires a key or stream_name")
        if name in result:
            raise ValueError(f"duplicate required stream name {name!r}")
        result.append(name)
    return tuple(result)


class StreamReadiness:
    """Track whether every configured stream has emitted at least one sample."""

    def __init__(self, required: Any):
        self.required = tuple(str(name) for name in required)
        if len(set(self.required)) != len(self.required):
            raise ValueError("required stream names must be unique")
        self._seen: set[str] = set()
        self._ready = threading.Condition()

    def mark(self, sample: StreamSample) -> None:
        name = str(sample.metadata.get(META_STREAM_NAME, sample.key))
        if name not in self.required:
            return
        with self._ready:
            self._seen.add(name)
            self._ready.notify_all()

    def wait(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))

        def ready() -> bool:
            return set(self.required).issubset(self._seen)

        with self._ready:
            remaining = deadline - time.monotonic()
            if not self._ready.wait_for(ready, timeout=max(0.0, remaining)):
                missing = sorted(set(self.required).difference(self._seen))
                seen = sorted(self._seen)
                raise TimeoutError(f"required streams not ready: missing={missing} seen={seen}")


@contextmanager
def ignore_sigint_during_cleanup() -> Iterator[None]:
    """Cleanup must finish once robot shutdown has started."""

    old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_sigint)
