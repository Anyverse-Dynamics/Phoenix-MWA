from __future__ import annotations

from typing import Any

import numpy as np


class ZmqEnvClient:
    """第三方 policy 侧最小 SDK。

    SDK 只暴露 reset/observe/step/close；policy 不需要知道 Prometheus 内部组件。
    所有请求/响应保持 dict，避免把 transport tuple 顺序变成隐式协议。
    """

    def __init__(self, socket: Any):
        self.socket = socket
        self._request_count = 0

    def reset(
        self,
        *,
        expect: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        obs_keys: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_options = dict(options or {})
        if expect is not None:
            if "expect" in reset_options:
                raise ValueError("reset() received both expect= and options['expect']")
            reset_options["expect"] = dict(expect)
        reply = self._request({
            "type": "reset",
            "options": reset_options,
            "obs_keys": list(obs_keys),
        })
        return reply["obs"], reply.get("info", {})

    def observe(
        self,
        *,
        obs_keys: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the latest observation without starting control or submitting actions."""
        reply = self._request({
            "type": "observe",
            "obs_keys": list(obs_keys),
        })
        return reply["obs"], dict(reply.get("info", {}))

    def step(
        self,
        actions: np.ndarray,
        *,
        obs_keys: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        reply = self._request({
            "type": "step",
            "actions": np.asarray(actions, dtype=np.float32),
            "obs_keys": list(obs_keys),
        })
        return (
            reply["obs"],
            float(reply.get("reward", 0.0)),
            bool(reply.get("terminated", False)),
            bool(reply.get("truncated", False)),
            dict(reply.get("info", {})),
        )

    def prepare_terminal(self, *, reason: str) -> dict[str, Any]:
        reply = self._request({
            "type": "prepare_terminal",
            "reason": str(reason),
        })
        return dict(reply.get("info", {}))

    def terminal(
        self,
        *,
        accept: bool,
        reason: str,
        outcome: str | None = None,
        discard: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "terminal",
            "accept": bool(accept),
            "reason": str(reason),
        }
        if outcome is not None:
            payload["outcome"] = str(outcome)
        if discard:
            payload["discard"] = True
        reply = self._request(payload)
        return dict(reply.get("info", {}))

    def close_session(self) -> None:
        self._request({"type": "close"})

    def close(self) -> None:
        try:
            self.close_session()
        finally:
            self.socket.close(linger=0)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._request_count += 1
        request_id = f"client:{self._request_count}"
        self.socket.send_pyobj({"request_id": request_id, **payload})
        reply = self.socket.recv_pyobj()
        if not isinstance(reply, dict):
            raise TypeError(f"ZMQ env reply must be dict, got {type(reply).__name__}")
        if reply.get("request_id") != request_id:
            raise RuntimeError(f"ZMQ env reply request_id mismatch: expected {request_id}, got {reply.get('request_id')}")
        if not reply.get("ok", False):
            raise RuntimeError(str(reply.get("message", "request failed")))
        return reply


def connect(addr: str) -> ZmqEnvClient:
    import zmq

    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect(str(addr))
    return ZmqEnvClient(socket)
