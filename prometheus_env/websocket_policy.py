from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np


def _pack_array(value: Any) -> Any:
    """Encode NumPy values with the msgpack extension used by Phoenix."""

    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: dict[Any, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def _msgpack_module() -> Any:
    try:
        import msgpack
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Phoenix transport requires msgpack; install Prometheus with "
            "`pip install -e '.[client]'`."
        ) from exc
    return msgpack


def packb(value: Any) -> bytes:
    return _msgpack_module().packb(value, default=_pack_array)


def unpackb(value: bytes) -> Any:
    return _msgpack_module().unpackb(value, object_hook=_unpack_array)


def normalize_websocket_addr(addr: str) -> str:
    value = str(addr).strip()
    if not value:
        raise ValueError("policy addr must not be empty")
    if "://" not in value:
        value = f"ws://{value}"
    if not value.startswith(("ws://", "wss://")):
        raise ValueError(f"policy addr must use ws:// or wss://, got {addr!r}")
    return value


class WebsocketPolicyClient:
    """Synchronous client for the Phoenix msgpack WebSocket transport."""

    def __init__(
        self,
        addr: str,
        *,
        connect_timeout_s: float = 600.0,
        retry_interval_s: float = 2.0,
        metadata_timeout_s: float = 30.0,
        inference_timeout_s: float = 60.0,
    ) -> None:
        self.addr = normalize_websocket_addr(addr)
        self.connect_timeout_s = float(connect_timeout_s)
        self.retry_interval_s = float(retry_interval_s)
        self.metadata_timeout_s = float(metadata_timeout_s)
        self.inference_timeout_s = float(inference_timeout_s)
        if self.connect_timeout_s < 0:
            raise ValueError("connect_timeout_s must be non-negative")
        if self.retry_interval_s <= 0:
            raise ValueError("retry_interval_s must be positive")
        if self.metadata_timeout_s < 0:
            raise ValueError("metadata_timeout_s must be non-negative")
        if self.inference_timeout_s < 0:
            raise ValueError("inference_timeout_s must be non-negative")
        self.last_timing: dict[str, float | int] = {}
        self._socket = self._connect()
        try:
            metadata_payload = self._socket.recv(
                timeout=None if self.metadata_timeout_s == 0 else self.metadata_timeout_s
            )
        except TimeoutError as exc:
            self.close()
            raise TimeoutError(
                f"policy metadata at {self.addr} exceeded "
                f"{self.metadata_timeout_s:.1f}s"
            ) from exc
        if isinstance(metadata_payload, str):
            self.close()
            raise RuntimeError(f"policy server rejected connection:\n{metadata_payload}")
        try:
            metadata = unpackb(metadata_payload)
        except BaseException:
            self.close()
            raise
        if not isinstance(metadata, Mapping):
            self.close()
            raise TypeError(
                f"policy metadata must be a mapping, got {type(metadata).__name__}"
            )
        self.metadata = dict(metadata)

    def _connect(self) -> Any:
        try:
            from websockets.sync.client import connect
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Phoenix transport requires websockets; install Prometheus with "
                "`pip install -e '.[client]'`."
            ) from exc

        started = time.monotonic()
        deadline = (
            None
            if self.connect_timeout_s == 0
            else started + self.connect_timeout_s
        )
        last_error: BaseException | None = None
        while deadline is None or time.monotonic() < deadline:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            open_timeout = 10.0 if remaining is None else max(0.1, min(10.0, remaining))
            try:
                return connect(
                    self.addr,
                    compression=None,
                    max_size=None,
                    open_timeout=open_timeout,
                    ping_interval=None,
                    ping_timeout=None,
                )
            except (OSError, TimeoutError) as exc:
                last_error = exc
                if deadline is not None and time.monotonic() >= deadline:
                    break
                delay = self.retry_interval_s
                if deadline is not None:
                    delay = min(delay, max(0.0, deadline - time.monotonic()))
                if delay > 0:
                    time.sleep(delay)
        elapsed = time.monotonic() - started
        raise TimeoutError(
            f"could not connect to policy server {self.addr} within {elapsed:.1f}s"
        ) from last_error

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        pack_start = time.perf_counter()
        payload = packb(dict(request))
        after_pack = time.perf_counter()
        self._socket.send(payload)
        after_send = time.perf_counter()
        try:
            response = self._socket.recv(
                timeout=(
                    None
                    if self.inference_timeout_s == 0
                    else self.inference_timeout_s
                )
            )
        except TimeoutError as exc:
            self.close()
            raise TimeoutError(
                f"policy inference at {self.addr} exceeded "
                f"{self.inference_timeout_s:.1f}s"
            ) from exc
        after_recv = time.perf_counter()
        if isinstance(response, str):
            raise RuntimeError(f"error in policy server:\n{response}")
        try:
            result = unpackb(response)
        except BaseException:
            self.close()
            raise
        after_unpack = time.perf_counter()
        self.last_timing = {
            "client_pack_s": after_pack - pack_start,
            "ws_send_s": after_send - after_pack,
            "ws_recv_s": after_recv - after_send,
            "client_unpack_s": after_unpack - after_recv,
            "request_bytes": len(payload),
            "response_bytes": len(response),
        }
        if not isinstance(result, Mapping):
            raise TypeError(
                f"policy response must be a mapping, got {type(result).__name__}"
            )
        return dict(result)

    def close(self) -> None:
        socket = getattr(self, "_socket", None)
        if socket is not None:
            socket.close()
            self._socket = None

    def __enter__(self) -> "WebsocketPolicyClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
