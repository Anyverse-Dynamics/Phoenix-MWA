"""Versioned cache for expensive ego-video catalog construction."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

_CATALOG_CACHE_FORMAT = "phoenix.encoder_pretrain.catalog.v1"
_CACHE_MODES = {"auto", "off", "require"}
_NON_SEMANTIC_DATA_KEYS = {
    "catalog_cache_dir",
    "catalog_cache_mode",
    "catalog_cache_version",
    "bucket_sampling",
    "clip_sampling_mode",
    "dataset_sampling",
    "dataset_temperature",
    "dataset_weights",
    "drop_last",
    "length_buckets",
    "length_probs",
    "num_workers",
    "per_device_frame_budget",
    "persistent_workers",
    "pin_memory",
    "prefetch_factor",
    "shuffle",
    "shuffle_block_size",
    "stride_buckets",
    "stride_probs",
    "unique_window_span",
    "video_backend_kwargs",
    "worker_restart_interval_batches",
}

CatalogT = TypeVar("CatalogT")


def _semantic_config(data_cfg: Any, only_dataset: str | None, version: str) -> dict[str, Any]:
    config = OmegaConf.to_container(data_cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("ego_video_data must be a mapping")
    config = {key: value for key, value in config.items() if key not in _NON_SEMANTIC_DATA_KEYS}
    return {
        "format": _CATALOG_CACHE_FORMAT,
        "version": version,
        "only_dataset": only_dataset,
        "config": config,
    }


def _fingerprint(data_cfg: Any, only_dataset: str | None, version: str) -> str:
    payload = json.dumps(
        _semantic_config(data_cfg, only_dataset, version),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_path(data_cfg: Any, only_dataset: str | None) -> tuple[Path, str, str]:
    version = str(data_cfg.get("catalog_cache_version", "v1"))
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError("catalog_cache_version may contain only letters, numbers, '.', '_', and '-'")
    fingerprint = _fingerprint(data_cfg, only_dataset, version)
    selection = "all" if only_dataset is None else re.sub(r"[^A-Za-z0-9._-]", "_", str(only_dataset))
    cache_dir = Path(str(data_cfg.catalog_cache_dir)).expanduser()
    return cache_dir / f"{selection}-{version}-{fingerprint[:16]}.pkl", fingerprint, version


def _load_catalog(path: Path, fingerprint: str) -> CatalogT:
    start = time.perf_counter()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != _CATALOG_CACHE_FORMAT:
        raise ValueError(f"Unsupported ego-video catalog cache format in {path}: {payload.get('format')!r}")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(f"Ego-video catalog fingerprint mismatch in {path}")
    logger.info("Loaded ego-video catalog cache %s in %.3fs", path, time.perf_counter() - start)
    return payload["catalog"]


def _write_catalog(path: Path, fingerprint: str, catalog: CatalogT) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(
                {
                    "format": _CATALOG_CACHE_FORMAT,
                    "fingerprint": fingerprint,
                    "catalog": catalog,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_or_build_ego_video_catalog(
    data_cfg: Any,
    only_dataset: str | None,
    builder: Callable[[], CatalogT],
) -> CatalogT:
    """Load a semantic-config cache or build it exactly once across local ranks."""

    cache_dir = data_cfg.get("catalog_cache_dir", None)
    mode = str(data_cfg.get("catalog_cache_mode", "auto" if cache_dir else "off"))
    if mode not in _CACHE_MODES:
        raise ValueError(f"Unsupported catalog_cache_mode={mode!r}; expected one of {sorted(_CACHE_MODES)}")
    if mode == "off":
        return builder()
    if not cache_dir:
        raise ValueError(f"catalog_cache_dir is required when catalog_cache_mode={mode!r}")

    artifact_path, fingerprint, _ = _artifact_path(data_cfg, only_dataset)
    if mode == "require":
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Required ego-video catalog cache does not exist: {artifact_path}")
        return _load_catalog(artifact_path, fingerprint)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_path.with_suffix(f"{artifact_path.suffix}.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if artifact_path.is_file():
            return _load_catalog(artifact_path, fingerprint)
        logger.info("Building ego-video catalog cache %s", artifact_path)
        catalog = builder()
        _write_catalog(artifact_path, fingerprint, catalog)
        logger.info("Wrote ego-video catalog cache %s", artifact_path)
        return catalog
