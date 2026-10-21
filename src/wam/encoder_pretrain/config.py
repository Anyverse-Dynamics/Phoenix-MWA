"""Configuration and read-only preflight; no model allocation or training."""
from pathlib import Path
import json

from omegaconf import OmegaConf

DINO_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def bundled_dino_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party/dino_runtime/dinov3"


def verify_bundled_dino(expected: str) -> str:
    observed = (bundled_dino_root() / ".source-commit").read_text().strip()
    if expected != DINO_COMMIT or observed != expected:
        raise RuntimeError(f"Bundled DINO commit mismatch: expected {expected}, got {observed}")
    return observed


def validate_config(cfg):
    """Resolve all paths before construction and reject unsupported recipes."""
    OmegaConf.resolve(cfg)
    model = cfg.framework.multi_teacher_distillation
    if model.mode not in {"joint", "vggt_only", "vjepa_only"}:
        raise ValueError("Training mode must be joint, vggt_only, or vjepa_only")
    adapter = model.adapter
    if int(adapter.width) != 1024 or int(adapter.depth) != 8 or list(adapter.supervision_taps) != [1, 3, 5, 7]:
        raise ValueError("This recipe requires the 1024-wide, 8-block adapter with taps [1,3,5,7]")
    if int(adapter.num_heads) != 16 or int(adapter.num_register_tokens) != 16:
        raise ValueError("This recipe requires 16 attention heads and 16 register tokens")
    if not 0 <= float(model.ema_momentum) < 1:
        raise ValueError("ema_momentum must be in [0,1)")
    trainer = cfg.trainer
    for key in ("max_train_steps", "gradient_accumulation_steps", "logging_frequency", "save_interval"):
        if int(trainer[key]) <= 0:
            raise ValueError(f"trainer.{key} must be positive")
    if trainer.precision not in {"no", "bf16", "fp16"}:
        raise ValueError("precision must be no, bf16, or fp16")
    if trainer.first_teacher not in {"vggt", "vjepa"}:
        raise ValueError("first_teacher must be vggt or vjepa")
    data = cfg.datasets.ego_video_data
    if data.video_view_mode != "all" or data.dataset_sampling != "stratified_view_temperature":
        raise ValueError("This recipe requires all-view stratified temperature sampling")
    if not set(data.length_buckets).issubset({4, 8, 12, 16}) or not data.length_buckets:
        raise ValueError("length_buckets must select from 4, 8, 12, 16")
    if not set(data.stride_buckets).issubset({1, 2}) or not data.stride_buckets:
        raise ValueError("stride_buckets must select from 1, 2")
    if int(data.per_device_frame_budget) < max(data.length_buckets):
        raise ValueError("Frame budget must fit the longest clip")
    if not data.datasets:
        raise ValueError("Configure at least one LeRobot V3 dataset")
    if trainer.get("resume_from") and not Path(trainer.resume_from).is_file():
        raise FileNotFoundError(trainer.resume_from)
    verify_bundled_dino(str(model.dino.source_commit))
    return cfg


def preflight(cfg) -> dict:
    validate_config(cfg)
    from wam.encoder.external_sources import (
        SourceContract, verify_source_contract,
    )
    from .data.clips import _entry_dataset_paths
    model = cfg.framework.multi_teacher_distillation
    sources = {"dino": {"commit": verify_bundled_dino(str(model.dino.source_commit))}}
    for name, package in (("vggt", "vggt_omega"), ("vjepa", "src")):
        if model.mode not in {"joint", name + "_only"}:
            continue
        section = model[name]
        sources[name] = verify_source_contract(SourceContract(
            str(section.source_path), package, str(section.source_commit),
            clean_paths=(package, "app") if name == "vjepa" else (package,),
        ))
    weights = [model.dino.checkpoint]
    weights += [model[name].checkpoint for name in ("vggt", "vjepa")
                if model.mode in {"joint", name + "_only"}]
    for path in weights:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    streams = []
    data = cfg.datasets.ego_video_data
    for entry in data.datasets:
        for path in _entry_dataset_paths(entry, Path(data.data_root_dir)):
            info = json.loads((path / "meta/info.json").read_text())
            keys = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
            if entry.video_key not in keys:
                raise ValueError(f"Primary video key {entry.video_key} is absent from {path}")
            streams.append({"dataset": str(entry.name), "path": str(path), "video_keys": keys})
    if not streams:
        raise ValueError("Dataset routes did not resolve to any physical datasets")
    return {"status": "preflight_passed", "sources": sources, "datasets": streams,
            "note": "Paths, source commits and video metadata only; weights and video decoding are not tested."}
