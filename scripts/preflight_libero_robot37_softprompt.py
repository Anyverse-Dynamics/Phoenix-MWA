#!/usr/bin/env python
"""CPU-only input identity, prompt wiring, and complete cache preflight."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from wam.embodiment_registry import EmbodimentRegistry
from wam.training.checkpoint import _joint96_identity
from wam.utils.config_resolvers import register_default_resolvers


def inspect_inputs(cfg=None):
    path = Path(cfg.init_checkpoint if cfg is not None else os.environ["WAM_PRETRAIN_CHECKPOINT"])
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    identity = _joint96_identity(payload["joint96_metadata"])
    step = payload.get("step")
    registry = EmbodimentRegistry(cfg.local_robot_softprompt.registry_path if cfg is not None else os.environ.get("EMBODIMENT_REGISTRY", str(ROOT / "configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json")))
    prompt = identity.get("soft_prompt", {})
    key = "mixtures.action.embodiment_prompt.embedding.weight"
    if (prompt.get('num_embodiments') != 28 or prompt.get('num_tokens') != 8 or prompt.get('context_dim') != 1024 or (tuple(payload['mot'][key].shape) != (28, 8192)) or (not torch.isfinite(payload['mot'][key]).all()) or (registry.content['embodiments'][0] != 'franka_panda')):
        raise ValueError("Pretrained Robot37/Panda soft prompt mismatch")
    if identity["action_video_attention"] != "current_and_noisy_future_v1":
        raise ValueError("Expected current/noisy-future attention")
    encoders = (cfg.model.joint_visual_codec.stvae_checkpoint_path,
                cfg.model.joint_visual_codec.dino_kwargs.checkpoint_path) if cfg is not None else (
                os.environ['STVAE_CHECKPOINT'], os.environ['DINO_ENCODER_CHECKPOINT'])
    for path_value in encoders:
        if not Path(path_value).is_file():
            raise FileNotFoundError(path_value)
    return {"checkpoint": str(path), "pretrain_step": step, "stvae_step": identity.get("stvae_step"),
        "embodiment_id": 0, "embodiment_name": "franka_panda", "registry_sha256": registry.sha256,
        "prompt_shape": [28, 8, 1024], "gpu_smoke": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-inputs", action="store_true")
    parser.add_argument("--resources", action="store_true")
    args, overrides = parser.parse_known_args()
    result = inspect_inputs()
    if args.cache_inputs:
        raise ValueError("This release reads existing validated visual caches; extraction is not part of its acceptance")
    else:
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            cfg = compose(config_name="train", overrides=["task=libero_joint96_cached_ab_robot37_softprompt", *overrides])
        from wam.training.config_validation import validate_training_config
        from wam.training.soft_prompt import validate_soft_prompt_config
        validate_training_config(cfg)
        validate_soft_prompt_config(cfg)
        if args.resources:
            from preflight_joint96_posttrain import check_resources
            result["resources"] = check_resources(cfg)
        result.update(batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate, num_epochs=cfg.num_epochs,
            cached_visuals=cfg.joint96_posttrain.cached_visuals)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
