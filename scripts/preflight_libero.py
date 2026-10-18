#!/usr/bin/env python
"""Validate configured LIBERO initialization and cached-data compatibility."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from preflight_libero_robot37_softprompt import inspect_inputs
from preflight_joint96_posttrain import check_resources, inspect_checkpoint
from wam.datasets.lerobot.transforms.libero_flu_bi import LIBERO_FLU_BI_CONTRACT
from wam.training.config_validation import validate_training_config
from wam.training.soft_prompt import validate_soft_prompt_config
from wam.utils.config_resolvers import register_default_resolvers


def validate_recipe(cfg):
    if cfg.resume is not None or cfg.local_robot_joint96.enabled:
        raise ValueError("Fresh LIBERO post-training only, not full-state resume")
    if not 0 <= cfg.min_learning_rate <= cfg.learning_rate or cfg.learning_rate <= 0:
        raise ValueError("Require positive learning_rate and 0 <= min_learning_rate <= learning_rate")
    if any(type(v) is not int or v <= 0 for v in (cfg.batch_size, cfg.gradient_accumulation_steps)):
        raise ValueError("Batch size and gradient accumulation must be positive integers")
    if not cfg.joint96_posttrain.cached_visuals:
        raise ValueError("Requires visual-cache-only")
    if cfg.joint96_posttrain.action_video_attention != "current_and_noisy_future_v1":
        raise ValueError("Wrong pretrained action/video attention")
    if cfg.joint96_posttrain.libero_frame_contract != LIBERO_FLU_BI_CONTRACT["version"]:
        raise ValueError("Wrong FLU/B=I frame contract")
    for split in ("train", "val"):
        ds = cfg.data[split]
        if ds._target_ != "wam.datasets.libero_joint96_cache.LiberoJoint96CachedDataset":
            raise ValueError("Both splits must bypass image decoding/preprocessing")
        stats = json.loads(Path(ds.pretrained_norm_stats).read_text())
        if stats.get("frame_contract") != json.loads(json.dumps(LIBERO_FLU_BI_CONTRACT)):
            raise ValueError("FLU/B=I normalization contract mismatch")
        if ds.processor.action_state_transforms[0]._target_ != (
                "wam.datasets.lerobot.transforms.libero_flu_bi.LiberoRight80FLUBIdentityTransform"):
            raise ValueError("Wrong action/proprio transform")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", action="store_true")
    parser.add_argument('--report', type=Path)
    args, overrides = parser.parse_known_args()
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "task=libero_posttrain", *overrides])
    result = inspect_inputs(cfg)
    validate_training_config(cfg)
    validate_soft_prompt_config(cfg)
    validate_recipe(cfg)
    if Path(cfg.init_checkpoint).resolve() != Path(result["checkpoint"]).resolve():
        raise ValueError("Inspected checkpoint is not the training checkpoint")
    checkpoint = inspect_checkpoint(cfg.init_checkpoint, cfg.joint96_posttrain.expected_pretrain_step)
    if args.resources:
        result["resources"] = check_resources(cfg)
    result.update(loss_coefficients=OmegaConf.to_container(cfg.model.loss, resolve=True), passed=True, frame_contract=LIBERO_FLU_BI_CONTRACT,
        checkpoint_identity=checkpoint["identity"], tensor_counts=checkpoint["tensors"],
        batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate, min_learning_rate=cfg.min_learning_rate,
        cached_visuals=True, fresh_optimizer=True,
        attention_backend=cfg.model.mot_attention_backend, num_epochs=cfg.num_epochs,
        save_every=cfg.save_every, current_conditioning=checkpoint["identity"].get(
            "current_conditioning", "legacy_clip_prefix"))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
