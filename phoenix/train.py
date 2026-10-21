"""Phoenix Training RTC command; config preflight never constructs the model."""
import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=()):
    default = OmegaConf.load(ROOT / "configs/phoenix_training_rtc.yaml")
    supplied = OmegaConf.load(path)
    config = OmegaConf.merge(default, supplied, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return OmegaConf.to_container(config, resolve=True)


def check_config(config):
    from .trainer import file_sha256
    from .training import training_contract
    from .model import Normalizer
    if config["schema"] != "phoenix.training_rtc_recipe.v1":
        raise ValueError("Expected Phoenix training config")
    sampling, evaluation, rtc = config["sampling"], config["evaluation"], config["training_time_rtc"]
    if sampling != dict(action_horizon=32, raw_window_frames=33, inference_video_frames=9, action_hz=30):
        raise ValueError("Phoenix training requires H32 / raw33 / video9 / 30Hz")
    if evaluation != dict(action_horizon=32, video_frames=9, denoising_steps=10):
        raise ValueError("Phoenix evaluation requires H32 / video9 / denoise10")
    if (rtc["enabled"] is not True or rtc["method"] != "condition" or rtc["start_step"] != 0
            or rtc["prefix_timestep"] != 0 or rtc["prefix_in_loss"] is not False
            or rtc["prefix_sampling"] not in {"uniform_inclusive", "zero_or_uniform_inclusive"}):
        raise ValueError("Unsupported Phoenix RTC semantics")
    if rtc["prefix_sampling"] == "uniform_inclusive" and rtc["no_prefix_probability"] != 0:
        raise ValueError("Uniform prefix sampling requires no_prefix_probability=0")
    contract = training_contract(rtc["max_prefix_steps"], rtc["no_prefix_probability"])
    init, data, options = config["initialization"], config["data"], config["training"]
    if init["mode"] != "weights_only" or init["resume_optimizer"] is not False:
        raise ValueError("Use resume_state for optimizer resume, initialization is weights-only")
    for field in ("batch_size", "gradient_accumulation_steps", "max_steps", "save_every"):
        if type(options[field]) is not int or options[field] <= 0:
            raise ValueError(f"Invalid training.{field}")
    if type(options["num_workers"]) is not int or options["num_workers"] < 0:
        raise ValueError("Invalid worker count")
    if options["mixed_precision"] not in ("no", "bf16"):
        raise ValueError("Phoenix supports FP32 or BF16 training")
    if not 0 <= options["min_learning_rate"] <= options["learning_rate"] or options["learning_rate"] <= 0:
        raise ValueError("Invalid learning rates")
    if config["validation"]["samples"] <= 0 or config["validation"]["every"] < 0:
        raise ValueError("Invalid validation settings")
    assets = {"initialization": init["checkpoint"], "normalization": data["normalization"],
              "manifest": data["manifest"], "text_index": data["text_cache_index"],
              "model_definition": config["model_definition"]}
    if data.get("visual_cache_index"):
        if options["mixed_precision"] != "bf16":
            raise ValueError("The frozen visual cache uses BF16; select training.mixed_precision=bf16")
        assets["visual_cache_index"] = data["visual_cache_index"]
    fingerprints = {name: file_sha256(path) for name, path in assets.items()}
    if fingerprints["initialization"] != init["sha256"] or fingerprints["normalization"] != data["normalization_sha256"]:
        raise ValueError("Initialization or normalization fingerprint differs")
    Normalizer(json.loads(Path(data["normalization"]).read_text()), data["gripper_unit"])
    definition = OmegaConf.load(config["model_definition"])
    model = definition.model if "model" in definition else definition
    model = OmegaConf.to_container(model, resolve=True, throw_on_missing=True)
    if (model.get("joint96_three_view") is not True or model.get("proprio_dim") != 80
            or model["action_dit_config"]["action_dim"] != 80
            or model.get("load_text_encoder") is not False):
        raise ValueError("Expected Phoenix Joint96 canonical80 model with cached T5")
    for field in ("future_delta", "understanding", "memory", "distillation"):
        if model.get(field, {}).get("enabled", False):
            raise ValueError(f"Unsupported Phoenix training branch: {field}")
    configured_rtc = model["action_dit_config"]["training_time_rtc"]
    if (configured_rtc.get("enabled") is not True
            or configured_rtc.get("max_delay_steps") != rtc["max_prefix_steps"]):
        raise ValueError("Model and training RTC settings differ")
    # Resolve the external frozen encoder files as part of the pinned run.
    codec = model["joint_visual_codec"]
    fingerprints["encoder"] = file_sha256(codec["dino_kwargs"]["checkpoint_path"])
    fingerprints["stvae"] = file_sha256(codec["stvae_checkpoint_path"])
    from .dataset import PhoenixDataset
    train = PhoenixDataset(data["manifest"], data["normalization"], data["text_cache_index"],
                           split="train", embodiment_id=data["embodiment_id"], gripper_unit=data["gripper_unit"],
                           visual_cache_index=data.get("visual_cache_index"))
    val = None
    if config["validation"]["every"]:
        val = PhoenixDataset(data["manifest"], data["normalization"], data["text_cache_index"],
                             split="val", embodiment_id=data["embodiment_id"], gripper_unit=data["gripper_unit"],
                             visual_cache_index=data.get("visual_cache_index"))
    contract = dict(schema="phoenix.training_run.v1", assets=fingerprints, rtc=contract,
                    embodiment_id=data["embodiment_id"], gripper_unit=data["gripper_unit"],
                    model_definition=model, train_windows=len(train), val_windows=len(val) if val else 0)
    return contract, train, val


def build_model(config, device):
    import torch
    from hydra.utils import instantiate
    from wam.utils.config_resolvers import register_default_resolvers
    from wam.utils.torch24_compat import install_safe_serialization_compat
    from wam.model.modules.codecs.dino_stvae.provenance import bundled_config
    from .training import initialize_from_pretrained
    from .checkpoint import _joint96_identity
    register_default_resolvers()
    install_safe_serialization_compat()
    payload = torch.load(config["initialization"]["checkpoint"], map_location="cpu", mmap=True, weights_only=True)
    expected_step = config["initialization"]["expected_step"]
    if expected_step is not None and payload.get("step") != expected_step:
        raise ValueError("Initialization checkpoint step differs")
    metadata = copy.deepcopy(payload["joint96_metadata"])
    if metadata["schema"] == "internw0.local_joint96.weights.v1":
        metadata["schema"] = "phoenix.local_joint96.weights.v1"
    _joint96_identity(metadata)
    registry = metadata["soft_prompt"]["registry"]
    identity = config["data"]["embodiment_id"]
    if (type(identity) is not int or not 0 <= identity < len(registry["embodiments"])
            or registry.get("slot_status", ["assigned"] * len(registry["embodiments"]))[identity] != "assigned"):
        raise ValueError("Training requires an assigned inherited embodiment prompt")
    definition = OmegaConf.load(config["model_definition"])
    model_cfg = definition.model if "model" in definition else definition
    model_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
    model_cfg.joint_visual_codec = bundled_config(OmegaConf.to_container(model_cfg.joint_visual_codec, resolve=True))
    dtype = torch.bfloat16 if config["training"]["mixed_precision"] == "bf16" else torch.float32
    # Registry is inherited from the pinned initializer, never recreated from a
    # data name. Frozen encoder/VAE paths remain external.
    with tempfile.TemporaryDirectory(prefix="phoenix-training-") as directory:
        path = Path(directory) / "phoenix_embodiments.json"
        path.write_text(json.dumps(registry))
        model_cfg.action_dit_config.embodiment_registry_path = str(path)
        # Joint96's public factory requires skip_dit_load_from_pretrain=false;
        # only the underlying component loader skips redundant policy weights.
        import wam.model.factory as factory
        original = factory.load_wan22_ti2v_5b_components
        def checkpoint_components(**kwargs):
            kwargs["skip_dit_load_from_pretrain"] = True
            return original(**kwargs)
        factory.load_wan22_ti2v_5b_components = checkpoint_components
        try:
            model = instantiate(model_cfg, model_dtype=dtype, device="cpu")
        finally:
            factory.load_wan22_ti2v_5b_components = original
        initialize_from_pretrained(model, payload)
    del payload
    model.to(device)
    model.set_runtime_device(device)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/phoenix_training_rtc.yaml"))
    parser.add_argument("--check-config", action="store_true", help="Hash assets and check contracts; no model or training")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    contract, train_dataset, val_dataset = check_config(config)
    if args.check_config:
        print(json.dumps(dict(status="CONFIG_READY", model="Phoenix", contract=contract,
                              training_started=False, model_constructed=False), indent=2))
        return 0
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
    plugin = None
    if config["distributed"]["deepspeed_zero1"]:
        options = config["training"]
        plugin = DeepSpeedPlugin(hf_ds_config={
            "zero_optimization": {"stage": 1},
            "train_micro_batch_size_per_gpu": options["batch_size"],
            "gradient_accumulation_steps": options["gradient_accumulation_steps"],
            "gradient_clipping": options["max_grad_norm"],
            "bf16": {"enabled": options["mixed_precision"] == "bf16"},
            "steps_per_print": 1000000,
        })
    handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]
    if config["distributed"]["cpu"]:
        handlers.append(InitProcessGroupKwargs(backend="gloo"))
    accelerator = Accelerator(gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"], cpu=config["distributed"]["cpu"],
        deepspeed_plugin=plugin,
        kwargs_handlers=handlers)
    set_seed(config["training"]["seed"])
    model = build_model(config, accelerator.device)
    if train_dataset.visual_cache is not None:
        train_dataset.visual_cache.validate_model(model, config["model_definition"])
    from .trainer import PhoenixTrainer, atomic_json
    trainer = PhoenixTrainer(model, train_dataset, val_dataset, config, accelerator, input_contract=contract)
    if config["resume_state"]:
        trainer.restore(config["resume_state"])
    elif any((Path(config["output_dir"]) / "checkpoints").glob("step_*")):
        raise FileExistsError("Existing training checkpoints require an explicit resume_state or a new output_dir")
    if accelerator.is_main_process:
        atomic_json(Path(config["output_dir"]) / "resolved_config.json", config)
        atomic_json(Path(config["output_dir"]) / "training_contract.json", trainer.contract)
        # Preserve bytes, not just values: inference binds this exact SHA256.
        destination = Path(config["output_dir"]) / "normalization.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(Path(config["data"]["normalization"]).read_bytes())
        temporary.replace(destination)
    accelerator.wait_for_everyone()
    trainer.train()
    accelerator.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
