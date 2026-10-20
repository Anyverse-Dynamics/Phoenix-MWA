"""Opt-in wiring for the local Robot80 soft-prompt experiment."""
from omegaconf import OmegaConf

from wam.embodiment_registry import EmbodimentRegistry


def validate_soft_prompt_config(cfg):
    select = lambda name, default=None: OmegaConf.select(cfg, name, default=default)
    enabled = select("local_robot_softprompt.enabled", False)
    count = select("model.action_dit_config.num_embodiments", 0)
    if not enabled:
        if count:
            raise ValueError("Use the explicit local_robot_softprompt recipe to enable prompts")
        return None
    libero_posttrain = select("joint96_posttrain.libero_softprompt.enabled", False)
    robotwin_posttrain = select("joint96_posttrain.robotwin_softprompt.enabled", False)
    robodojo_real_posttrain = select("joint96_posttrain.robodojo_real_softprompt.enabled", False)
    if sum(bool(x) for x in (libero_posttrain, robotwin_posttrain, robodojo_real_posttrain)) > 1:
        raise ValueError("Choose exactly one posttraining prompt domain")
    posttrain = libero_posttrain or robotwin_posttrain or robodojo_real_posttrain
    if posttrain:
        raise ValueError("Benchmark post-training is excluded from this release")
    if (not select("local_robot_joint96.enabled", False) and not posttrain) or not select("model.joint96_three_view", False):
        raise ValueError("Soft-prompt integration currently supports local Robot Joint96 pretraining")
    if select("model.understanding.enabled", False):
        raise ValueError("Soft prompts currently require T5, not VLM conditioning")
    if type(count) is not int or count <= 0:
        raise ValueError("Enabled soft prompts require a positive integer num_embodiments")
    tokens = select("model.action_dit_config.soft_prompt_tokens")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("soft_prompt_tokens must be a positive integer")
    path = select("local_robot_softprompt.registry_path")
    if not path or path != select("model.action_dit_config.embodiment_registry_path"):
        raise ValueError("Reader and model must share one explicit prompt registry")
    registry = EmbodimentRegistry(path)
    if registry.num_embodiments != count:
        raise ValueError("Model/registry embodiment count mismatch")
    init = select("local_robot_softprompt.init_checkpoint")
    if init:
        if select("resume") is not None or select("init_checkpoint") is not None:
            raise ValueError("Prompt weights-only initialization cannot be combined with resume/init_checkpoint")
    return registry


def wrap_soft_prompt_datasets(model, train_dataset, val_dataset, cfg):
    registry = validate_soft_prompt_config(cfg)
    prompt = getattr(getattr(model, "action_expert", None), "embodiment_prompt", None)
    if registry is None:
        if prompt is not None:
            raise ValueError("Prompt model requires an explicitly enabled training recipe")
        return train_dataset, val_dataset
    if prompt is None or prompt.registry is None or prompt.registry.sha256 != registry.sha256:
        raise ValueError("Model/reader prompt registry mismatch")
    if train_dataset is val_dataset or val_dataset is None:
        raise ValueError("Prompt training requires separate train and validation datasets")
    from wam.datasets.soft_prompt import SoftPromptDataset
    wrapped = tuple(SoftPromptDataset(ds, registry.path) for ds in (train_dataset, val_dataset))
    model.soft_prompt_training_contract = {
        "frame_ablation": cfg.data.get("frame_ablation", None),
        "package_sha256": train_dataset.package.fingerprint,
    }
    return wrapped
