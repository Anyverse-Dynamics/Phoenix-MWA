"""Public weight loaders must reject changed or unknown attention semantics."""

import copy

from omegaconf import OmegaConf
import pytest
import torch

from scripts.preflight_joint96_posttrain import inspect_checkpoint
from policy_fixture import policy
from wam.model.modules.conditioning.frame_attention import JOINT96_ACTION_VIDEO_ATTENTION
from wam.training.checkpoint import (
    _joint96_identity,
    joint96_checkpoint_metadata,
    load_joint96_initialization,
    load_joint96_soft_prompt_initialization,
    load_wam_checkpoint,
    save_wam_checkpoint,
)


ENTRIES = (
    "reload_plain", "reload_prompt", "initialize_plain", "initialize_prompt",
    "prompt_migration",
)
BAD_POLICIES = ("current_only", "unknown_mode", "missing", None, "", 17)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_case(tmp_path, entry):
    source = policy(entry in ("reload_prompt", "initialize_prompt"))
    target = policy(entry in ("reload_prompt", "initialize_prompt", "prompt_migration"))
    path = tmp_path / "parent/checkpoints/weights/step_000005.pt"
    path.parent.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({"local_robot_joint96": {"enabled": True}, "data": {}}),
        path.parents[2] / "config.yaml",
    )
    save_wam_checkpoint(source, path, step=5)
    # Make source and destination observably different before testing rejection.
    with torch.no_grad():
        for module in (target.mot, target.proprio_encoder):
            for parameter in module.parameters():
                parameter.add_(0.25)
    return source, target, path


def snapshot(model):
    return {
        name: {key: tensor.clone() for key, tensor in module.state_dict().items()}
        for name, module in (
            ("mot", model.mot),
            ("proprio_encoder", model.proprio_encoder),
            ("action_proprio_encoder", model.action_proprio_encoder),
        )
        if module is not None
    }


def load(entry, model, path):
    if entry.startswith("reload_"):
        return load_wam_checkpoint(model, path)
    if entry.startswith("initialize_"):
        return load_joint96_initialization(model, path, expected_step=5)
    return load_joint96_soft_prompt_initialization(model, path, expected_step=5)


def alter_policy(path, attention):
    payload = torch.load(path, weights_only=True)
    if attention == "missing":
        payload["joint96_metadata"].pop("action_video_attention")
    else:
        payload["joint96_metadata"]["action_video_attention"] = attention
    torch.save(payload, path)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("attention", BAD_POLICIES)
def test_rejects_attention_before_any_parameter_mutation(tmp_path, entry, attention):
    _, target, path = make_case(tmp_path, entry)
    alter_policy(path, attention)
    before = snapshot(target)
    paths_before = copy.deepcopy(target.model_paths)
    with pytest.raises(ValueError, match="action_video_attention"):
        load(entry, target, path)
    after = snapshot(target)
    for module, tensors in before.items():
        for key, value in tensors.items():
            torch.testing.assert_close(value, after[module][key], atol=0, rtol=0)
    assert target.model_paths == paths_before


@pytest.mark.parametrize("entry", ENTRIES)
def test_same_attention_loads_weights_and_keeps_migration_prompt(tmp_path, entry):
    source, target, path = make_case(tmp_path, entry)
    before = snapshot(target)
    load(entry, target, path)
    after = snapshot(target)
    for module, tensors in snapshot(source).items():
        for key, value in tensors.items():
            torch.testing.assert_close(
                value, after[module][key], atol=0, rtol=0,
            )
    if entry == "prompt_migration":
        prompt_key = "mixtures.action.embodiment_prompt.embedding.weight"
        torch.testing.assert_close(
            before["mot"][prompt_key], target.mot.state_dict()[prompt_key], atol=0, rtol=0,
        )


@pytest.mark.parametrize("attention", BAD_POLICIES)
def test_preflight_rejects_incompatible_or_unspecified_attention(tmp_path, attention):
    _, _, path = make_case(tmp_path, "initialize_plain")
    alter_policy(path, attention)
    with pytest.raises(ValueError, match="action_video_attention"):
        inspect_checkpoint(path, 5)


def test_attention_is_part_of_checkpoint_identity(tmp_path):
    model, _, path = make_case(tmp_path, "reload_plain")
    identity = _joint96_identity(joint96_checkpoint_metadata(model))
    assert identity["action_video_attention"] == JOINT96_ACTION_VIDEO_ATTENTION
    assert inspect_checkpoint(path, 5)["action_video_attention"] == JOINT96_ACTION_VIDEO_ATTENTION
