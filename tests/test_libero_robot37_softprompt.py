import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from wam.embodiment_registry import EmbodimentRegistry
from wam.training.libero_robot37_softprompt import LiberoPromptDataset, validate_libero_prompt
from wam.model.modules.conditioning.embodiment_prompt import EmbodimentPrompt


class Base(Dataset):
    sample_episode_ranges = ((0, 2),)
    fixed_validation_panel = [{"index": 0}]

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return {"sample_id": i, "action": torch.ones(32, 80) * i}

    def collate_fn(self, samples):
        # Exercise native collators which do not retain arbitrary extra keys.
        return {"action": torch.stack([x["action"] for x in samples])}


def registry():
    return EmbodimentRegistry("configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json")


def test_fixed_panda_does_not_modify_sampling_or_base_samples():
    b = Base()
    wrapped = LiberoPromptDataset(b, registry())
    assert len(wrapped) == len(b)
    assert wrapped.sample_episode_ranges == b.sample_episode_ranges
    assert "embodiment_id" not in b[0]
    assert wrapped[0]["embodiment_id"].dtype == torch.long
    assert int(wrapped[0]["embodiment_id"]) == 0
    batch = wrapped.collate_fn([wrapped[0], wrapped[1]])
    assert batch["embodiment_id"].tolist() == [0, 0]
    assert batch["action"].shape == (2, 32, 80)
    assert wrapped.fixed_validation_panel[0]["embodiment_id"] == 0
    assert "embodiment_id" not in b.fixed_validation_panel[0]


def test_prompt_mask_and_reserved_slot():
    p = EmbodimentPrompt(28, 8, 1024, registry().path)
    context = torch.randn(2, 128, 1024)
    mask = torch.zeros(2, 128, dtype=torch.bool)
    out, visible = p(context, mask, torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 136, 1024)
    assert torch.equal(out[:, 8:], context)
    assert visible[:, :8].all() and not visible[:, 8:].any()
    out[:, :8].sum().backward()
    assert p.embedding.weight.grad[0].abs().sum() > 0
    assert not p.embedding.weight.grad[1:].any()
    with pytest.raises(ValueError):
        p(context, mask, torch.full((2,), 18, dtype=torch.long))


def test_native_prompt_cfg_remains_opt_in():
    cfg = OmegaConf.create({"joint96_posttrain": {"enabled": True, "libero_softprompt": {"embodiment_id": 0}},
        "local_robot_joint96": {"enabled": False}, "local_robot_softprompt": {"init_checkpoint": None},
        "resume": None, "data": {"train": {"domain": "libero"}, "val": {"domain": "libero"}},
        "model": {"action_dit_config": {"num_embodiments": 28, "soft_prompt_tokens": 8}}})
    validate_libero_prompt(cfg)
    cfg.joint96_posttrain.libero_softprompt.embodiment_id = 1
    with pytest.raises(ValueError):
        validate_libero_prompt(cfg)


def test_bad_pair_binding_rejected_before_features(tmp_path):
    from wam.datasets.libero_stvae_rebind import load_bound_pair
    old = tmp_path / "old"
    old.mkdir()
    (old / "contract.json").write_text(json.dumps({"reference_stvae_root": str(tmp_path / "missing")}))
    def reject(*args):
        raise ValueError("invalid original provenance")
    with pytest.raises(ValueError, match="original provenance"):
        load_bound_pair(old, tmp_path / "new", tmp_path / "binding.json", reject)


def test_pair_binding_stable_on_revalidation_and_rejects_changed_snapshot(tmp_path):
    from wam.datasets.libero_stvae_rebind import binding_content, digest, load_bound_pair
    vae, new, old = (tmp_path / n for n in ("vae", "new", "old"))
    vae.mkdir()
    snapshot = {"dataset": "same-rgb-sources"}
    window = {"state_frames": 33, "action_horizon": 32, "rgb_offsets": list(range(0, 33, 4)),
              "latent_frames": 3, "pad_policy": "repeat"}
    encoder = {"stvae_step": 150000, "teacher_identity_verified": True, "teacher_sha256": "dino110k",
               "teacher_checkpoint_state": "ema_student", "teacher_output_source": "backbone",
               "teacher_preprocessing": {"size": 256}, "dinov3_source_commit": "pinned"}
    vr = {"reference_stvae_root": str(old), "dataset_length": 277713,
          "dataset_snapshot_sha256": digest(snapshot), "window": window}
    (vae / "contract.json").write_text(json.dumps(vr))
    for view in ("head", "wrist"):
        (vae / view).mkdir()
        (vae / view / "contract.json").write_text(json.dumps({**vr, "view": view}))
    contracts = []
    for view in ("head", "wrist"):
        c = {"view": view, "selection": "full_dataset", "dataset_length": 277713,
             "dataset_snapshot": snapshot, "dataset_snapshot_sha256": digest(snapshot),
             "posterior": "mean", "extra_latent_scaling": "none", "cache_dtype": "bfloat16",
             "encoder": encoder, "window": window, "world_size": 32}
        (new / view).mkdir(parents=True)
        (new / view / "contract.json").write_text(json.dumps(c))
        contracts.append(c)
    report = {"status": "passed", "scope": "full_dataset", "feature_hashes_verified": True,
              "samples_per_view": 277713, "world_size": 32, "stvae_step": 150000,
              "views": {v: {"contract_sha256": digest(c)} for v,c in zip(("head", "wrist"), contracts)}}
    def pair(*args):
        return [], [vr, vr, {"encoder": encoder}, {"encoder": encoder}]
    for attempt in ("attempt01", "attempt02"):
        d = new / ".launch" / attempt
        d.mkdir(parents=True)
        (d / "VALIDATED.json").write_text(json.dumps(report))
        actual = binding_content(vae, new, pair)
        if attempt == "attempt01":
            initial = actual
            (new / "binding.json").write_text(json.dumps(actual))
        else:
            assert actual == initial
    roots, loaded = load_bound_pair(vae, new, new / "binding.json", pair)
    assert len(roots) == len(loaded) == 4
    changed = contracts[0].copy()
    changed["dataset_snapshot_sha256"] = "foreign"
    (new / "head/contract.json").write_text(json.dumps(changed))
    load_bound_pair(vae, new, new / "binding.json", pair)
