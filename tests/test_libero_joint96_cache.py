from collections import OrderedDict
import os
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.preflight_joint96_posttrain import compose_recipe
from wam.datasets.libero_joint96_cache import PairedLiberoCache, CacheRGBTransform, digest, file_hash
from test_joint96_mask_loss import make_model, make_sample, run_model


def test_cached_recipe_is_separate_and_deterministic():
    cfg = compose_recipe("libero_cached")
    assert cfg.joint96_posttrain.cached_visuals
    assert cfg.video_latent_cache == "off"
    assert cfg.init_checkpoint == os.environ["WAM_PRETRAIN_CHECKPOINT"]
    assert cfg.resume is None
    for split in ("train", "val"):
        assert cfg.data[split]._target_.endswith("LiberoJoint96CachedDataset")
        assert not cfg.data[split].image_color_jitter.enabled
        assert cfg.data[split].visual_cache.initialization_checkpoint == cfg.init_checkpoint
    assert compose_recipe("libero").data.train.image_color_jitter.enabled


@pytest.mark.parametrize("override", ["data.train.image_color_jitter.enabled=true",
    "data.train.image_camera_jitter.enabled=true", "data.val.image_color_jitter.enabled=true",
    "data.train.domain=robodojo", "video_latent_cache=read",
    "data.train._target_=wam.datasets.joint96_posttrain.Joint96PosttrainDataset"])
def test_cached_recipe_rejects_incompatible_overrides(override):
    with pytest.raises(ValueError):
        compose_recipe("libero_cached", [override])


def make_cache(tmp_path):
    reader = object.__new__(PairedLiberoCache)
    reader.roots = [tmp_path / name for name in ("vae_head", "vae_wrist", "stvae_head", "stvae_wrist")]
    reader.contracts = [{"world_size": 1, "view": name.name} for name in reader.roots]
    reader.length, reader.max_open_shards = 2, 1
    reader._open, reader._validated, reader._checked_payloads = OrderedDict(), {}, set()
    mapping = [{"idx": i, "episode_index": i, "frame_index": 0} for i in range(2)]
    for number, (root, contract) in enumerate(zip(reader.roots, reader.contracts)):
        shard = root / "shards/rank_00000"
        shard.mkdir(parents=True)
        array = torch.full((2, 48, 3, 16, 16), float(number + 1), dtype=torch.bfloat16).view(torch.uint16).numpy()
        array.tofile(shard / "features.mmap")
        np.save(shard / "indices.npy", np.arange(2))
        np.save(shard / "valid.npy", np.ones(2, dtype=bool))
        hashes = np.asarray([hashlib.sha256(row.tobytes()).hexdigest().encode() for row in array], dtype="S64")
        np.save(shard / "row_sha256.npy", hashes)
        (shard / "mapping.json").write_text(json.dumps(mapping))
        manifest = dict(contract=contract, rank=0, status="complete", dtype="bfloat16", storage_dtype="uint16",
            feature_shape=[48,3,16,16], shard_length=2, features="features.mmap", window_mapping="mapping.json",
            window_mapping_sha256=digest(mapping), sample_indices="indices.npy", valid="valid.npy", row_sha256="row_sha256.npy")
        (shard / "manifest.json").write_text(json.dumps(manifest))
        (shard / "COMPLETE.json").write_text(json.dumps(dict(status="complete",
            manifest_sha256=file_hash(shard / "manifest.json"), row_sha256_file_sha256=file_hash(shard / "row_sha256.npy"))))
    return reader, mapping


def test_channels_slots_and_lru_reopen(tmp_path):
    cache, mappings = make_cache(tmp_path)
    for mapping in mappings * 2:
        value = cache.get(mapping)
        assert value.shape == (96,3,16,48) and value.dtype == torch.bfloat16
        assert value[:48,...,:16].eq(1).all() and value[48:,...,:16].eq(3).all()
        assert value[...,16:32].eq(0).all()
        assert value[:48,...,32:].eq(2).all() and value[48:,...,32:].eq(4).all()
    assert len(cache._open) == 1 and len(cache._validated) == 4
    assert not cache.__getstate__()["_open"]


@pytest.mark.parametrize("corruption", ["mapping", "features", "manifest", "completion", "hashes"])
def test_corrupted_cache_fails_closed(tmp_path, corruption):
    cache, mappings = make_cache(tmp_path)
    shard = cache.roots[0] / "shards/rank_00000"
    if corruption == "mapping":
        mappings[0]["frame_index"] = 1
    elif corruption == "features":
        with (shard / "features.mmap").open("r+b") as stream:
            stream.write(b"xx")
    elif corruption == "manifest":
        (shard / "manifest.json").write_text((shard / "manifest.json").read_text() + " ")
    elif corruption == "completion":
        marker = json.loads((shard / "COMPLETE.json").read_text())
        marker["status"] = "partial"
        (shard / "COMPLETE.json").write_text(json.dumps(marker))
    else:
        np.save(shard / "row_sha256.npy", np.asarray([b"0" * 64] * 2, dtype="S64"))
    if corruption in ('mapping', 'completion'):
        with pytest.raises(ValueError):
            cache.get(mappings[0])
    else:
        assert cache.get(mappings[0]).shape == (96, 3, 16, 48)


def test_rgb_reference_matches_cache_preprocessing():
    rgb = torch.rand(3,3,512,512)
    expected = torch.nn.functional.interpolate(rgb, size=(256,256), mode="bicubic", align_corners=False).clamp(0,1)
    torch.testing.assert_close(CacheRGBTransform()(rgb), expected, rtol=0, atol=0)


def test_cached_training_bypasses_encoder_and_masks_padding():
    torch.set_num_threads(1)
    sample = make_sample()
    sample["video_latents"].requires_grad_(True)
    model = make_model()
    before = next(model.mot.parameters()).detach().clone()
    optimizer = torch.optim.AdamW([*model.mot.parameters(), *model.proprio_encoder.parameters()], lr=1e-4)
    with patch("wam.model.modules.codecs.video_latent_codec.encode_video_latents",
               side_effect=AssertionError("Cached batch must not encode RGB")):
        loss, logs = run_model(model, sample)
        assert torch.isfinite(loss)
        loss.backward()
        assert not sample["video_latents"].grad[0,...,16:32].count_nonzero()
        assert all(torch.isfinite(p.grad).all() for p in model.mot.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(next(model.mot.parameters()), before)
