from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wam.model.modules.codecs.joint_visual_encoder import JointVisualEncoder, _load_stvae_encoder_class, encoder_source_sha256
from wam.model.modules.codecs.video_latent_codec import encode_video_latents
from wam.model.backbones.wan22.helpers.joint96_init import expand_wan48_state_dict


class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.calls = []
        self.clears = 0
        self.model = SimpleNamespace(clear_cache=self.clear_cache)

    def clear_cache(self):
        self.clears += 1

    def encode(self, clips, **kwargs):
        clip = clips[0]
        self.calls.append(clip.clone())
        values = clip.mean((0, 2, 3))[::4]
        return values[None, None, :, None, None].expand(1, 48, len(values), 16, 16).clone()


class ToySTVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.last_input = None

    def encode_raw_video_latents(self, rgb, *, num_views):
        self.last_input = rgb.clone()
        outputs = []
        for view in range(num_views):
            clip = rgb[..., view * 256:(view + 1) * 256]
            values = clip.mean((1, 3, 4))[:, ::4] + 2
            outputs.append(values[:, None, :, None, None].expand(-1, 48, -1, 16, 16))
        return torch.cat(outputs, dim=-1)


def fixture_encoder():
    vae, stvae = ToyVAE(), ToySTVAE()
    encoder = JointVisualEncoder(vae, stvae, allow_cpu_for_tests=True)
    return vae, stvae, encoder


def test_channel_order_view_order_and_shared_rgb():
    vae, stvae, encoder = fixture_encoder()
    rgb = torch.stack([torch.full((3, 9, 256, 256), value) for value in (-0.5, 0.0, 0.5)])[None]
    pad = torch.tensor([[False, True, False]])
    z = encoder.encode(rgb, camera_is_pad=pad)
    assert z.shape == (1, 96, 3, 16, 48)
    assert z.dtype == torch.bfloat16
    torch.testing.assert_close(z[:, :48, :, :, :16], torch.full_like(z[:, :48, :, :, :16], -0.5))
    torch.testing.assert_close(z[:, 48:, :, :, :16], torch.full_like(z[:, 48:, :, :, :16], 1.5))
    assert torch.count_nonzero(z[..., 16:32]) == 0
    assert vae.clears == 4 and len(vae.calls) == 2
    torch.testing.assert_close(vae.calls[0], stvae.last_input[0, ..., :256])
    torch.testing.assert_close(vae.calls[1], stvae.last_input[0, ..., 512:768])
    assert not z.requires_grad and all(not p.requires_grad for p in encoder.parameters())
    assert not encoder.training and not stvae.training
    assert not any(key.startswith("vae.") for key in encoder.state_dict())


def test_padding_nan_invariance_and_condition_geometry():
    vae, stvae, encoder = fixture_encoder()
    rgb = torch.zeros(1, 3, 3, 1, 256, 256)
    pad = torch.tensor([[False, True, True]])
    before = encoder.encode(rgb, camera_is_pad=pad)
    rgb[:, 1:] = float("nan")
    after = encoder.encode(rgb, camera_is_pad=pad)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert after.shape == (1, 96, 1, 16, 48)
    model = SimpleNamespace(joint96_three_view=True, joint_visual_encoder=encoder)
    torch.testing.assert_close(encode_video_latents(model, rgb, camera_is_pad=pad), after)


def test_mask_and_rgb_validation():
    vae, stvae, encoder = fixture_encoder()
    rgb = torch.zeros(1, 3, 3, 1, 256, 256)
    with pytest.raises(ValueError, match="camera_is_pad"):
        encoder.encode(rgb, camera_is_pad=None)
    with pytest.raises(ValueError, match="real camera"):
        encoder.encode(rgb, camera_is_pad=torch.ones(1, 3, dtype=torch.bool))
    rgb[:, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        encoder.encode(rgb, camera_is_pad=torch.tensor([[False, True, True]]))


def test_fingerprint_fail_closed(tmp_path):
    target = tmp_path / "src/fastwam/models/wan22/online_dinov3_stvae_encoder.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n")
    first = encoder_source_sha256(tmp_path)
    target.write_text("x = 2\n")
    assert encoder_source_sha256(tmp_path) is first is None


class ToyDiT(nn.Module):
    patch_size = (1, 2, 2)

    def __init__(self, channels):
        super().__init__()
        self.patch_embedding = nn.Conv3d(channels, 8, self.patch_size)
        self.head = nn.Module()
        self.head.head = nn.Linear(8, 4 * channels)
        self.block = nn.Linear(8, 8)


def test_wan48_expansion_preserves_pretrained_geometry():
    old, new = ToyDiT(48), ToyDiT(96)
    expanded = expand_wan48_state_dict(old.state_dict(), new)
    new.load_state_dict(expanded, strict=True)
    torch.testing.assert_close(new.patch_embedding.weight[:, :48], old.patch_embedding.weight, rtol=0, atol=0)
    assert new.patch_embedding.weight[:, 48:].count_nonzero() == 0
    torch.testing.assert_close(new.block.weight, old.block.weight, rtol=0, atol=0)
    for key in ("weight", "bias"):
        before = getattr(old.head.head, key).reshape(4, 48, -1)
        after = getattr(new.head.head, key).reshape(4, 96, -1)
        torch.testing.assert_close(before, after[:, :48], rtol=0, atol=0)
        assert after[:, 48:].count_nonzero() == 0
    bad = dict(old.state_dict())
    del bad["block.weight"]
    with pytest.raises(ValueError, match="not strict"):
        expand_wan48_state_dict(bad, new)
