"""CPU-only Joint96 contracts using real small VideoDiT, ActionDiT and MoT."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from wam.model.backbones.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from wam.model.backbones.wan22.wan_video_dit import WanVideoDiT
from wam.model.modules.conditioning.frame_attention import (
    build_frame_memory_attention_mask,
    build_joint96_attention_mask,
)
from wam.model.modules.conditioning.input_builder import build_wam_inputs, joint96_camera_valid_mask
from wam.model.modules.experts.action_dit import ActionDiT
from wam.model.modules.memory.proprio_encoder import ProprioContextEncoder
from wam.model.modules.mot.mixture_of_transformers import MoT
from wam.training.losses import _joint96_masked_mse, frame_training_loss


@pytest.fixture(autouse=True)
def cpu_thread_limit():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(*, checkpoint=False):
    torch.manual_seed(703)
    common = dict(text_dim=16, freq_dim=16, eps=1e-6, num_heads=2, attn_head_dim=8, num_layers=2)
    video = WanVideoDiT(
        hidden_dim=32, in_dim=96, out_dim=96, ffn_dim=64, patch_size=(1, 2, 2),
        has_image_input=False, seperated_timestep=True, use_gradient_checkpointing=checkpoint, **common,
    )
    action = ActionDiT(hidden_dim=24, action_dim=80, ffn_dim=48, use_gradient_checkpointing=checkpoint, **common)
    mot = MoT({"video": video, "action": action}, attention_backend="sdpa", mot_checkpoint_mixed_attn=checkpoint)
    return SimpleNamespace(
        joint96_three_view=True, device=torch.device("cpu"), torch_dtype=torch.float32,
        video_expert=video, action_expert=action, mot=mot, vae=SimpleNamespace(temporal_downsample_factor=4),
        proprio_dim=80, proprio_encoder=ProprioContextEncoder(80, 16),
        understanding_enabled=False, memory_enabled=False, memory_video_enabled=False,
        future_delta_enabled=False, semantic_future_alignment_enabled=False, track_bridge=None,
        loss_lambda_video=1., loss_lambda_action=1.,
        train_video_scheduler=WanContinuousFlowMatchScheduler(shift=6),
        train_action_scheduler=WanContinuousFlowMatchScheduler(shift=1),
        _action_token_seq_len_for_mask=lambda horizon: horizon,
    )


def make_sample():
    torch.manual_seed(705)
    sample = dict(
        video_latents=torch.randn(2, 96, 3, 16, 48),
        video_shape=torch.tensor([[3, 3, 9, 256, 256]]).expand(2, -1),
        camera_is_pad=torch.tensor([[False, True, False], [False, False, False]]),
        action=torch.randn(2, 32, 80), proprio=torch.randn(2, 1, 80),
        context=torch.randn(2, 4, 16), context_mask=torch.ones(2, 4, dtype=torch.bool),
        action_dim_is_pad=torch.zeros(2, 80, dtype=torch.bool),
        proprio_dim_is_pad=torch.zeros(2, 80, dtype=torch.bool),
        action_is_pad=torch.zeros(2, 32, dtype=torch.bool),
        action_mask=torch.ones(2, 32, 80, dtype=torch.bool),
        image_is_pad=torch.zeros(2, 9, dtype=torch.bool),
    )
    sample["action_dim_is_pad"][0, 40:] = True
    sample["proprio_dim_is_pad"][0, 40:] = True
    sample["action_is_pad"][1, 30:] = True
    sample["action_mask"][1, 8, :] = False
    sample["image_is_pad"][1, 5:] = True
    return sample


def action_valid(sample):
    return sample["action_mask"] & ~sample["action_dim_is_pad"][:, None] & ~sample["action_is_pad"][:, :, None]


def run_model(model, sample):
    torch.manual_seed(707)
    return frame_training_loss(model, sample)


def test_checkpoint_every_layer_preserves_loss_and_parameter_gradients():
    outputs = []
    for stride in (2, 1):
        model, sample = make_model(checkpoint=True), make_sample()
        model.mot.checkpoint_layer_stride = stride
        loss, _ = run_model(model, sample)
        loss.backward()
        gradients = {"mot." + name: p.grad.clone() for name, p in model.mot.named_parameters() if p.grad is not None}
        gradients.update({"proprio." + name: p.grad.clone()
                          for name, p in model.proprio_encoder.named_parameters() if p.grad is not None})
        outputs.append((loss.detach(), gradients))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    assert outputs[0][1].keys() == outputs[1][1].keys()
    for name in outputs[0][1]:
        torch.testing.assert_close(outputs[0][1][name], outputs[1][1][name], rtol=0, atol=0)


def test_camera_canvas_and_token_order():
    mask = torch.tensor([[False, True, False]])
    valid = joint96_camera_valid_mask(mask, batch_size=1, height=2, width=6, device="cpu")
    assert valid.flatten().tolist() == [True, True, False, False, True, True] * 2
    for bad in (torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool), mask.float()):
        with pytest.raises(ValueError):
            joint96_camera_valid_mask(bad, batch_size=1, height=2, width=6, device="cpu")


def test_sample_equal_mse_ignores_dimension_count_and_invalid_nans():
    pred = torch.tensor([[1., float("nan"), float("inf")], [3., 3., 3.]], requires_grad=True)
    valid = torch.tensor([[True, False, False], [True, True, True]])
    loss, mse = _joint96_masked_mse(pred, torch.zeros_like(pred), valid, torch.ones(2), name="test")
    assert loss.item() == mse.item() == 5.
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[1., 0., 0.], [1., 1., 1.]]))


def test_sample_equal_mse_applies_per_sample_timestep_weight():
    pred = torch.tensor([[1., 9.], [3., 3.]], requires_grad=True)
    loss, raw = _joint96_masked_mse(pred, torch.zeros_like(pred), torch.tensor([[True, False], [True, True]]),
                                    torch.tensor([2., .5]), name="test")
    assert loss.item() == 3.25
    assert raw.item() == 5.


def test_video_loss_one_camera_and_three_cameras_are_sample_equal():
    pred = torch.ones(2, 48, 2, 2, 6, requires_grad=True)
    with torch.no_grad():
        pred[1].mul_(3.)
    valid = joint96_camera_valid_mask(torch.tensor([[False, True, True], [False, False, False]]),
                                     batch_size=2, height=2, width=6, device="cpu")[:, None, None]
    loss, _ = _joint96_masked_mse(pred, torch.zeros_like(pred), valid, torch.ones(2), name="video")
    assert loss.item() == 5.
    loss.backward()
    # d(mean_i MSE_i)/dp = 2 * error / (batch_size * valid_count_i).
    expected = torch.zeros_like(pred)
    expected[0, :, :, :, :2] = 2.0 * 1.0 / (2 * (48 * 2 * 2 * 2))
    expected[1] = 2.0 * 3.0 / (2 * (48 * 2 * 2 * 6))
    torch.testing.assert_close(pred.grad, expected, rtol=0, atol=0)
    assert not pred.grad[0, :, :, :, 2:].any()
    assert pred.grad[0, :, :, :, :2].sum(dtype=torch.float64).item() == pytest.approx(1.)
    assert pred.grad[1].sum(dtype=torch.float64).item() == pytest.approx(3.)


def test_loss_does_not_collectively_mix_different_eval_sources():
    pred = torch.tensor([[2., 2.]], requires_grad=True)
    with patch("wam.training.losses._distributed_sum_detached", side_effect=AssertionError("trainer owns reduction")):
        loss, raw = _joint96_masked_mse(pred, torch.zeros_like(pred), torch.ones_like(pred, dtype=torch.bool),
                                      torch.ones(1), name="test")
    assert loss.item() == raw.item() == 4.
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.full_like(pred, 2.))


def test_equal_rank_gradient_average_matches_one_global_batch():
    shared = torch.tensor(2., requires_grad=True)
    targets = (torch.tensor([[1., 1.], [3., 3.]]), torch.tensor([[5., 5.], [7., 7.]]))
    rank_gradients = []
    rank_losses = []
    for target in targets:
        loss, _ = _joint96_masked_mse(shared.expand_as(target), target, torch.ones_like(target, dtype=torch.bool),
                                    torch.ones(2), name="test")
        rank_gradients.append(torch.autograd.grad(loss, shared)[0])
        rank_losses.append(loss.detach())
    target = torch.cat(targets)
    loss, _ = _joint96_masked_mse(shared.expand_as(target), target, torch.ones_like(target, dtype=torch.bool),
                                torch.ones(4), name="test")
    torch.testing.assert_close(torch.stack(rank_gradients).mean(), torch.autograd.grad(loss, shared)[0])
    torch.testing.assert_close(torch.stack(rank_losses).mean(), loss.detach())


def test_sample_equal_mse_rejects_a_sample_without_supervision():
    with pytest.raises(ValueError, match="no valid supervised"):
        _joint96_masked_mse(torch.zeros(2, 3), torch.zeros(2, 3),
                            torch.tensor([[True, True, True], [False, False, False]]), torch.ones(2), name="test")


def test_no_history_attention_layout():
    mask = build_joint96_attention_mask(video_seq_len=18, action_seq_len=4, video_tokens_per_frame=6, device="cpu")
    assert mask[:6, :6].all()
    assert not mask[:6, 6:].any()
    assert mask[6:18, :18].all()
    assert not mask[:18, 18:].any()
    assert mask[18:, :6].all()
    assert mask[18:, 6:18].all()
    assert mask[18:, 18:].all()
    memory_mask = build_frame_memory_attention_mask(
        video_seq_len=18, action_seq_len=4, video_tokens_per_frame=6, num_condition_frames=1,
        num_anchor_condition_frames=0, device="cpu",
    )
    assert not memory_mask[18:, 6:18].any()
    changed = torch.zeros_like(mask)
    changed[18:, 6:18] = True
    torch.testing.assert_close(mask ^ memory_mask, changed)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_action_prediction_reads_noisy_future_video_with_gradients(checkpoint):
    model, sample = make_model(checkpoint=checkpoint), make_sample()
    sample["video_latents"].requires_grad_()
    captured = {}
    pre = model.video_expert.pre_dit
    post = model.action_expert.post_dit

    def capture_loss(*args, **kwargs):
        result = _joint96_masked_mse(*args, **kwargs)
        if kwargs["name"] == "action":
            captured["action_loss"] = result[0]
        return result

    def capture_pre(*args, **kwargs):
        captured["noisy_video"] = kwargs["x"]
        return pre(*args, **kwargs)

    def capture_post(*args, **kwargs):
        prediction = post(*args, **kwargs)
        captured["action_prediction"] = prediction
        return prediction

    with patch.object(model.video_expert, "pre_dit", side_effect=capture_pre), \
         patch.object(model.action_expert, "post_dit", side_effect=capture_post), \
         patch("wam.training.losses._joint96_masked_mse", side_effect=capture_loss):
        run_model(model, sample)
    noisy = captured["noisy_video"]
    assert not torch.equal(noisy[0, :, 1:, :, :16], sample["video_latents"][0, :, 1:, :, :16])
    captured["action_loss"].backward()
    grad = sample["video_latents"].grad
    assert torch.isfinite(grad).all()
    assert grad[0, :, 1:, :, :16].count_nonzero()
    assert grad[1, :, 1, :, :16].count_nonzero()
    assert not grad[0, :, :, :, 16:32].any()
    assert not grad[1, :, 2:].any()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in model.video_expert.parameters())


def test_action_prediction_changes_with_future_noise_not_the_current_frame():
    model, sample = make_model(), make_sample()
    pre = model.video_expert.pre_dit
    post = model.action_expert.post_dit
    video_post = model.video_expert.post_dit
    predictions, current_predictions = [], []

    def perturb_future(*args, **kwargs):
        changed = kwargs["x"].clone()
        changed[0, :, 1:, :, :16] += 3.0
        return pre(*args, **{**kwargs, "x": changed})

    def capture_action(*args, **kwargs):
        result = post(*args, **kwargs)
        predictions.append(result.detach().clone())
        return result

    def capture_video(*args, **kwargs):
        result = video_post(*args, **kwargs)
        current_predictions.append(result[:, :, 0].detach().clone())
        return result

    with torch.no_grad(), \
         patch.object(model.action_expert, "post_dit", side_effect=capture_action), \
         patch.object(model.video_expert, "post_dit", side_effect=capture_video):
        run_model(model, sample)
        with patch.object(model.video_expert, "pre_dit", side_effect=perturb_future):
            run_model(model, sample)
    assert (predictions[0][0] - predictions[1][0]).abs().max() > 1e-6
    torch.testing.assert_close(predictions[0][1], predictions[1][1], rtol=0, atol=0)
    torch.testing.assert_close(current_predictions[0], current_predictions[1], rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_real_forward_backward_no_history_masks_noise_and_targets(checkpoint):
    model, sample = make_model(checkpoint=checkpoint), make_sample()
    sample["video_latents"].requires_grad_()
    sample["action"].requires_grad_()
    sample["proprio"].requires_grad_()
    captured = {}
    original_video = model.video_expert.pre_dit
    original_action = model.action_expert.pre_dit
    original_post_video = model.video_expert.post_dit
    original_post_action = model.action_expert.post_dit
    original_mot = model.mot.forward

    def video_pre(**kwargs):
        captured["noisy_video"] = kwargs["x"]
        captured["frame_ids"] = kwargs["frame_ids"]
        captured["timesteps"] = kwargs["timestep"]
        return original_video(**kwargs)

    def action_pre(**kwargs):
        captured["noisy_action"] = kwargs["action_tokens"]
        return original_action(**kwargs)

    def video_post(*args):
        result = original_post_video(*args)
        result.retain_grad()
        captured["pred_video"] = result
        return result

    def action_post(*args):
        result = original_post_action(*args)
        result.retain_grad()
        captured["pred_action"] = result
        return result

    def mot_forward(**kwargs):
        captured["mot"] = kwargs
        result = original_mot(**kwargs)
        captured["tokens"] = result
        return result

    with patch.object(model.video_expert, "pre_dit", side_effect=video_pre), \
         patch.object(model.action_expert, "pre_dit", side_effect=action_pre), \
         patch.object(model.video_expert, "post_dit", side_effect=video_post), \
         patch.object(model.action_expert, "post_dit", side_effect=action_post), \
         patch.object(model.mot, "forward", side_effect=mot_forward), \
         patch("wam.training.losses._encode_single_memory_frame", side_effect=AssertionError("history called")), \
         patch("wam.training.losses.build_frame_vlm_pack", side_effect=AssertionError("VLM called")):
        total, stats = run_model(model, sample)
        total.backward()

    assert torch.isfinite(total)
    torch.testing.assert_close(total.detach(), .5 * stats["loss_vae_raw"] + .5 * stats["loss_stvae_raw"] + stats["loss_action"])
    torch.testing.assert_close(stats["loss_video"], stats["loss_vae"] + stats["loss_stvae"])
    assert captured["timesteps"].shape == (2, 2)
    assert captured["frame_ids"].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert not captured["noisy_video"][0, :, :, :, 16:32].any()
    assert not captured["noisy_video"][1, :, 2:].any()
    torch.testing.assert_close(captured["noisy_video"][0, :, 0, :, :16], sample["video_latents"][0, :, 0, :, :16])
    valid_action = action_valid(sample)
    assert not captured["noisy_action"][~valid_action].any()
    assert not sample["action"].grad[~valid_action].any()
    assert not sample["proprio"].grad[0, :, 40:].any()
    assert not sample["video_latents"].grad[0, :, :, :, 16:32].any()
    assert not sample["video_latents"].grad[1, :, 2:].any()
    assert not captured["pred_video"].grad[:, :, :1].any()
    assert not captured["pred_video"].grad[0, :, :, :, 16:32].any()
    assert not captured["pred_video"].grad[1, :, 2:].any()
    assert not captured["pred_action"].grad[~valid_action].any()
    for branch in (model.video_expert, model.action_expert):
        assert any(p.grad is not None and p.grad.count_nonzero() for p in branch.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in branch.parameters())
    payload = captured["mot"]
    for name, valid in payload["key_masks_all"].items():
        torch.testing.assert_close(valid, payload["query_masks_all"][name])
        assert not captured["tokens"][name][~valid].any()
    key_valid = torch.cat(list(payload["key_masks_all"].values()), dim=1)
    mask = model.mot._apply_key_mask(payload["attention_mask"], key_valid)
    mask = model.mot._apply_query_mask(mask, key_valid)
    assert mask.any(dim=-1).all()
    for batch_index in range(2):
        live, pad = key_valid[batch_index], ~key_valid[batch_index]
        assert not mask[batch_index, 0][live][:, pad].any()
        assert torch.equal(mask[batch_index, 0][pad][:, pad], torch.eye(int(pad.sum()), dtype=torch.bool))


def test_padding_perturbations_cannot_change_forward_or_parameter_gradients():
    model, sample = make_model(), make_sample()
    changed = {k: v.detach().clone() for k, v in sample.items()}
    changed["video_latents"][0, :, :, :, 16:32] = float("nan")
    changed["video_latents"][1, :, 2:] = 1e6
    changed["action"][~action_valid(sample)] = float("nan")
    changed["proprio"][0, :, 40:] = float("nan")
    total1, stats1 = run_model(model, sample)
    total1.backward()
    grads1 = {n: p.grad.clone() for n, p in model.mot.named_parameters() if p.grad is not None}
    model.mot.zero_grad(set_to_none=True)
    total2, stats2 = run_model(model, changed)
    total2.backward()
    torch.testing.assert_close(total1, total2, rtol=0, atol=0)
    for key in stats1:
        torch.testing.assert_close(stats1[key], stats2[key], rtol=0, atol=0)
    for name, param in model.mot.named_parameters():
        if name in grads1:
            torch.testing.assert_close(param.grad, grads1[name], rtol=0, atol=0)


def test_mot_blocks_directly_perturbed_padding_tokens():
    model = make_model()
    torch.manual_seed(801)
    batch = 2
    video = model.video_expert.pre_dit(
        x=torch.randn(batch, 96, 3, 2, 6), timestep=torch.ones(batch, 2),
        context=torch.randn(batch, 3, 16), context_mask=torch.ones(batch, 3, dtype=torch.bool),
        fuse_vae_embedding_in_latents=True,
    )
    action = model.action_expert.pre_dit(
        action_tokens=torch.randn(batch, 4, 80), timestep=torch.ones(batch),
        context=torch.randn(batch, 3, 16), context_mask=torch.ones(batch, 3, dtype=torch.bool),
    )
    valid = {
        "video": torch.tensor([[True, False, True] * 3, [True] * 9]),
        "action": torch.tensor([[True, True, False, False], [True] * 4]),
    }
    states = {"video": video, "action": action}
    base = {name: state["tokens"].detach().requires_grad_() for name, state in states.items()}
    changed = {name: tensor.detach().masked_fill(~valid[name][:, :, None], float("nan")).requires_grad_()
               for name, tensor in base.items()}
    kwargs = dict(
        attention_mask=build_joint96_attention_mask(video_seq_len=9, action_seq_len=4, video_tokens_per_frame=3, device="cpu"),
        freqs_all={name: state["freqs"].detach() for name, state in states.items()},
        t_mod_all={name: state["t_mod"].detach() for name, state in states.items()},
        context_all={name: {"context": state["context"].detach(), "mask": state["context_mask"] & valid[name][:, :, None]}
                     for name, state in states.items()},
        key_masks_all=valid, query_masks_all=valid,
    )
    baseline = model.mot(embeds_all=base, **kwargs)
    altered = model.mot(embeds_all=changed, **kwargs)
    sum(value.square().sum() for value in altered.values()).backward()
    for name in states:
        torch.testing.assert_close(baseline[name], altered[name], rtol=0, atol=0)
        assert torch.isfinite(altered[name]).all()
        assert not altered[name][~valid[name]].any()
        assert not changed[name].grad[~valid[name]].any()
        assert changed[name].grad[valid[name]].count_nonzero()


def test_real_forward_all_cameras_present():
    model, sample = make_model(), make_sample()
    for key in ("camera_is_pad", "image_is_pad", "action_is_pad", "action_dim_is_pad"):
        sample[key].fill_(False)
    sample["action_mask"].fill_(True)
    loss, diagnostics = run_model(model, sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in diagnostics.values())


@pytest.mark.parametrize("case", ["camera", "future", "first", "action", "mask_missing", "bad_latents", "cached_shape"])
def test_malformed_or_all_padding_sample_fails_closed(case):
    model, sample = make_model(), make_sample()
    if case == "camera":
        sample["camera_is_pad"][0] = True
    elif case == "future":
        sample["image_is_pad"][0, 1:] = True
    elif case == "first":
        sample["image_is_pad"][0, 0] = True
    elif case == "action":
        sample["action_dim_is_pad"][0] = True
    elif case == "mask_missing":
        del sample["camera_is_pad"]
    elif case == "bad_latents":
        sample["video_latents"] = sample["video_latents"][:, :48]
    else:
        sample["video_shape"] = torch.tensor([2, 3, 9, 256, 256])
    with pytest.raises(ValueError):
        run_model(model, sample)


@pytest.mark.parametrize("field", ["memory_video_anchor", "memory_video_recent", "vlm_current_images", "track_teacher_feature"])
def test_auxiliary_samples_are_rejected(field):
    model, sample = make_model(), make_sample()
    sample[field] = torch.zeros(1)
    with pytest.raises(ValueError, match="auxiliary inputs"):
        run_model(model, sample)


@pytest.mark.parametrize("flag", ["memory_enabled", "memory_video_enabled", "understanding_enabled", "future_delta_enabled", "semantic_future_alignment_enabled"])
def test_auxiliary_model_modes_are_rejected(flag):
    model, sample = make_model(), make_sample()
    setattr(model, flag, True)
    with pytest.raises(ValueError, match=flag):
        run_model(model, sample)


@pytest.mark.parametrize("name", ["action_loss_weight", "action_dim_loss_weight", "action_loss_weighted_valid_cells"])
def test_source_weights_cannot_silently_change_equal_sample_objective(name):
    model, sample = make_model(), make_sample()
    shape = (2, 80) if name == "action_dim_loss_weight" else (2,)
    sample[name] = torch.ones(shape) * 2
    with pytest.raises(ValueError, match=name):
        run_model(model, sample)


def test_rgb_encoder_receives_camera_mask_and_input_builder_zeros_latent_padding():
    model, sample = make_model(), make_sample()
    del sample["video_latents"]
    sample["video"] = torch.zeros(2, 3, 3, 9, 256, 256)
    sample["video"][0, 1] = float("nan")
    def encode(model_arg, video, **kwargs):
        assert model_arg is model
        torch.testing.assert_close(kwargs["camera_is_pad"], sample["camera_is_pad"])
        assert torch.isfinite(video).all()
        assert not video[0, 1].any()
        return torch.ones(2, 96, 3, 16, 48)
    with patch("wam.model.modules.conditioning.input_builder.video_codec.encode_video_latents", side_effect=encode) as codec:
        inputs = build_wam_inputs(model, sample)
    assert codec.call_count == 1
    assert not inputs["input_latents"][0, :, :, :, 16:32].any()
    assert inputs["input_latents"][0, :, :, :, :16].all()


def test_full_rgb_to_loss_uses_actual_codec_dispatch():
    class TinyCodec(nn.Module):
        def __init__(self):
            super().__init__()
            self.mask = None

        def encode(self, video, *, camera_is_pad):
            self.mask = camera_is_pad
            means = video.mean(dim=(2, 4, 5))[:, :, ::4]
            canvas = torch.cat([
                means[:, view, None, :, None, None].expand(-1, 48, -1, 16, 16)
                for view in range(3)
            ], dim=-1)
            return torch.cat((canvas, canvas + 2), dim=1)

    model, sample = make_model(), make_sample()
    model.joint_visual_encoder = TinyCodec()
    del sample["video_latents"]
    sample["video"] = torch.zeros(2, 3, 3, 9, 256, 256)
    sample["video"][0, 1] = float("nan")
    loss, diagnostics = run_model(model, sample)
    loss.backward()
    torch.testing.assert_close(model.joint_visual_encoder.mask, sample["camera_is_pad"])
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert model.action_expert.head.weight.grad.count_nonzero()
    assert model.video_expert.head.head.weight.grad.count_nonzero()


def test_default_training_does_not_enter_joint96_route():
    model = SimpleNamespace(joint96_three_view=False)
    with patch("wam.training.losses.build_wam_inputs", side_effect=RuntimeError("legacy path")), \
         patch("wam.training.losses._joint96_training_loss", side_effect=AssertionError("opt-in only")):
        with pytest.raises(RuntimeError, match="legacy path"):
            frame_training_loss(model, {})


def test_default_video_loss_retains_element_weighting():
    from wam.training.losses import _frame_video_loss
    model = SimpleNamespace(train_video_scheduler=SimpleNamespace(training_weight=lambda t: torch.ones_like(t)))
    pred = torch.tensor([[[[[1., 0., 0.]]]], [[[[3., 3., 3.]]]]])
    loss = _frame_video_loss(model, pred_video=pred, target_video=torch.zeros_like(pred), image_is_pad=None,
                             latent_spatial_valid_mask=torch.tensor([[[True, False, False]], [[True, True, True]]]),
                             timestep=torch.ones(2))
    assert loss.item() == 7.
