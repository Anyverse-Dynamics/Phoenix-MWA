"""CPU integration of the joint loop, with real DiTs and stubbed frozen codecs."""
from unittest.mock import patch

import pytest
import torch

from wam.inference.frame_joint import infer_frame_joint_window
from wam.inference.online_action_policy import infer_online_action_chunk
from wam.model.modules.conditioning.frame_attention import (
    build_frame_memory_attention_mask, build_joint96_attention_mask,
)
from test_joint96_mask_loss import cpu_thread_limit, make_model, make_sample


def inference_model():
    model = make_model()
    model.eval = lambda: model.mot.eval()
    model.infer_video_scheduler = model.train_video_scheduler
    model.infer_action_scheduler = model.train_action_scheduler
    return model


def inference_kwargs():
    sample = make_sample()
    return dict(
        prompt=None, input_image=torch.zeros(1, 3, 3, 256, 256),
        num_frames=9, action_horizon=32, num_inference_steps=3, seed=719,
        context=sample["context"][:1], context_mask=sample["context_mask"][:1],
        proprio=sample["proprio"][0], camera_is_pad=sample["camera_is_pad"][:1],
        action_dim_is_pad=sample["action_dim_is_pad"][:1],
        proprio_dim_is_pad=sample["proprio_dim_is_pad"][:1],
    )


def run_inference(model, kwargs, first, *, perturb_step=None):
    inputs, masks, action_predictions = [], [], []
    pre, mot, post = model.video_expert.pre_dit, model.mot.forward, model.action_expert.post_dit

    def encode(model_arg, **encode_kwargs):
        assert model_arg is model
        torch.testing.assert_close(encode_kwargs["camera_is_pad"], kwargs["camera_is_pad"])
        return first.clone()

    def capture_video(*args, **kw):
        if len(inputs) == perturb_step:
            kw["x"] = kw["x"].clone()
            kw["x"][:, :, 1:, :, :16] += 2
        inputs.append(kw["x"].clone())
        assert kw["fuse_vae_embedding_in_latents"]
        assert kw["timestep"].shape == (1, 2)
        assert kw["frame_ids"].tolist() == [[0, 1, 2]]
        return pre(*args, **kw)

    def capture_mot(*args, **kw):
        masks.append(kw)
        return mot(*args, **kw)

    def capture_action(*args, **kw):
        prediction = post(*args, **kw)
        action_predictions.append(prediction.clone())
        return prediction

    with patch("wam.inference.frame_joint.video_codec.encode_input_image_latents_tensor", side_effect=encode), \
         patch("wam.inference.frame_joint.video_codec.decode_latents", side_effect=lambda model_arg, z, **kw: z.clone()), \
         patch("wam.inference.frame_joint._encode_online_condition_frame", side_effect=AssertionError("no history")), \
         patch.object(model.video_expert, "pre_dit", side_effect=capture_video), \
         patch.object(model.mot, "forward", side_effect=capture_mot), \
         patch.object(model.action_expert, "post_dit", side_effect=capture_action):
        result = infer_frame_joint_window(model, **kwargs)
    return result, inputs, masks, action_predictions


def test_joint96_inference_updates_future_video_and_actions_each_iteration():
    model, kwargs = inference_model(), inference_kwargs()
    first = make_sample()["video_latents"][:1, :, :1]
    result, inputs, masks, actions = run_inference(model, kwargs, first)
    assert len(inputs) == len(actions) == len(masks) == 3
    assert torch.isfinite(result["action"]).all()
    assert torch.isfinite(result["video"]).all()
    assert not result["action"][:, 40:].any()
    assert not result["video"][:, :, :, :, 16:32].any()
    for index, (video, payload) in enumerate(zip(inputs, masks)):
        assert video.shape == (1, 96, 3, 16, 48)
        assert not video[:, :, :, :, 16:32].any()
        torch.testing.assert_close(video[:, :, :1, :, :16], first[:, :, :, :, :16])
        if index:
            assert not torch.equal(video[:, :, 1:], inputs[index - 1][:, :, 1:])
        torch.testing.assert_close(payload["attention_mask"], build_joint96_attention_mask(
            video_seq_len=576, action_seq_len=32, video_tokens_per_frame=192, device="cpu",
        ))
        for name in ("video", "action"):
            torch.testing.assert_close(payload["key_masks_all"][name], payload["query_masks_all"][name])
        effective = model.mot._apply_key_mask(payload["attention_mask"], torch.cat(
            [payload["key_masks_all"]["video"], payload["key_masks_all"]["action"]], dim=1,
        ))
        valid = payload["key_masks_all"]["video"][0]
        assert not effective[0, 0, 576:, :576][:, ~valid].any()
        assert effective[0, 0, 576:, :576][:, valid].all()


def test_later_joint_denoising_action_reads_the_updated_future_video():
    kwargs, first = inference_kwargs(), make_sample()["video_latents"][:1, :, :1]
    baseline, _, _, actions = run_inference(inference_model(), kwargs, first)
    altered, _, _, changed = run_inference(inference_model(), kwargs, first, perturb_step=1)
    torch.testing.assert_close(actions[0], changed[0], rtol=0, atol=0)
    assert (actions[1] - changed[1]).abs().max() > 1e-6
    assert (baseline["action"][:, :40] - altered["action"][:, :40]).abs().max() > 1e-6


def test_libero_right_arm_joint_inference_reads_future_video():
    from hydra.utils import instantiate
    from scripts.preflight_joint96_posttrain import compose_recipe
    from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    cfg = compose_recipe("libero_cached")
    processor = instantiate(cfg.data.train.processor)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.data.train.pretrained_norm_stats))
    raw = {"state": {field.key: torch.zeros(33, field.raw_shape)
                     for field in cfg.data.train.shape_meta.state},
           "action": {"default": torch.zeros(32, 7)}}
    encoded = processor.action_state_merger.forward(
        processor.normalizer.forward(processor.action_state_transform(raw)),
    )
    kwargs, first = inference_kwargs(), make_sample()["video_latents"][:1, :, :1]
    kwargs.update(action_dim_is_pad=encoded["action_dim_is_pad"][None],
                  proprio_dim_is_pad=encoded["state_dim_is_pad"][None],
                  proprio=encoded["state"][:1])
    valid = ~kwargs["action_dim_is_pad"][0]
    assert valid.nonzero().flatten().tolist() == [*range(47, 53), 56]
    baseline, _, _, actions = run_inference(inference_model(), kwargs, first)
    altered, _, _, changed = run_inference(inference_model(), kwargs, first, perturb_step=1)
    torch.testing.assert_close(actions[0], changed[0], rtol=0, atol=0)
    assert (baseline["action"][:, valid] - altered["action"][:, valid]).abs().max() > 1e-6
    assert not baseline["action"][:, ~valid].any()
    assert not altered["action"][:, ~valid].any()
    assert not altered["video"][..., 16:32].any()


def test_inference_padding_values_cannot_change_the_generated_action():
    kwargs, first = inference_kwargs(), make_sample()["video_latents"][:1, :, :1]
    baseline, *_ = run_inference(inference_model(), kwargs, first)
    first[:, :, :, :, 16:32] = float("nan")
    kwargs["proprio"][:, 40:] = float("nan")
    altered, *_ = run_inference(inference_model(), kwargs, first)
    torch.testing.assert_close(baseline["action"], altered["action"], rtol=0, atol=0)
    torch.testing.assert_close(baseline["video"], altered["video"], rtol=0, atol=0)


@pytest.mark.parametrize("key", ["action_dim_is_pad", "proprio_dim_is_pad", "proprio"])
def test_joint_inference_requires_native_padding_and_proprio(key):
    kwargs = inference_kwargs()
    kwargs[key] = None
    with pytest.raises(ValueError, match="requires"):
        infer_frame_joint_window(inference_model(), **kwargs)


@pytest.mark.parametrize("key", ["memory_inputs", "vlm_current_images", "action"])
def test_joint_inference_cannot_take_ground_truth_action_or_auxiliary_inputs(key):
    kwargs = inference_kwargs()
    kwargs[key] = {}
    with pytest.raises(ValueError, match="does not accept|cannot condition"):
        infer_frame_joint_window(inference_model(), **kwargs)


def test_original_memory_inference_keeps_its_attention_and_history():
    model, kwargs = inference_model(), inference_kwargs()
    model.joint96_three_view = False
    model.video_expert.fuse_vae_embedding_in_latents = True
    kwargs["num_inference_steps"] = 1
    for key in ("camera_is_pad", "action_dim_is_pad", "proprio_dim_is_pad"):
        kwargs.pop(key)
    first = make_sample()["video_latents"][:1, :, :1]
    with patch("wam.inference.frame_joint.video_codec.encode_input_image_latents_tensor", return_value=first), \
         patch("wam.inference.frame_joint.video_codec.decode_latents", side_effect=lambda model_arg, z, **kw: z.clone()), \
         patch("wam.inference.frame_joint._encode_online_condition_frame",
               return_value=(first, torch.ones(1, 1, dtype=torch.bool))) as memory, \
         patch("wam.inference.frame_joint.build_joint96_attention_mask", side_effect=AssertionError("opt-in only")), \
         patch.object(model.mot, "forward", wraps=model.mot.forward) as mot:
        result = infer_frame_joint_window(model, **kwargs)
    assert memory.call_count == 2
    assert torch.isfinite(result["action"]).all()
    torch.testing.assert_close(mot.call_args.kwargs["attention_mask"], build_frame_memory_attention_mask(
        video_seq_len=960, action_seq_len=32, video_tokens_per_frame=192,
        num_condition_frames=3, num_anchor_condition_frames=1, device="cpu",
    ))


def test_joint96_rejects_static_video_cache_action_inference():
    with pytest.raises(ValueError, match="infer_frame_joint_window"):
        infer_online_action_chunk(
            inference_model(), prompt=None, input_image=torch.zeros(3, 256, 256), action_horizon=32,
        )
