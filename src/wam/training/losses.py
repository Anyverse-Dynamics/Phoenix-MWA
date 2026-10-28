"""Joint96 pretraining objective; no auxiliary memory or VLM losses."""

import torch
from wam.model.modules.conditioning.input_builder import (
    build_wam_inputs,
    joint96_camera_valid_mask,
)
from wam.model.modules.conditioning.frame_attention import build_joint96_attention_mask
from wam.model.modules.conditioning.video_rope import sequence_start_video_rope_time_ids


def _joint96_masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local sample mean, independent of valid camera/action dimension counts."""
    if prediction.shape != target.shape:
        raise ValueError(f"joint96 {name} prediction and target shapes must match.")
    valid = valid.to(device=prediction.device, dtype=torch.bool).expand_as(prediction)
    counts = valid.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError(
            f"joint96 {name} has a sample with no valid supervised elements."
        )
    # Mask before subtraction/squaring so invalid NaN/Inf values cannot leak.
    prediction = prediction.float().masked_fill(~valid, 0.0)
    target = target.float().masked_fill(~valid, 0.0)
    per_sample = (prediction - target).square().flatten(1).sum(dim=1) / counts
    weight = weight.to(device=prediction.device, dtype=torch.float32).reshape(-1)
    if weight.numel() != per_sample.numel():
        raise ValueError(f"joint96 {name} requires one timestep weight per sample.")
    # Equal per-rank training batches are averaged by DDP/DeepSpeed. Eval ranks
    # may hold different sources, so only the trainer may aggregate metrics.
    return (per_sample * weight).mean(), per_sample.detach().mean()


def _joint96_temporal_valid_mask(model, inputs: dict) -> torch.Tensor:
    latents = inputs["input_latents"]
    batch_size, _, frames, _, _ = latents.shape
    image_is_pad = inputs.get("image_is_pad")
    if image_is_pad is None:
        return torch.ones((batch_size, frames), dtype=torch.bool, device=latents.device)
    if bool(image_is_pad[:, 0].any()):
        raise ValueError(
            "joint96 requires a real first conditioning frame in every sample."
        )
    factor = int(model.vae.temporal_downsample_factor)
    if int(image_is_pad.shape[1]) != 1 + (frames - 1) * factor:
        raise ValueError("joint96 image padding does not match the latent timeline.")
    future_pad = image_is_pad[:, 1:].reshape(batch_size, frames - 1, factor).all(dim=2)
    if bool(future_pad.all(dim=1).any()):
        raise ValueError("joint96 requires real future frames in every sample.")
    return ~torch.cat((image_is_pad[:, :1], future_pad), dim=1)


def _joint96_training_loss(model, sample, *, tiled: bool):
    from wam.model.modules.conditioning.embodiment_prompt import action_prompt_kwargs

    inputs = build_wam_inputs(model, sample, tiled=tiled)
    if float(model.loss_lambda_video) != 1.0 or float(model.loss_lambda_action) != 1.0:
        raise ValueError(
            "joint96 requires lambda_video=lambda_action=1 (0.5 VAE + 0.5 CSVAE + action)."
        )
    if bool(getattr(model.action_expert, "training_time_rtc_enabled", False)):
        raise ValueError("joint96 does not enable training-time RTC prefixes.")
    if bool(getattr(model.video_expert, "action_conditioned", False)):
        raise ValueError(
            "joint96 requires the Joint video expert, not GT-action context conditioning."
        )
    for name in ("action_dim_loss_weight", "action_loss_weight"):
        value = inputs.get(name)
        if value is not None and not bool((value == 1).all()):
            raise ValueError(
                f"joint96 sample-equal loss does not support non-unit {name}."
            )
    for name in (
        "action_loss_weighted_valid_cells",
        "action_loss_weighted_valid_cells_pool",
    ):
        value = inputs.get(name)
        if value is not None and bool(value.any()):
            raise ValueError(f"joint96 sample-equal loss does not support {name}.")

    input_latents = inputs["input_latents"]
    batch_size, _, frames, height, width = input_latents.shape
    patch_t, patch_h, patch_w = map(int, model.video_expert.patch_size)
    if patch_t != 1 or height % patch_h or (width // 3) % patch_w:
        raise ValueError(
            "joint96 patches must align independently to every camera and frame."
        )
    spatial_valid = joint96_camera_valid_mask(
        inputs["camera_is_pad"],
        batch_size=batch_size,
        height=height,
        width=width,
        device=input_latents.device,
    )
    frame_valid = _joint96_temporal_valid_mask(model, inputs)
    latent_valid = spatial_valid[:, None, None] & frame_valid[:, None, :, None, None]
    input_latents = input_latents.masked_fill(~latent_valid, 0.0)
    noise_video = torch.randn_like(input_latents).masked_fill(~latent_valid, 0.0)
    timestep_video = model.train_video_scheduler.sample_training_t(
        batch_size=batch_size,
        device=model.device,
        dtype=input_latents.dtype,
    )
    noisy_video = model.train_video_scheduler.add_noise(
        input_latents, noise_video, timestep_video
    )
    noisy_video = noisy_video.masked_fill(~latent_valid, 0.0)
    noisy_video[:, :, :1] = input_latents[:, :, :1]
    target_video = model.train_video_scheduler.training_target(
        input_latents, noise_video, timestep_video
    )
    target_video = target_video.masked_fill(~latent_valid, 0.0)
    target_video[:, :, :1] = 0.0

    action = inputs["action"]
    action_valid = torch.ones_like(action, dtype=torch.bool)
    if inputs.get("action_mask") is not None:
        action_valid &= inputs["action_mask"]
    if inputs.get("action_is_pad") is not None:
        action_valid &= ~inputs["action_is_pad"].unsqueeze(-1)
    if inputs.get("action_dim_is_pad") is not None:
        action_valid &= ~inputs["action_dim_is_pad"].unsqueeze(1)
    if bool((~action_valid.flatten(1).any(dim=1)).any()):
        raise ValueError(
            "joint96 requires valid action supervision in every Robot sample."
        )
    action = action.masked_fill(~action_valid, 0.0)
    noise_action = torch.randn_like(action).masked_fill(~action_valid, 0.0)
    timestep_action = model.train_action_scheduler.sample_training_t(
        batch_size=batch_size,
        device=model.device,
        dtype=action.dtype,
    )
    noisy_action = model.train_action_scheduler.add_noise(
        action, noise_action, timestep_action
    )
    noisy_action = noisy_action.masked_fill(~action_valid, 0.0)
    target_action = model.train_action_scheduler.training_target(
        action, noise_action, timestep_action
    )
    target_action = target_action.masked_fill(~action_valid, 0.0)

    video_pre = model.video_expert.pre_dit(
        x=noisy_video,
        timestep=timestep_video[:, None].expand(-1, frames - 1),
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        fuse_vae_embedding_in_latents=True,
        frame_ids=sequence_start_video_rope_time_ids(
            batch_size, frames, device=input_latents.device
        ),
    )
    action_pre = model.action_expert.pre_dit(
        action_tokens=noisy_action,
        timestep=timestep_action,
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        **action_prompt_kwargs(model.action_expert, sample),
    )
    grid = (frames, height // patch_h, width // patch_w)
    if tuple(video_pre["meta"]["grid_size"]) != grid:
        raise ValueError(
            "joint96 VideoDiT grid does not match the three-view latent canvas."
        )
    spatial_token_valid = joint96_camera_valid_mask(
        inputs["camera_is_pad"],
        batch_size=batch_size,
        height=grid[1],
        width=grid[2],
        device=input_latents.device,
    ).flatten(1)
    tokens_per_frame = int(spatial_token_valid.shape[1])
    video_valid = (frame_valid[:, :, None] & spatial_token_valid[:, None, :]).flatten(1)
    action_token_valid = action_valid.any(dim=-1)
    if tuple(action_pre["tokens"].shape[:2]) != tuple(action_token_valid.shape):
        raise ValueError("joint96 requires one ActionDiT token per action step.")
    if tuple(video_pre["tokens"].shape[:2]) != tuple(video_valid.shape):
        raise ValueError("joint96 video token count does not match its validity mask.")
    attention_mask = build_joint96_attention_mask(
        video_seq_len=int(video_valid.shape[1]),
        action_seq_len=int(action_token_valid.shape[1]),
        video_tokens_per_frame=tokens_per_frame,
        device=input_latents.device,
    )
    validity = {"video": video_valid, "action": action_token_valid}
    states = {"video": video_pre, "action": action_pre}
    contexts = {}
    for name, pre in states.items():
        mask = pre["context_mask"]
        if mask.ndim == 3:
            mask = mask & validity[name][:, :, None]
        elif mask.ndim == 4:
            mask = mask & validity[name][:, None, :, None]
        else:
            raise ValueError(
                "joint96 expert context masks must be [B,S,L] or [B,1,S,L]."
            )
        contexts[name] = {"context": pre["context"], "mask": mask}
    tokens_out = model.mot(
        embeds_all={name: pre["tokens"] for name, pre in states.items()},
        attention_mask=attention_mask,
        freqs_all={name: pre["freqs"] for name, pre in states.items()},
        context_all=contexts,
        t_mod_all={name: pre["t_mod"] for name, pre in states.items()},
        key_masks_all=validity,
        query_masks_all=validity,
    )
    pred_video = model.video_expert.post_dit(tokens_out["video"], video_pre)[:, :, 1:]
    pred_action = model.action_expert.post_dit(tokens_out["action"], action_pre)
    future_valid = latent_valid[:, :, 1:]
    video_weight = model.train_video_scheduler.training_weight(timestep_video)
    loss_vae, mse_vae = _joint96_masked_mse(
        pred_video[:, :48],
        target_video[:, :48, 1:],
        future_valid,
        video_weight,
        name="VAE",
    )
    loss_csvae, mse_csvae = _joint96_masked_mse(
        pred_video[:, 48:],
        target_video[:, 48:, 1:],
        future_valid,
        video_weight,
        name="CSVAE",
    )
    loss_action, mse_action = _joint96_masked_mse(
        pred_action,
        target_action,
        action_valid,
        model.train_action_scheduler.training_weight(timestep_action),
        name="action",
    )
    loss_video = 0.5 * (loss_vae + loss_csvae)
    return loss_video + loss_action, {
        "loss_video": loss_video.detach(),
        "loss_action": loss_action.detach(),
        "loss_vae": 0.5 * loss_vae.detach(),
        "loss_csvae": 0.5 * loss_csvae.detach(),
        # Raw branch losses include timestep weighting, but not the 0.5 coefficients.
        "loss_vae_raw": loss_vae.detach(),
        "loss_csvae_raw": loss_csvae.detach(),
        "mse_vae": mse_vae,
        "mse_csvae": mse_csvae,
        "mse_action": mse_action,
    }


def frame_training_loss(model, sample, tiled=False, global_step=None):
    if not getattr(model, "joint96_three_view", False):
        raise ValueError("This release supports Joint96 pretraining only")
    return _joint96_training_loss(model, sample, tiled=tiled)
