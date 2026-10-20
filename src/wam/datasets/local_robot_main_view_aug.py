"""Window-level main-camera augmentations; all randomness is identity-local.

These mild blur/fog/glass proxies are NOT the benchmark corruption algorithms.
The immutable accepted reader is inherited unchanged; projection, sampling,
normalization, color and language random streams remain the baseline's.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F

from wam.datasets.local_robot import LocalRobot80Dataset
from wam.datasets.local_robot_contracts import digest

NOISE_TYPES = ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur")
DEFAULT_CONFIG = dict(
    version="main_view_all9_v1", enabled=False, mask_prob=.075, sensor_noise_prob=.075,
    scope="all_nine_frames", mutually_exclusive=True, require_auxiliary_real_view=True,
    noise_types=list(NOISE_TYPES), noise_type_weights=[.2] * 5,
    motion_radius=[2, 4], motion_angle=[-45., 45.], gaussian_sigma=[.5, 1.2],
    zoom_max=[1.02, 1.08], zoom_step=.01, fog_alpha=[.05, .15], fog_grid_size=8,
    glass_sigma=[.5, .8], glass_max_displacement=1,
)


def validate_config(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown main-view options: {sorted(unknown)}")
        cfg.update(dict(config))
    for key in ("enabled", "mutually_exclusive", "require_auxiliary_real_view"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if (cfg["version"] != "main_view_all9_v1" or cfg["scope"] != "all_nine_frames"
            or not cfg["mutually_exclusive"] or not cfg["require_auxiliary_real_view"]):
        raise ValueError("Main-view recipe requires versioned all-nine, exclusive, auxiliary-safe augmentation")
    for key in ("mask_prob", "sensor_noise_prob"):
        if not math.isfinite(float(cfg[key])) or not 0 <= float(cfg[key]) <= 1:
            raise ValueError(f"Invalid probability: {key}")
    if float(cfg["mask_prob"]) + float(cfg["sensor_noise_prob"]) > 1:
        raise ValueError("Exclusive augmentation probabilities exceed one")
    if tuple(cfg["noise_types"]) != NOISE_TYPES or list(cfg["noise_type_weights"]) != [.2] * 5:
        raise ValueError("Exactly five equally weighted sensor-noise proxies required")
    bounds = {"motion_radius": (1, 8), "motion_angle": (-90, 90),
              "gaussian_sigma": (.01, 3), "zoom_max": (1.001, 1.2),
              "fog_alpha": (0, .3), "glass_sigma": (.01, 2)}
    for key, (lo, hi) in bounds.items():
        values = cfg[key]
        if len(values) != 2 or not lo <= float(values[0]) <= float(values[1]) <= hi:
            raise ValueError(f"Invalid mild strength bounds: {key}")
    if not .001 <= float(cfg["zoom_step"]) <= .05 or not 2 <= int(cfg["fog_grid_size"]) <= 32:
        raise ValueError("Invalid zoom step/fog grid")
    if int(cfg["glass_max_displacement"]) != 1:
        raise ValueError("Glass proxy requires one-pixel displacement")
    return cfg


def eligible_main_view(camera_is_pad, camera_keys, camera_roles, selected_views):
    if bool(camera_is_pad[0]) or camera_roles[0] != "main" or not camera_keys[0]:
        return False
    views = {v["key"]: v for v in selected_views}
    main = views.get(camera_keys[0], {})
    if not main.get("available", False) or not main.get("path"):
        return False
    main_path = Path(main["path"]).resolve()
    for slot in (1, 2):
        view = views.get(camera_keys[slot], {})
        if (not bool(camera_is_pad[slot]) and camera_roles[slot] not in ("pad", "main")
                and camera_keys[slot] and camera_keys[slot] != camera_keys[0]
                and view.get("available", False) and view.get("path")
                and Path(view["path"]).resolve() != main_path):
            return True
    return False


def select_mode(*, seed, identity, cfg, eligible, training):
    if not training or not cfg["enabled"] or not eligible:
        return 0
    rng = np.random.default_rng(int(digest([seed, "main_view_aug", identity]), 16))
    u = rng.random()
    return 1 if u < cfg["mask_prob"] else (2 if u < cfg["mask_prob"] + cfg["sensor_noise_prob"] else 0)


def _filter(rgb, kernel):
    radius = kernel.shape[-1] // 2
    weight = kernel.to(rgb)[None, None].expand(3, 1, -1, -1)
    return F.conv2d(F.pad(rgb, (radius,) * 4, mode="reflect"), weight, groups=3)


def _gaussian(rgb, sigma):
    radius = int(math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=rgb.dtype, device=rgb.device)
    kernel = torch.exp(-x.square() / (2 * sigma * sigma))
    kernel = kernel[:, None] * kernel[None, :]
    return _filter(rgb, kernel / kernel.sum())


def sensor_noise(rgb, kind, *, cfg, rng):
    """RGB [9,3,H,W] in [0,1]; a single shared corruption across time."""
    params = torch.zeros(10, dtype=torch.float32)
    if kind == 0:
        radius = int(rng.integers(int(cfg["motion_radius"][0]), int(cfg["motion_radius"][1]) + 1))
        angle = float(rng.uniform(*cfg["motion_angle"]))
        y, x = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1), indexing="ij")
        theta = math.radians(angle)
        along, across = x * math.cos(theta) + y * math.sin(theta), -x * math.sin(theta) + y * math.cos(theta)
        kernel = torch.exp(-across.square() / .5) * (along.abs() <= radius)
        result = _filter(rgb, kernel / kernel.sum())
        params[0:2] = torch.tensor([radius, angle])
    elif kind == 1:
        sigma = float(rng.uniform(*cfg["gaussian_sigma"]))
        result = _gaussian(rgb, sigma)
        params[2] = sigma
    elif kind == 2:
        maximum = float(rng.uniform(*cfg["zoom_max"]))
        n = int(math.ceil((maximum - 1) / cfg["zoom_step"])) + 1
        result = rgb.clone()
        for scale in np.linspace(1, maximum, n)[1:]:
            theta = rgb.new_tensor([[1 / scale, 0, 0], [0, 1 / scale, 0]])[None].expand(9, -1, -1)
            grid = F.affine_grid(theta, rgb.shape, align_corners=False)
            result = result + F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=False)
        result = result / n
        params[3:5] = torch.tensor([maximum, n])
    elif kind == 3:
        alpha = float(rng.uniform(*cfg["fog_alpha"]))
        size = int(cfg["fog_grid_size"])
        field = torch.from_numpy(rng.random((1, 1, size, size)).astype(np.float32)).to(rgb)
        field = F.interpolate(field, rgb.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        result = rgb * (1 - alpha * field) + alpha * field
        params[5:7] = torch.tensor([alpha, size])
    elif kind == 4:
        sigma = float(rng.uniform(*cfg["glass_sigma"]))
        blur = _gaussian(rgb, sigma)
        height, width = rgb.shape[-2:]
        dy, dx = rng.integers(-1, 2, (2, height, width))
        y = torch.from_numpy((np.arange(height)[:, None] + dy).clip(0, height - 1)).to(rgb.device)
        x = torch.from_numpy((np.arange(width)[None, :] + dx).clip(0, width - 1)).to(rgb.device)
        result = _gaussian(blur[:, :, y, x], sigma)
        params[7:9] = torch.tensor([sigma, 1])
    else:
        raise ValueError("Invalid noise type")
    result = result.clamp(0, 1)
    if not torch.isfinite(result).all():
        raise ValueError("Sensor corruption produced nonfinite RGB")
    return result, params


def augment_main(video, camera_is_pad, *, camera_keys, camera_roles, selected_views,
                 seed, identity, config=None, training=True):
    cfg = validate_config(config)
    eligible = eligible_main_view(camera_is_pad, camera_keys, camera_roles, selected_views)
    mode = select_mode(seed=seed, identity=identity, cfg=cfg, eligible=eligible, training=training)
    mask = torch.zeros(3, dtype=torch.bool)
    noise_type, params = -1, torch.zeros(10, dtype=torch.float32)
    out = video
    if mode:
        if video.shape != (3, 3, 9, 256, 256):
            raise ValueError("Main augmentation requires the full three-view nine-frame window")
        out = video.clone()
        if mode == 1:
            mask[0] = True
            out[0] = -1  # No clean main RGB remains; effective validity is authoritative.
        else:
            rng = np.random.default_rng(int(digest([seed, "main_sensor_noise", identity]), 16))
            noise_type = int(rng.choice(5, p=cfg["noise_type_weights"]))
            rgb = ((video[0] + 1) / 2).permute(1, 0, 2, 3)
            rgb, params = sensor_noise(rgb, noise_type, cfg=cfg, rng=rng)
            out[0] = rgb.permute(1, 0, 2, 3) * 2 - 1
    return out, dict(main_aug_eligible=eligible, main_aug_mode=mode,
                    augmentation_camera_mask=mask, main_noise_type=noise_type, main_noise_params=params)


class LocalRobot80MainAugDataset(LocalRobot80Dataset):
    def __init__(self, *args, main_view_augmentation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_view_augmentation = validate_config(main_view_augmentation)
        if self.split != "train" and self.main_view_augmentation["enabled"]:
            raise ValueError("Validation cannot enable main-camera augmentation")

    def read_window(self, episode_id, start, *, draw=0, with_video=True, with_text=True, normalize=True):
        item = super().read_window(episode_id, start, draw=draw, with_video=with_video,
                                   with_text=with_text, normalize=normalize)
        if with_video:
            ep = self.package.episode(episode_id)
            identity = [self.mixture.epoch if self.split == "train" else 0,
                        ep["record"]["sample_id"], int(start), int(draw)]
            item["video"], meta = augment_main(item["video"], item["camera_is_pad"],
                camera_keys=item["camera_keys"], camera_roles=item["camera_roles"],
                selected_views=ep["record"]["selected_views"], seed=self.seed, identity=identity,
                config=self.main_view_augmentation, training=self.split == "train")
            item.update(meta)
        return item


def count_batch(sample, sources):
    """Consumed draws only; fixed schema on every rank including empty categories."""
    groups = ["all", "color_off", "color_on", *sources]
    names = [f"main_aug/{g}/{k}" for g in groups
             for k in ("total", "eligible", "mask", "noise", "none", *NOISE_TYPES)]
    values = dict.fromkeys(names, 0.)
    if "main_aug_block_lane" in sample:
        block = torch.stack((sample["main_aug_block_lane"], sample["main_aug_block_anchor"],
                             sample["main_aug_block_clean_count"],
                             sample["main_aug_block_singleton"]), dim=1).detach().cpu().tolist()
        values.update({
            "main_aug/block/total": sum(lane == 0 for lane, _, _, _ in block),
            "main_aug/block/reserved": sum(bool(anchor) for _, anchor, _, _ in block),
            "main_aug/block/singleton": sum(lane == 0 and bool(single) for lane, _, _, single in block),
            "main_aug/block/violations": sum(lane == 0 and clean < 1 for lane, _, clean, _ in block),
        })
    # Accelerator may already have moved metadata to CUDA. One small packed
    # transfer, rather than per-field/per-sample .item() synchronizations.
    packed = torch.stack((sample["main_aug_mode"].float(), sample["main_noise_type"].float(),
                          sample["main_aug_eligible"].float(), sample["color_params"][:, 0].float()), dim=1).detach().cpu().tolist()
    for i, source in enumerate(sample["source_id"]):
        mode, kind, eligible, color_on = packed[i]
        mode, kind = int(mode), int(kind)
        flags = ("total", "eligible" if eligible else None,
                 ("none", "mask", "noise")[mode], NOISE_TYPES[kind] if kind >= 0 else None)
        color = "color_on" if color_on else "color_off"
        for group in ("all", color, source):
            for flag in flags:
                if flag:
                    values[f"main_aug/{group}/{flag}"] += 1
    return Counter(values)
