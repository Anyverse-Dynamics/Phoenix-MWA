"""Opt-in, deterministic block-level main-camera protection (2026-10-11)."""

from collections import defaultdict

import numpy as np
import torch

from wam.datasets.local_robot_contracts import digest
from wam.datasets.local_robot_main_view_aug import (
    eligible_main_view, sensor_noise, validate_config,
)

BLOCK_POLICY = "reserve_clean_v1"


def plan_block_modes(*, seed, identity, valid, eligible, mask_prob, noise_prob):
    """Preserve eligible-lane marginals while reserving a real clean main view."""
    valid, eligible = np.asarray(valid, dtype=bool), np.asarray(eligible, dtype=bool)
    if valid.shape != (4,) or eligible.shape != (4,) or np.any(eligible & ~valid):
        raise ValueError("Block protection requires four consistent camera flags")
    if not valid.any():
        raise ValueError("Block has no real main view; padding cannot satisfy protection")
    if not (0 <= mask_prob <= 1 and 0 <= noise_prob <= 1 and mask_prob + noise_prob <= 1):
        raise ValueError("Invalid block augmentation probabilities")
    rng = np.random.default_rng(int(digest([seed, BLOCK_POLICY, identity]), 16))
    modes = np.zeros(4, dtype=np.int64)
    anchor = -1
    m = int(eligible.sum())
    if not np.any(valid & ~eligible):
        anchor = int(rng.choice(np.flatnonzero(valid)))
        if m == 1:
            return modes, anchor, True
        # Uniformly reserving one of m lanes leaves (m-1)/m augmentation chances.
        mask_prob, noise_prob = mask_prob * m / (m - 1), noise_prob * m / (m - 1)
        if mask_prob + noise_prob > 1:
            raise ValueError("Requested probabilities cannot preserve a clean main view")
    for lane in np.flatnonzero(eligible):
        if lane != anchor:
            u = rng.random()
            modes[lane] = 1 if u < mask_prob else (2 if u < mask_prob + noise_prob else 0)
    return modes, anchor, False


def defer_block_augmentation(item, *, video_record, seed, epoch, config, lane):
    cfg = validate_config(config)
    if not cfg["enabled"]:
        raise ValueError("Block protection requires enabled train augmentation")
    views = {v["key"]: v for v in video_record["selected_views"]}
    main = views.get(item["camera_keys"][0], {})
    valid = (not bool(item["camera_is_pad"][0]) and item["camera_roles"][0] == "main"
             and bool(main.get("available")) and bool(main.get("path")))
    eligible = eligible_main_view(item["camera_is_pad"], item["camera_keys"],
                                  item["camera_roles"], video_record["selected_views"])
    item["_block_main_aug"] = dict(seed=int(seed), epoch=int(epoch), cfg=cfg,
                                    valid=valid, eligible=eligible, lane=int(lane))


def apply_block_augmentation(samples):
    pending = ["_block_main_aug" in item for item in samples]
    if not any(pending):
        return
    if not all(pending):
        raise ValueError("Cannot mix protected and legacy samples in one batch")
    groups = defaultdict(list)
    for item in samples:
        groups[item["block_id"]].append(item)
    for group in groups.values():
        group.sort(key=lambda item: item["_block_main_aug"]["lane"])
        meta = [item["_block_main_aug"] for item in group]
        if [m["lane"] for m in meta] != list(range(4)):
            raise ValueError("Block protection requires all four distinct lanes")
        first = meta[0]
        if any(any(m[k] != first[k] for k in ("seed", "epoch", "cfg")) for m in meta):
            raise ValueError("Block augmentation configuration/epoch mismatch")
        cfg = first["cfg"]
        identity = [first["epoch"], [
            [item["group_id"], item["physical_chunk_uid"], item["view_id"]]
            for item in group
        ]]
        modes, anchor, singleton = plan_block_modes(
            seed=first["seed"], identity=identity,
            valid=[m["valid"] for m in meta], eligible=[m["eligible"] for m in meta],
            mask_prob=cfg["mask_prob"], noise_prob=cfg["sensor_noise_prob"],
        )
        clean_count = sum(m["valid"] and mode == 0 for m, mode in zip(meta, modes))
        if not clean_count:
            raise RuntimeError("Block augmentation lost every clean main view")
        for lane, (item, m, mode) in enumerate(zip(group, meta, modes)):
            video = item["video"]
            if video.shape != (3, 3, 9, 256, 256):
                raise ValueError("Block augmentation requires three-view nine-frame video")
            mask, noise_type, params = torch.zeros(3, dtype=torch.bool), -1, torch.zeros(10)
            if mode:
                video = video.clone()
                if mode == 1:
                    mask[0] = True
                    video[0] = -1
                else:
                    rng = np.random.default_rng(int(digest([
                        first["seed"], BLOCK_POLICY, "noise", identity, lane]), 16))
                    noise_type = int(rng.choice(5, p=cfg["noise_type_weights"]))
                    rgb = ((video[0] + 1) / 2).permute(1, 0, 2, 3)
                    rgb, params = sensor_noise(rgb, noise_type, cfg=cfg, rng=rng)
                    video[0] = rgb.permute(1, 0, 2, 3) * 2 - 1
            item.update(video=video, main_aug_eligible=m["eligible"], main_aug_mode=int(mode),
                        augmentation_camera_mask=mask, main_noise_type=noise_type,
                        main_noise_params=params, main_aug_block_anchor=lane == anchor,
                        main_aug_block_valid_main=m["valid"], main_aug_block_lane=lane,
                        main_aug_block_clean_count=int(clean_count),
                        main_aug_block_singleton=singleton)
            del item["_block_main_aug"]
