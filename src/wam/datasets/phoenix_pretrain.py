"""Phoenix view-paired reads from an immutable stride-one sampling plan.

The plan controls ordering, explicit HABIT view selection and RH20T local start
offsets. Numeric projection, normalization, decoding and text caches keep their
accepted data contracts. There is no cross-sample or contrastive loss here.
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wam.datasets.interndata_multiview import (
    INTERNDATA_SOURCES,
    interndata_camera_variants,
)
from wam.datasets.local_robot import (
    LocalRobot80Dataset,
    augment_color,
    decode_views,
)
from wam.datasets.local_robot_main_view_aug import augment_main
from wam.datasets.local_robot_text import (
    LocalRobotTextCache,
    validate_language_augmentation_prob,
)
from wam.datasets.local_robot36_text import EpisodeLanguage3Cache
from wam.datasets.local_robot36_video import arx_video_targets
from wam.datasets.phoenix_robot import (
    SOURCE as RH20T_SOURCE,
    PhoenixRobotDataset,
    PhoenixRobotStateReader,
)


class MultiViewStateReader(PhoenixRobotStateReader):
    """Retain the complete clock for checks between paired observations."""

    def attach_metadata(self, item, table, record, profile):
        super().attach_metadata(item, table, record, profile)
        clock = (
            "source.timestamp_ms"
            if profile["source"] == RH20T_SOURCE
            else "timestamp"
        )
        times = np.asarray(table[clock].to_pylist(), dtype=np.float64)
        if times.shape != (33,) or not np.isfinite(times).all():
            raise ValueError("Multiview requires the complete finite 33-row clock")
        item["multiview_timestamps"] = torch.from_numpy(times.copy())
        item["multiview_clock"] = clock


def _check_tensor_equal(left, right, key, *, approximate=False):
    a, b = left[key], right[key]
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError(f"Paired {key} must be tensors")
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f"Paired {key} has different tensor geometry")
    equal = (
        torch.allclose(a, b, atol=2e-5, rtol=1e-4)
        if approximate
        else torch.equal(a, b)
    )
    if not equal:
        raise ValueError(f"Paired physical chunk disagrees on {key}")


def validate_paired_samples(samples):
    """Fail closed if a plan pairs different actions, clocks or task identities.

    Wrist availability is deliberately excluded: an external-view combination
    may have a valid wrist clip while another accepted combination lacks it.
    """
    groups = defaultdict(list)
    for sample in samples:
        if "pair_id" in sample:
            # The expected pair must remain comparable even when one corrupt
            # plan row advertises a different physical chunk identity.
            key = sample["block_id"], sample["pair_id"]
            groups[key].append(sample)
    for group in groups.values():
        if len(group) < 2:
            continue
        first = group[0]
        for other in group[1:]:
            for key in (
                "group_id",
                "physical_chunk_uid",
                "physical_episode_uid",
                "source_id",
                "profile_id",
                "stats_group",
                "task_index",
                "split",
                "multiview_clock",
            ):
                if first[key] != other[key]:
                    raise ValueError(f"Paired physical chunk disagrees on {key}")
            for key in ("action", "proprio"):
                _check_tensor_equal(first, other, key, approximate=True)
            for key in (
                "action_mask",
                "proprio_mask",
                "action_dim_is_pad",
                "proprio_dim_is_pad",
            ):
                _check_tensor_equal(first, other, key)
            # Absolute timestamps must agree, not just the relative frame grid.
            # In particular a large RH20T timestamp must not get relative slack.
            if not torch.allclose(
                first["multiview_timestamps"],
                other["multiview_timestamps"],
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError("Paired physical chunk has different timestamps")
            for key in ("embodiment_id", "color_params", "context_mask"):
                if key in first or key in other:
                    if key not in first or key not in other:
                        raise ValueError(f"Paired physical chunk is missing {key}")
                    _check_tensor_equal(first, other, key)
            for key in ("prompt", "language_variant", "rgb_variant", "rgb_variants_sha256"):
                if key in first or key in other:
                    if key not in first or key not in other or first[key] != other[key]:
                        raise ValueError(f"Paired physical chunk disagrees on {key}")


class PhoenixPretrainDataset(PhoenixRobotDataset):
    """Keep the accepted validation panel; opt training into paired view blocks."""

    def __init__(self, *args, multiview_plan_root=None, verify_pair_acceptance=True,
                 rgb_variants_root=None, rgb_variants_sha256=None,
                 rgb_selection_root=None, rgb_selection_sha256=None,
                 language_augmentation_prob=0.5, main_view_block_policy="none", **kwargs):
        from wam.datasets.local_robot_block_aug import BLOCK_POLICY
        if main_view_block_policy not in ("none", BLOCK_POLICY):
            raise ValueError("Unknown main-view block augmentation policy")
        self.main_view_block_policy = main_view_block_policy
        self.language_augmentation_prob = validate_language_augmentation_prob(
            language_augmentation_prob
        )
        super().__init__(*args, **kwargs)
        if main_view_block_policy != "none" and (
            self.split != "train" or not self.main_view_augmentation["enabled"]
        ):
            raise ValueError("Block protection requires enabled training augmentation")
        self.rgb_variants = None
        self.rgb_excluded_ids = frozenset()
        rgb_args = (rgb_variants_root, rgb_selection_root)
        if any(value is not None for value in rgb_args):
            if not all(value is not None for value in rgb_args) or multiview_plan_root is None:
                raise ValueError('RGB augmentation requires both pinned artifacts and the parent plan')
            from wam.datasets.phoenix_rgb_variants import RGBVariants, load_selection
            from wam.runtime_contracts import recorded_manifest_id
            self.rgb_variants = RGBVariants(rgb_variants_root,
                expected_sha256=rgb_variants_sha256, package_sha256=self.package.fingerprint)
            selection, _ = load_selection(rgb_selection_root,
                expected_sha256=rgb_selection_sha256, package_sha256=self.package.fingerprint,
                parent_plan_sha256=recorded_manifest_id(multiview_plan_root, '_PLAN_COMPLETE'))
            if set(selection['excluded_roots']) != self.rgb_variants.excluded_roots:
                raise ValueError('RGB variants and sample selection exclude different roots')
            self.rgb_excluded_ids = frozenset(selection['excluded_episode_ids'])
        if self.split != "train":
            if self.rgb_variants is not None:
                kept = [(step, panel) for step, panel in zip(self.eval_steps, self.fixed_validation_panel)
                        if int(step[1]) not in self.rgb_excluded_ids]
                self.eval_steps = [step for step, _ in kept]
                self.fixed_validation_panel = [dict(panel, index=i, seed=self.seed+i)
                                               for i, (_, panel) in enumerate(kept)]
                if not self.eval_steps:
                    raise ValueError('RGB exclusions removed the entire validation panel')
            return
        if self.coverage_mode or self.coverage_steps or self._rh_positions is not None:
            raise ValueError("Multiview planning cannot use the old forced smoke sampler")
        if multiview_plan_root is None:
            raise ValueError("Training requires a frozen multiview_plan_root")
        from wam.datasets.multiview_plan import MultiViewPlanSampler, validate_pair_acceptance

        if verify_pair_acceptance:
            validate_pair_acceptance(multiview_plan_root)

        self.mixture = MultiViewPlanSampler(
            multiview_plan_root,
            package=self.package,
            batch_size=self.batch_size,
            seed=self.seed,
            rgb_selection_root=rgb_selection_root,
            rgb_selection_sha256=rgb_selection_sha256,
        )
        self.reader = MultiViewStateReader(self.reader.limit >> 20)
        # LocalRobot80Dataset.__getstate__ recreates this same reader in workers.
        self.reader_class = MultiViewStateReader
        self.sampling_contract = copy.deepcopy(self.mixture.sampling_contract)
        if self.rgb_variants is not None:
            self.sampling_contract['rgb_variants'] = self.rgb_variants.contract

    def read_window(self, episode_id, start, **kwargs):
        if int(episode_id) in getattr(self, 'rgb_excluded_ids', ()):
            raise ValueError('Excluded RGB episode reached the reader')
        return super().read_window(episode_id, start, **kwargs)

    def __getitem__(self, index):
        if self.split != "train":
            return super().__getitem__(index)
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        sample = self.mixture.sample(int(index))
        if getattr(self, "main_view_block_policy", "none") != "none":
            sample["block_lane"] = int(index) % 4
        return self.read_planned_window(sample)

    def read_planned_window(
        self,
        sample,
        *,
        with_video=True,
        with_text=True,
        normalize=True,
    ):
        episode_id, start = int(sample["episode_id"]), int(sample["start"])
        if episode_id in getattr(self, 'rgb_excluded_ids', ()):
            raise ValueError('Excluded RGB episode reached the sampling plan')
        episode = self.package.episode(episode_id)
        profile = copy.deepcopy(self.package.manifest["profiles"][episode["profile"]])
        record = episode["record"]
        item = LocalRobot80Dataset.read_window(
            self,
            episode_id,
            start,
            with_video=False,
            with_text=False,
            normalize=normalize,
        )
        targets = item.pop("_video_timestamps", None)
        variants = record.get("camera_variants")
        policy = getattr(self.mixture, "manifest", {}).get("interndata_multiview")
        if policy and profile["source"] in INTERNDATA_SOURCES:
            variants = interndata_camera_variants(record)
        variant_index = int(sample.get("view_variant", -1))
        if variants:
            if not 0 <= variant_index < len(variants):
                raise ValueError("A multiview plan must explicitly name a camera variant")
            variant = variants[variant_index]
            profile.update(
                cameras=variant["cameras"],
                camera_roles=variant["camera_roles"],
            )
            record = dict(
                record,
                selected_views=variant["selected_views"],
                t5_task_bindings=variant["t5_task_bindings"],
            )
        elif variant_index != -1:
            raise ValueError("Only an episode with camera_variants accepts view_variant")

        if profile["source"] == RH20T_SOURCE:
            wrist_valid = item.pop("_rh_wrist_valid")
            if not wrist_valid:
                profile["cameras"][2] = ""
                profile["camera_roles"][2] = "pad"
        item.update(
            camera_keys=tuple(profile["cameras"]),
            camera_roles=tuple(profile["camera_roles"]),
            camera_is_pad=torch.tensor(
                [not key for key in profile["cameras"]], dtype=torch.bool
            ),
            group_id=str(sample["group_id"]),
            physical_chunk_uid=str(sample["physical_chunk_uid"]),
            view_id=str(sample["view_id"]),
            view_variant=variant_index,
            block_id=int(sample["block_id"]),
            pair_id=str(sample["pair_id"]),
            is_coverage_padding=bool(sample["is_coverage_padding"]),
        )
        # Neither local frame_index nor view-specific sample_id can identify a
        # physical RH20T chunk after different leading-frame trims.
        identity = [
            int(self.mixture.epoch),
            item["group_id"],
            item["physical_chunk_uid"],
        ]
        video_record = record
        if getattr(self, 'rgb_variants', None) is not None:
            video_record, item['rgb_variant'] = self.rgb_variants.select(
                record, seed=self.seed, identity=identity)
            item['rgb_variants_sha256'] = self.rgb_variants.fingerprint
            explicit_clock = targets is not None or profile['source'] == 'arx456_xt_motion_dedup_20260916'
            if explicit_clock and item['rgb_variant'] == 'direct_resize':
                raise ValueError('Explicit-clock RGB sources require a separately validated time mapping')
        if with_video:
            if profile["source"] == "arx456_xt_motion_dedup_20260916":
                targets = arx_video_targets(record, start)
            video, item["camera_is_pad"] = decode_views(
                video_record, profile, start, targets_by_key=targets
            )
            video, item["color_params"] = augment_color(
                video,
                item["camera_is_pad"],
                seed=self.seed,
                identity=identity,
                enabled=self.color_augmentation,
            )
            if getattr(self, "main_view_block_policy", "none") != "none":
                from wam.datasets.local_robot_block_aug import defer_block_augmentation
                item["video"] = video
                defer_block_augmentation(
                    item, video_record=video_record, seed=self.seed,
                    epoch=self.mixture.epoch, config=self.main_view_augmentation,
                    lane=sample["block_lane"],
                )
            else:
                item["video"], metadata = augment_main(
                    video,
                    item["camera_is_pad"],
                    camera_keys=item["camera_keys"],
                    camera_roles=item["camera_roles"],
                    selected_views=video_record["selected_views"],
                    seed=self.seed,
                    identity=identity,
                    config=self.main_view_augmentation,
                    training=True,
                )
                item.update(metadata)
        if with_text:
            item.update(self._resolve_planned_text(record, profile, item, identity))
        return item

    def _resolve_planned_text(self, record, profile, item, identity):
        if self.text_pid != os.getpid():
            self.text_caches, self.text_pid = {}, os.getpid()
        binding = record.get("t5_episode_binding")
        if binding is None:
            matches = [
                candidate
                for candidate in record["t5_task_bindings"]
                if int(candidate["task_index"]) == item["task_index"]
            ]
            if len(matches) != 1:
                raise ValueError("Multiview sample has no unique task binding")
            binding = matches[0]
        is_rh20t = profile["source"] == RH20T_SOURCE
        if is_rh20t and (
            len(record["t5_task_bindings"]) != 1 or binding["variant_count"] != 1
        ):
            raise ValueError("RH20T must retain its single original task route")
        provider = RH20T_SOURCE if is_rh20t else binding["provider_id"]
        key = binding["registry"], provider, binding["registry_sha256"]
        if key not in self.text_caches:
            cache_class = (
                EpisodeLanguage3Cache
                if binding.get("route_kind") == "episode"
                else LocalRobotTextCache
            )
            self.text_caches[key] = cache_class(*key)
        language_options = {}
        if not is_rh20t and binding.get("route_kind") != "episode":
            language_options["augmentation_prob"] = self.language_augmentation_prob
        return self.text_caches[key].resolve(
            binding,
            seed=self.seed,
            identity=identity,
            augment=self.language_augmentation and not is_rh20t,
            **language_options,
        )

    @staticmethod
    def collate_fn(samples):
        validate_paired_samples(samples)
        from wam.datasets.local_robot_block_aug import apply_block_augmentation
        apply_block_augmentation(samples)
        return LocalRobot80Dataset.collate_fn(samples)
