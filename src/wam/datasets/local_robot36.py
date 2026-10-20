"""Opt-in raw36 dataset; baseline reader/augmentations remain unchanged."""
from __future__ import annotations

import os

import numpy as np
import torch

from wam.datasets.local_robot import LocalRobot80Dataset, NativeStateReader, augment_color, decode_views
from wam.datasets.local_robot_contracts import digest
from wam.datasets.local_robot_main_view_aug import augment_main, validate_config
from wam.datasets.local_robot_text import LocalRobotTextCache
from wam.datasets.local_robot36_contracts import Robot36Projector
from wam.datasets.local_robot36_package import Robot36Package
from wam.datasets.local_robot36_text import EpisodeLanguage3Cache
from wam.datasets.local_robot36_video import (
    ABC_CAMERA_MAX_OFFSET_NS, ABC_TIMESTAMP_POLICY, abc_camera_bounds, arx_video_targets,
)


class Robot36StateReader(NativeStateReader):
    projector_class = Robot36Projector

    def extra_columns(self, record, profile):
        if profile['source'] != 'abc130k':
            return []
        return ['source.timestamp_ns'] + [v['timestamp_column'] for v in record['selected_views']]

    def attach_metadata(self, item, table, record, profile):
        if profile['source'] != 'abc130k':
            return
        reference = np.asarray(table['source.timestamp_ns'].to_pylist(), dtype=np.int64)
        if (np.any(np.diff(reference) <= 0)
                or np.any(reference < int(record['original_common_from_ns']))
                or np.any(reference > int(record['original_common_to_ns']))):
            raise ValueError(f"ABC reference clock outside episode: episode={record.get('episode_index')}")
        targets = {}
        for view in record['selected_views']:
            if view.get('timestamp_policy') != ABC_TIMESTAMP_POLICY:
                raise ValueError('unreviewed ABC camera clock mapping')
            abc_camera_bounds(record, view)
            stamps = np.asarray(table[view['timestamp_column']].to_pylist(), dtype=np.int64)
            if np.any(stamps <= 0) or np.any(np.diff(stamps) < 0) or np.any(abs(stamps-reference) > ABC_CAMERA_MAX_OFFSET_NS):
                raise ValueError(f"ABC native camera timestamp alignment exceeds 50ms: "
                                 f"episode={record.get('episode_index')} camera={view['key']}")
            targets[view['key']] = ((stamps[::4] - int(view['reference_origin_ns'])) / 1e9
                                    + float(view['start_s']))
        item['_video_timestamps'] = targets

    def validate_timestamps(self, times, record, profile):
        policy = profile.get("timestamp_policy", {"mode": "native_uniform"})
        if policy["mode"] == "native_uniform":
            return super().validate_timestamps(times, record, profile)
        if policy["mode"] != "reviewed_monotonic" or not policy.get("review_reference"):
            raise ValueError("unreviewed Robot36 timestamp mode")
        if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
            raise ValueError("reviewed native timestamps must remain strictly increasing")


class Robot36Dataset(LocalRobot80Dataset):
    package_class = Robot36Package
    reader_class = Robot36StateReader

    def __init__(self, *args, main_view_augmentation=None, coverage_mode=False, coverage_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_view_augmentation = validate_config(main_view_augmentation)
        self.coverage_mode = bool(coverage_mode)
        self.coverage_steps = int(coverage_steps)
        if self.split != "train" and self.main_view_augmentation["enabled"]:
            raise ValueError("validation cannot enable main-view augmentation")
        if self.coverage_steps < 0:
            raise ValueError('coverage_steps cannot be negative')
        if (self.coverage_mode or self.coverage_steps) and self.package.manifest["purpose"] != "audit":
            raise ValueError("forced profile coverage is restricted to audit packages")

    def read_window(self, episode_id, start, *, draw=0, with_video=True, with_text=True, normalize=True):
        item = super().read_window(episode_id, start, draw=draw, with_video=False,
                                   with_text=False, normalize=normalize)
        targets = item.pop('_video_timestamps', None)
        ep = self.package.episode(episode_id)
        profile = self.package.manifest["profiles"][ep["profile"]]
        record = ep["record"]
        identity = [self.mixture.epoch if self.split == "train" else 0, record["sample_id"], int(start), int(draw)]
        # HABIT variants are selected after the physical episode/window draw.
        variants = record.get("camera_variants")
        if variants:
            selected = int(digest([self.seed, "camera_variant", identity]), 16) % len(variants) if self.split == "train" else 0
            view = variants[selected]
            profile = dict(profile, cameras=view["cameras"], camera_roles=view["camera_roles"])
            record = dict(record, selected_views=view["selected_views"], t5_task_bindings=view["t5_task_bindings"])
            item.update(camera_keys=tuple(profile["cameras"]), camera_roles=tuple(profile["camera_roles"]))
        if with_video:
            if profile['source'] == 'arx456_xt_motion_dedup_20260916':
                targets = arx_video_targets(record, int(start))
            video, item["camera_is_pad"] = decode_views(record, profile, int(start), targets_by_key=targets)
            video, item["color_params"] = augment_color(video, item["camera_is_pad"], seed=self.seed,
                                                        identity=identity, enabled=self.color_augmentation)
            item["video"], metadata = augment_main(video, item["camera_is_pad"],
                camera_keys=item["camera_keys"], camera_roles=item["camera_roles"], selected_views=record["selected_views"],
                seed=self.seed, identity=identity, config=self.main_view_augmentation, training=self.split == "train")
            item.update(metadata)
        if with_text:
            if self.text_pid != os.getpid():
                self.text_caches, self.text_pid = {}, os.getpid()
            binding = record.get("t5_episode_binding")
            if binding is None:
                matches = [b for b in record["t5_task_bindings"] if int(b["task_index"]) == item["task_index"]]
                if len(matches) != 1:
                    raise ValueError("Robot36 has no unique task binding")
                binding = matches[0]
            key = binding["registry"], binding["provider_id"], binding["registry_sha256"]
            if key not in self.text_caches:
                cls = EpisodeLanguage3Cache if binding["route_kind"] == "episode" else LocalRobotTextCache
                self.text_caches[key] = cls(*key)
            item.update(self.text_caches[key].resolve(binding, seed=self.seed, identity=identity,
                                                    augment=self.language_augmentation))
        return item

    def __getitem__(self, index):
        forced = self.coverage_mode or (self.split == 'train' and int(index) <
            self.coverage_steps * self.batch_size * self.mixture.bucket_wave_world_size)
        if not forced:
            return super().__getitem__(index)
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        ds = self.datasets[(int(index) // 4) % len(self.datasets)]
        episode, start = ds._global_to_episode_frame((int(index) // (4 * len(self.datasets)) * 4 + int(index) % 4) % len(ds))
        return self.read_window(episode, start, draw=int(index))
