"""Phoenix robot reader over the frozen Robot36 + RH20T v6 data contract."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from wam.resources import resource_path
from wam.pretrain_compat import bounded_smoke_enabled

import numpy as np
import torch

from wam.datasets.local_robot import LocalRobot80Dataset, augment_color, decode_views
from wam.datasets.local_robot_contracts import LocalRobotProjector, digest, sha256, profile_identity
from wam.datasets.local_robot_package import LocalRobotPackage, read_json, unpack
from wam.datasets.local_robot36 import Robot36Dataset, Robot36StateReader
from wam.datasets.local_robot36_contracts import Robot36Projector, _ee
from wam.datasets.local_robot36_package import Robot36Package
from wam.datasets.local_robot_main_view_aug import augment_main
from wam.datasets.local_robot_raw_bi_contracts import RawBIProjector, _component, I
from wam.datasets.local_robot_text import LocalRobotTextCache

SCHEMA = 'phoenix.local_robot37.rh20t_v6.v1'
RH_SCHEMA = SCHEMA + '.eef'
SOURCE = 'rh20t_v6_20261003'
ROBOT_GROUP = {1:'flexiv', 2:'flexiv', 3:'ur5', 4:'ur5', 5:'panda', 6:'kuka', 7:'kuka'}


def remap_habit_media(record, aliases):
    """Use hash-verified resize-only media while preserving numeric/T5 identity."""
    if record.get('source_id') != 'habit' or not aliases:
        return record
    result = copy.deepcopy(record)
    old_root = result['source_root']
    if old_root not in aliases:
        raise ValueError('HABIT resize alias missing')
    result['numeric_source_root'] = old_root
    result['source_root'] = aliases[old_root]
    for views in [result['selected_views'], *[v['selected_views'] for v in result['camera_variants']]]:
        for view in views:
            original = view['source_root']
            if original not in aliases:
                raise ValueError('HABIT variant resize alias missing')
            relative = Path(view['path']).relative_to(original)
            view.update(source_root=aliases[original], path=str(Path(aliases[original])/relative))
    return result


def rh_profile(cfg):
    if cfg not in range(1, 8):
        raise ValueError('RH20T cfg must be 1..7')
    ee = _ee('state', 'state.future', [0, 6], 40, 'xyz_rpy',
             semantics='next_observation_target')
    ee['action_target_offset'] = 1
    gripper = _component('gripper.state', 'gripper.state.future', [0, 1], [56, 57],
                         target_offset=1, semantics='next_observation_target')
    p = dict(schema=RH_SCHEMA, source=SOURCE, body=f'cfg{cfg}', cfg=cfg,
        selection_entry='D37', stats_group=f'raw37/rh20t_v6/{ROBOT_GROUP[cfg]}', native_stride=1,
        cameras=['observation.images.primary', '', 'observation.images.wrist'],
        camera_roles=['main', 'pad', 'right_wrist'], b_policy='user_confirmed_identity_all_cfg',
        adapter=dict(dim=80, action_translation_frame='source',
            state_pose_output='xyz_rot6d', action_pose_output='xyz_rotvec',
            action_pose_semantics='absolute_target', slots=dict(right=dict(ee=ee, gripper=gripper)),
            frame_transform=dict(world_from_source=I, source_ee_from_canonical_ee=I)),
        timestamp_policy=dict(mode='native_uniform'),
        units=dict(position='m', rotation='rad', gripper='native_per_robot_type_unverified',
                   grouping='user_confirmed_4_arm_types_including_merged_UR5_grippers'),
        origin_policy='preserve_corrected_v6_tcp_no_extra_offset_or_encoding_correction')
    p['id'] = profile_identity(p, f'rh20t_v6_cfg{cfg}_')
    return p


class RH20TProjector(RawBIProjector):
    def __init__(self, profile):
        if profile != rh_profile(profile.get('cfg')):
            raise ValueError('RH20T v6 profile drift')
        LocalRobotProjector.__init__(self, profile)
        self.profile = copy.deepcopy(profile)
        self.a = torch.eye(3)

    @property
    def columns(self):
        return sorted(set(super().columns + ['mask.gripper', 'mask.gripper_action_valid',
            'mask.action_valid', 'mask.chunk32_start', 'mask.wrist_valid']))

    def project_rows(self, data, indices=None, kind='state', next_indices=None):
        if indices is None:
            indices = list(range(self._length(data)))
        out, pad, valid = super().project_rows(data, indices, kind, next_indices)
        ix = torch.as_tensor(indices, dtype=torch.long)
        key = 'mask.gripper' if kind == 'state' else 'mask.gripper_action_valid'
        valid[:, 56] &= torch.as_tensor(data[key]).reshape(-1).bool()[ix]
        if kind == 'action':
            valid &= torch.as_tensor(data['mask.action_valid']).reshape(-1).bool()[ix, None]
        out[~valid] = 0
        return out, pad, valid

    def project_window(self, data):
        if not bool(torch.as_tensor(data['mask.chunk32_start']).reshape(-1)[0]):
            raise ValueError('RH20T start is not admitted by v6')
        if not torch.as_tensor(data['mask.action_valid']).reshape(-1)[:32].all():
            raise ValueError('RH20T window contains a terminal/invalid action')
        if not torch.equal(data['state.future'][:32], data['state'][1:33]):
            raise ValueError('RH20T future state no longer matches adjacent v6 row')
        return super().project_window(data)


def projector(profile):
    return RH20TProjector(profile) if profile['schema'] == RH_SCHEMA else Robot36Projector(profile)


class PhoenixRobotPackage(LocalRobotPackage):
    def __init__(self, root, *, audit=False, verify_index=True):
        self.root = resource_path(root).resolve()
        self.manifest = read_json(self.root/'manifest.json')
        from wam.runtime_contracts import recorded_manifest_id, require_file
        self.fingerprint = recorded_manifest_id(self.root)
        if self.manifest['schema'] != SCHEMA:
            raise ValueError('Phoenix robot schema/manifest drift')
        if not audit and self.manifest['purpose'] != 'training_candidate':
            raise ValueError('Phoenix audit package is not a training package')
        require_file(self.root / 'episodes.sqlite')
        base = self.manifest['base_package']
        self.base = Robot36Package(base['root'], verify_index=verify_index)
        if self.base.fingerprint != base['manifest_sha256']:
            raise ValueError('Frozen Robot36 base drift')
        self._db, self._pid = None, None

    @lru_cache(maxsize=32)
    def _root(self, root_id):
        return unpack(self.db.execute('SELECT payload FROM roots WHERE id=?', (root_id,)).fetchone()[0])

    @lru_cache(maxsize=32)
    def episode(self, episode_id):
        row = self.db.execute('SELECT profile,split,unit,ranges,record,root_id,base_episode_id FROM episodes WHERE id=?',
                              (int(episode_id),)).fetchone()
        if row is None:
            raise KeyError(episode_id)
        profile, split, unit, ranges, record, rid, base_id = row
        if base_id is not None:
            original = self.base.episode(base_id)
            if (original['profile'], original['split'], original['unit'], original['ranges']) != (
                    profile, split, unit, unpack(ranges)):
                raise ValueError('Inherited Robot36 membership changed')
            if self.manifest.get('habit_resize256'):
                original = dict(original, record=remap_habit_media(original['record'], self.manifest['habit_resize256']['aliases']))
            return dict(original, id=int(episode_id))
        return dict(id=int(episode_id), profile=profile, split=split, unit=unit,
                    ranges=unpack(ranges), record=unpack(record), root=self._root(rid))

    def close(self):
        super().close()
        self.base.close()


class PhoenixRobotStateReader(Robot36StateReader):
    projector_class = staticmethod(projector)

    def extra_columns(self, record, profile):
        if profile['source'] == SOURCE:
            return ['action', 'source.timestamp_ms']
        return super().extra_columns(record, profile)

    def attach_metadata(self, item, table, record, profile):
        super().attach_metadata(item, table, record, profile)
        if profile['source'] == SOURCE:
            times = np.asarray(table['source.timestamp_ms'].to_pylist(), dtype=np.float64)
            if not np.array_equal(np.diff(times), np.full(32, 100.0)):
                raise ValueError('RH20T v6 window is not the native 100ms grid')
            stored = torch.tensor(table['action'].to_pylist(), dtype=torch.float32)[:32]
            if not torch.allclose(item['action'][:, 47:53], stored, atol=2e-5, rtol=1e-4):
                raise ValueError('Recomputed EEF motion differs from corrected v6 action')
            # The model's view mask is clip-level: conservatively hide the whole
            # wrist clip if any of its nine selected frames is invalid.
            item['_rh_wrist_valid'] = (not record['wrist_black_placeholder'] and
                bool(np.asarray(table['mask.wrist_valid'].to_pylist(), dtype=bool)[::4].all()))


class PhoenixRobotDataset(Robot36Dataset):
    package_class = PhoenixRobotPackage
    reader_class = PhoenixRobotStateReader

    def __init__(self, *args, coverage_mode=False, coverage_steps=0, rh_only=False, **kwargs):
        # Only the explicit bounded training entry can force full-index coverage.
        super().__init__(*args, coverage_mode=False, coverage_steps=0, **kwargs)
        if (coverage_mode or coverage_steps) and (not bounded_smoke_enabled()
                                                  or not kwargs.get('allow_smoke_stats')):
            raise ValueError('Forced coverage requires the explicit Phoenix smoke gate')
        self.coverage_mode, self.coverage_steps = bool(coverage_mode), int(coverage_steps)
        if self.coverage_steps < 0 or self.split != 'train' and (coverage_mode or coverage_steps):
            raise ValueError('Invalid coverage settings')
        if rh_only:
            if not bounded_smoke_enabled():
                raise ValueError('RH-only selection is smoke-only')
            self._rh_positions = [i for i, ds in enumerate(self.datasets) if ds.profile['source'] == SOURCE]
            if self.split == 'validation':
                self.eval_steps = [r for r in self.eval_steps if r[0] in self._rh_positions]
                self.fixed_validation_panel = [dict(index=j, source=SOURCE, seed=self.seed+j,
                    episode_id=int(eid), start=int(start)) for j, (_, eid, start) in enumerate(self.eval_steps)]
        else:
            self._rh_positions = None

    def read_window(self, episode_id, start, *, draw=0, with_video=True, with_text=True, normalize=True):
        ep = self.package.episode(episode_id)
        p = self.package.manifest['profiles'][ep['profile']]
        if p['source'] != SOURCE:
            return super().read_window(episode_id, start, draw=draw, with_video=with_video,
                                      with_text=with_text, normalize=normalize)
        item = LocalRobot80Dataset.read_window(self, episode_id, start, draw=draw,
                            with_video=False, with_text=False, normalize=normalize)
        wrist = item.pop('_rh_wrist_valid')
        p = copy.deepcopy(p)
        if not wrist:
            p['cameras'][2], p['camera_roles'][2] = '', 'pad'
        item.update(camera_keys=tuple(p['cameras']), camera_roles=tuple(p['camera_roles']),
                    camera_is_pad=torch.tensor([False, True, not wrist]))
        record = ep['record']
        identity = [self.mixture.epoch if self.split == 'train' else 0, record['sample_id'], int(start), int(draw)]
        if with_video:
            video, item['camera_is_pad'] = decode_views(record, p, int(start))
            video, item['color_params'] = augment_color(video, item['camera_is_pad'], seed=self.seed,
                                                       identity=identity, enabled=self.color_augmentation)
            item['video'], metadata = augment_main(video, item['camera_is_pad'],
                camera_keys=item['camera_keys'], camera_roles=item['camera_roles'],
                selected_views=record['selected_views'], seed=self.seed, identity=identity,
                config=self.main_view_augmentation, training=self.split == 'train')
            item.update(metadata)
        if with_text:
            binding = record['t5_task_bindings'][0]
            if binding['task_index'] != item['task_index'] or binding['variant_count'] != 1:
                raise ValueError('RH20T must use its single original physical-task route')
            if self.text_pid != os.getpid():
                self.text_caches, self.text_pid = {}, os.getpid()
            key = binding['registry'], SOURCE, binding['registry_sha256']
            if key not in self.text_caches:
                self.text_caches[key] = LocalRobotTextCache(*key)
            item.update(self.text_caches[key].resolve(binding, seed=self.seed, identity=identity, augment=False))
        return item

    def __getitem__(self, index):
        if self._rh_positions is not None and self.split == 'train':
            if not 0 <= int(index) < len(self):
                raise IndexError(index)
            ds = self.datasets[self._rh_positions[(int(index)//4) % len(self._rh_positions)]]
            flat = (int(index)//(4*len(self._rh_positions))*4 + int(index)%4) % len(ds)
            eid, start = ds._global_to_episode_frame(flat)
            return self.read_window(eid, start, draw=int(index))
        return super().__getitem__(index)
