"""Read-only frozen-input validation without file fingerprints.

Scene state and joints must agree within 1e-6; RGB is reported separately because
different rendering hardware need not be pixel-identical. Seeds and instructions
are exact, and replay never replaces a seed or executes an expert trajectory.
"""
import json
import os
from pathlib import Path
import numpy as np
from eval.robotwin.scene_state import scene_state, plain


def compare(a, b, key='scene'):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError(f'Frozen scene keys differ: {key}')
        for k in a: compare(a[k], b[k], key+'.'+k)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError(f'Frozen scene length differs: {key}')
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, f'{key}[{i}]')
    elif isinstance(a, float):
        if not np.isclose(a, b, rtol=0, atol=1e-6):
            raise ValueError(f'Frozen scene value differs: {key}: {a} != {b}')
    elif a != b:
        raise ValueError(f'Frozen scene value differs: {key}: {a} != {b}')


def freeze_or_verify(env, task, index, seed, instruction, info):
    folder = Path(os.environ['WAM_RW_INPUT_SET']) / task / f'episode_{index:02d}'
    saved = json.loads((folder / 'scene.json').read_text())
    current = dict(task=task, index=index, seed=seed, instruction=instruction,
                   scene_info=plain(info), scene_state=scene_state(env))
    compare({k: saved[k] for k in current}, current)
    obs = env.get_obs()
    with np.load(folder / 'observation.npz', allow_pickle=False) as original:
        np.testing.assert_allclose(original['joint'], obs['joint_action']['vector'], rtol=0, atol=1e-6)
        rgb = {}
        for cam in ('head_camera', 'left_camera', 'right_camera'):
            actual = np.asarray(obs['observation'][cam]['rgb'])
            expected = original[cam]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError(f'Frozen RGB geometry differs: {cam}')
            rgb[cam] = dict(exact=bool(np.array_equal(actual, expected)),
                            mean_absolute_pixel_error=float(np.abs(actual.astype(float)-expected).mean()))
    return obs, dict(frozen_scene_verified=True, scene_atol=1e-6, joint_verified=True,
                     rgb_comparison=rgb, file_hash_validation=False)
