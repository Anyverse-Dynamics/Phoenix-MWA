"""Bounded checkpoint reload, cache/current RGB parity and real joint denoising."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
import torch
from wam.inference.robocasa_joint96 import RoboCasaJoint96Policy
from wam.datasets.robocasa_posttrain import load_package
from wam.datasets.robocasa_visual_cache import (
    RoboCasaVisualCache, decode_episode, window_mapping, pack_rgb, compare, atomic_json, read_json)
from wam.posttrain.robocasa_paths import resolve_source


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'config', 'package', 'cache', 'selection', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--allow-smoke', action='store_true')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(a.device)
    package = load_package(a.package)
    selection = read_json(a.selection)
    if selection['package_sha256'] != package['sha256']:
        raise ValueError('Foreign smoke selection')
    keys = {tuple(selection['train'][0]), tuple(selection['val'][0])}
    episodes = [ep for ep in package['episodes'] if (ep['task'], ep['episode']) in keys]
    policy = RoboCasaJoint96Policy.from_checkpoint(a.checkpoint, a.config, a.package,
        device=a.device, allow_smoke=a.allow_smoke, execute_steps=12, num_inference_steps=10, seed=7)
    visual = RoboCasaVisualCache(a.cache, package['plan_path'])
    report = {'checkpoint': str(Path(a.checkpoint).resolve()), 'device': torch.cuda.get_device_name(),
              'torch': str(torch.__version__), 'mode': 'joint_video_actions', 'denoising_steps': 10,
              'full_dataset_certified': False, 'simulation_evaluated': False, 'samples': []}
    for ep in episodes:
        images, _ = decode_episode(ep)
        rgb = pack_rgb(images, window_mapping(ep['length'], 0))
        pad = torch.tensor([[False, True, True]], device=a.device)
        with torch.inference_mode():
            clip = policy.model.joint_visual_encoder.encode(rgb.to(a.device), camera_is_pad=pad).cpu()
            current = policy.model.joint_visual_encoder.encode(rgb[:, :, :, :1].to(a.device), camera_is_pad=pad).cpu()
        cached = visual.get(ep['task'], ep['episode'], 0)['video_latents'][None]
        checks = {}
        for branch, span in [('vae', slice(0,48)), ('stvae', slice(48,96))]:
            for label, left, right in [('cache_vs_online_clip', cached[:,span], clip[:,span]),
                                      ('clip_prefix_vs_single', clip[:,span,:1], current[:,span]),
                                      ('cache_prefix_vs_single', cached[:,span,:1], current[:,span])]:
                checks[branch + '_' + label] = compare(left, right, atol=.01, rtol=.01)
        raw_table = pq.read_table(resolve_source(ep['parquet']), columns=['observation.state'])
        raw = torch.tensor(raw_table['observation.state'][0].as_py(), dtype=torch.float32)[None]
        instruction = ep['prompt'].split('instruction: ', 1)[1]
        start = time.monotonic()
        actions = policy.predict(images[0], raw, instruction)
        finite = all(np.isfinite(v).all() for v in actions.values())
        if not finite or policy.last_canonical.shape != (32,80) or sum(v.shape[-1] for v in actions.values()) != 29:
            raise ValueError('Invalid Delta80 / absolute joint29 controller output')
        row = {'task': ep['task'], 'episode': ep['episode'], 'split': ep['split'],
               'checks': checks, 'finite_actions': finite, 'action_shapes': {k:list(v.shape) for k,v in actions.items()},
               'denoising_seconds': time.monotonic()-start}
        report['samples'].append(row)
        atomic_json(out / 'progress.json', report)
    report['passed'] = all(c['passed'] for row in report['samples'] for c in row['checks'].values())
    report['inference_passed'] = all(row['finite_actions'] for row in report['samples'])
    atomic_json(out / 'report.json', report)
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise ValueError('Measured cache/online parity failed; see report, do not claim acceptance')


if __name__ == '__main__':
    main()
