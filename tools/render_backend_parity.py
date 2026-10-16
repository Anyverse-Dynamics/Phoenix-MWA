"""Compare direct versus subprocess rendering under identical seeds/actions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')


def child(args):
    import random
    import numpy as np
    import torch
    os.environ['EVAL_OUTPUT'] = str(args.output.resolve())
    os.environ['LIBERO_CONFIG_PATH'] = str(args.output.resolve()/'libero_config')
    from wam.posttrain.guard import install_import_guard
    install_import_guard()
    from eval.libero.bootstrap import setup_libero_paths
    setup_libero_paths()
    from eval.libero_plus import egl_rebind
    egl_rebind.install()
    args.output.mkdir(parents=True, exist_ok=True)
    from eval.libero_plus.runner import benchmark, initial_states
    from eval.libero.eval_libero_single import SubprocessLiberoEnv
    from libero.libero.envs import OffScreenRenderEnv
    import hashlib
    suites = benchmark()
    plan = json.loads(args.plan.read_text())
    records = {}
    for suite in ('libero_goal', 'libero_object', 'libero_spatial', 'libero_10'):
        row = next(r for r in plan if r['suite'] == suite)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        if args.mode == 'direct':
            env = OffScreenRenderEnv(bddl_file_name=row['bddl_file'], camera_heights=256,
                camera_widths=256, render_gpu_device_id=-1)
            env.seed(42)
        else:
            env = SubprocessLiberoEnv(task_suite_name=suite, task_id=row['task_id'],
                seed=42, render_gpu_device_id=-1, report_dir=str(args.output/suite))
        try:
            env.reset()
            env.set_init_state(initial_states(suites[suite], row['task_id'])[0])
            for step in range(33):
                action = [0,0,0,0,0,0,-1] if step < 30 else [.1,-.05,0,.02,0,0,-1]
                obs, reward, done, info = env.step(action)
                if step in (0,29,32):
                    for key in ('agentview_image', 'robot0_eye_in_hand_image',
                                'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
                        value = np.asarray(obs[key])
                        assert np.isfinite(value).all()
                        records[f'{suite}/{step}/{key}'] = hashlib.sha256(value.tobytes()).hexdigest()
        finally:
            env.close()
    (args.output/'observations.json').write_text(json.dumps(records, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('direct', 'subprocess'))
    args = parser.parse_args()
    if args.mode:
        child(args)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for mode in ('direct', 'subprocess'):
        subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
            '--plan', str(args.plan.resolve()), '--output', str(args.output/mode),
            '--mode', mode], check=True)
    before = json.loads((args.output/'direct/observations.json').read_text())
    after = json.loads((args.output/'subprocess/observations.json').read_text())
    result = {'passed': before == after and len(before) == 60, 'compared_fields': len(before),
        'image_pairs': 24, 'proprio_fields': 36,
        'scope': 'same seeded four initial states, 30 warmup + 3 controlled actions, compare steps 1/30/33',
        'mismatches': [key for key in before if before[key] != after.get(key)],
        'direct_sha256': before, 'subprocess_sha256': after}
    (args.output/'parity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
