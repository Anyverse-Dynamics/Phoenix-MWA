"""Check GR1 registration, assets and offscreen rendering before policy startup."""
import argparse
import random
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-root', required=True)
    p.add_argument('--env', nargs='+', required=True)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    sys.path.append(str(Path(a.sim_root).resolve(strict=True)))
    import gymnasium as gym
    import numpy as np
    import robocasa
    import robosuite
    from Robocasa_tabletop.eval_files.simulation_env import SimulationInferenceEnv, SimulationConfig

    print('Simulator Python:', sys.executable, flush=True)
    print('robocasa:', robocasa.__file__, flush=True)
    print('robosuite:', robosuite.__file__, flush=True)
    for name in a.env:
        gym.spec(name)
    random.seed(a.seed)
    np.random.seed(a.seed)
    env = SimulationInferenceEnv().setup_environment(
        SimulationConfig(env_name=a.env[0], n_episodes=1, n_envs=1, seed=a.seed))
    try:
        obs, _ = env.reset(seed=a.seed)
        rgb = np.asarray(obs['video.ego_view_bg_crop_pad_res256_freq20'])
        if rgb.shape != (1, 1, 256, 256, 3) or rgb.dtype != np.uint8 or rgb.max() == rgb.min():
            raise ValueError(f'Invalid simulator camera frame: shape={rgb.shape}, dtype={rgb.dtype}')
        print(f'PREFLIGHT PASS: {len(a.env)} registered tasks; GR1 reset and nonblank RGB {rgb.shape}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
