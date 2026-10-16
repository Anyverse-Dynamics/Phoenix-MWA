import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

from wam import robocasa_simulation as runtime
from test_robocasa_scene_seed import FakeRoboCasa


SEEDS = np.random.default_rng(7).integers(0, 2**32, size=50).tolist()


class SeedListTests(unittest.TestCase):
    def make_env(self, seeds=None, **kwargs):
        seeds = seeds or SEEDS[:5]
        return runtime.SeedListRoboCasaEnv(FakeRoboCasa(seeds[0]), episode_seeds=seeds, **kwargs)

    def test_replay_independent_of_bootstrap_seed_and_previous_random_draws(self):
        expected = [self.make_env([seed]).reset()[0] for seed in SEEDS[:5]]
        env = self.make_env()
        for i, reference in enumerate(expected):
            env.env.env.rng.random(i * 91)
            np.random.random(i * 31)
            random.random()
            actual, info = env.reset(seed=7 if i == 0 else None)
            np.testing.assert_array_equal(actual, reference)
            self.assertEqual(info['scene_seed'], SEEDS[i])
        self.assertEqual(env.env.received_seeds, SEEDS[:5])

    def test_vector_autoreset_and_cleanup_are_not_extra_evaluated_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audit.jsonl'
            env = self.make_env(audit_path=path)
            vector = gym.vector.SyncVectorEnv([lambda: env])
            with patch.object(runtime, 'scene_fingerprint', side_effect=lambda e, o: {'backend_seed': e.unwrapped.env.seed}):
                vector.reset(seed=7)
                for i, seed in enumerate(SEEDS[:5]):
                    self.assertEqual(env.scene_seed, seed)
                    vector.step(np.zeros((1, 1)))
                    # Gymnasium >=1 defaults to NEXT_STEP; older versions reset
                    # on the terminal step. Exercise the installed lifecycle.
                    if i < 4 and env.scene_seed == seed:
                        vector.step(np.zeros((1, 1)))
                vector.reset()
            with self.assertRaisesRegex(RuntimeError, 'exhausted'):
                env.step(np.zeros(1))
            audit = [json.loads(line) for line in path.read_text().splitlines()]
            evaluated = [r for r in audit if r['evaluation_episode']]
            self.assertEqual([r['scene_seed'] for r in evaluated], SEEDS[:5])
            self.assertEqual([r['episode_index'] for r in evaluated], list(range(5)))
            self.assertEqual([r['backend_seed'] for r in evaluated], SEEDS[:5])
            self.assertIn(sum(not r['evaluation_episode'] for r in audit), (1, 2))
            vector.close()

    def test_manifest_validation_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'seeds.json'
            manifest = {'protocol': runtime.SEED_LIST_PROTOCOL, 'episode_seeds': SEEDS}
            path.write_text(json.dumps(manifest))
            loaded = runtime.load_scene_seed_list(path, 5)
            self.assertEqual(loaded['episode_seeds'], SEEDS[:5])
            self.assertEqual(loaded['manifest_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'too short'):
                runtime.load_scene_seed_list(path, 51)
            for values in ([], [True], [-1], [2**32], [1.2], ['7']):
                path.write_text(json.dumps({**manifest, 'episode_seeds': values}))
                with self.assertRaises(ValueError):
                    runtime.load_scene_seed_list(path, 1)

    def test_factory_seeds_backend_constructor_and_preserves_policy_hash_seed(self):
        config = SimpleNamespace(seed=7, n_envs=1, n_episodes=5, env_name='test')
        fake = FakeRoboCasa(SEEDS[0])
        with patch.dict(os.environ, PYTHONHASHSEED='7'), patch.object(runtime.gym, 'make', return_value=fake) as make, \
                patch.object(runtime, '_wrap_simulation_env', side_effect=lambda e, c: e):
            env = runtime.make_seed_list_env(config, 0, episode_seeds=SEEDS[:5])
        make.assert_called_once_with('test', enable_render=True, seed=SEEDS[0])
        env.reset(seed=7)
        self.assertEqual(fake.env.seed, SEEDS[0])

    def test_actual_client_selects_list_keeps_policy_seed_and_records_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'results'
            path = Path(tmp) / 'seeds.json'
            path.write_text(json.dumps({'protocol': runtime.SEED_LIST_PROTOCOL, 'episode_seeds': SEEDS}))
            client = Mock()
            client.get_server_metadata.return_value = {
                'policy': 'robocasa_delta80_slot18_joint_v1', 'execute_steps': 12,
                'inference_mode': 'joint_video_actions', 'seed': 7,
                'seed_mode': 'fixed_per_replan', 'embodiment_id': 18, 'action_dim': 80}
            runner = Mock(env=None)
            runner.run_simulation.return_value = ['test', [False] * 5]
            modules = {'deployment.model_server.tools.websocket_policy_client': SimpleNamespace(
                WebsocketClientPolicy=Mock(return_value=client)),
                'Robocasa_tabletop.eval_files.simulation_env': SimpleNamespace(
                    SimulationConfig=SimpleNamespace, MultiStepConfig=SimpleNamespace)}
            script = Path(__file__).parents[1] / 'tools/eval_robocasa_joint96.py'
            with patch.dict(sys.modules, modules), patch.dict(os.environ), patch.object(sys, 'path', list(sys.path)), \
                    patch.object(sys, 'argv', [str(script), '--output', str(output), '--env', 'test',
                        '--episodes', '5', '--sim-root', tmp, '--seed', '7',
                        '--scene-seed-mode', 'seed_list', '--scene-seed-list', str(path)]), \
                    patch.object(runtime, 'ensure_python_hash_seed'), \
                    patch.object(runtime, 'create_seeded_runner') as fixed, \
                    patch.object(runtime, 'create_legacy_unseeded_runner') as legacy, \
                    patch.object(runtime, 'create_seed_list_runner', return_value=runner) as factory:
                runpy.run_path(str(script), run_name='__main__')
            fixed.assert_not_called()
            legacy.assert_not_called()
            self.assertEqual(factory.call_args.kwargs['episode_seeds'], SEEDS[:5])
            self.assertEqual(runner.run_simulation.call_args.args[0].seed, 7)
            saved = json.loads((output / 'config.json').read_text())
            self.assertEqual(saved['scene_seed']['episode_seeds'], SEEDS[:5])
            self.assertEqual(saved['server']['seed'], 7)
            client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
