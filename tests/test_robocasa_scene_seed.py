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

from wam import robocasa_simulation as seed_runtime


class FakeRoboCasa(gym.Env):
    def __init__(self, seed):
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.observation_space = gym.spaces.Box(0, 1, (3,), np.float64)
        self.env = SimpleNamespace(seed=seed, rng=np.random.default_rng(seed))
        self.retained_rng = self.env.rng
        self.received_seeds = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        # Match the bug in the actual Gym adapter: this does NOT seed self.env.rng.
        np.random.seed(seed)
        self.received_seeds.append(seed)
        return np.array([self.retained_rng.random(), np.random.random(), random.random()]), {}

    def step(self, action):
        return np.zeros(3), 0., True, False, {}

    def close(self):
        self.closed = True


class SceneSeedTests(unittest.TestCase):
    def make_env(self, seed=7, stride=1):
        return seed_runtime.SeededRoboCasaEnv(FakeRoboCasa(seed), seed=seed, stride=stride)

    def test_legacy_reset_does_not_reseed_backend_or_replace_rng(self):
        base = FakeRoboCasa(None)
        expected = np.random.default_rng(123)
        base.env.rng.bit_generator.state = expected.bit_generator.state
        wrapper = seed_runtime.LegacyUnseededRoboCasaEnv(base)
        first, _ = wrapper.reset(seed=7)
        second, _ = wrapper.reset(seed=7)
        self.assertEqual((first[0], second[0]), (expected.random(), expected.random()))
        self.assertIs(base.retained_rng, base.env.rng)
        self.assertIsNone(base.env.seed)
        self.assertEqual(base.received_seeds, [7, 7])

    def test_legacy_factory_reuses_original_simulator_factory(self):
        config = SimpleNamespace(seed=7, n_envs=1, env_name='test')
        factory = Mock(return_value=FakeRoboCasa(None))
        with patch.dict(sys.modules, {'Robocasa_tabletop.eval_files.simulation_env':
                                     SimpleNamespace(_create_single_env=factory)}):
            env = seed_runtime.make_legacy_unseeded_env(config, 0)
        factory.assert_called_once_with(config, 0)
        self.assertIsNone(env.env.env.seed)

    def test_legacy_reset_audit_records_none_and_detects_accidental_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'env_0.jsonl'
            env = seed_runtime.LegacyUnseededRoboCasaEnv(FakeRoboCasa(None), audit_path=path)
            with patch.object(seed_runtime, 'scene_fingerprint', return_value={'backend_seed': None}):
                env.reset(seed=7)
            row = json.loads(path.read_text())
            self.assertIsNone(row['backend_seed'])
            self.assertIsNone(row['scene_seed'])
            self.assertEqual(row['gym_reset_seed'], 7)
            env.env.env.seed = 7
            with self.assertRaisesRegex(RuntimeError, 'unexpectedly fixed'):
                env.reset()
        with self.assertRaisesRegex(ValueError, 'seed=None'):
            seed_runtime.LegacyUnseededRoboCasaEnv(FakeRoboCasa(7))

    def test_actual_eval_entrypoint_can_select_legacy_backend_only(self):
        self._check_legacy_entrypoint(['--scene-seed-mode', 'legacy_unseeded'])

    def test_actual_eval_entrypoint_defaults_to_legacy_backend(self):
        self._check_legacy_entrypoint([])

    def test_actual_eval_entrypoint_accepts_delta80_slot18(self):
        self._check_legacy_entrypoint([], {'policy': 'robocasa_delta80_slot18_joint_v1',
                                          'embodiment_id': 18, 'action_dim': 80})

    def test_actual_eval_entrypoint_rejects_wrong_delta80_slot_or_dim(self):
        for slot, dim in ((13, 80), (18, 47)):
            with self.assertRaisesRegex(ValueError, 'action space/prompt slot'):
                self._check_legacy_entrypoint([], {'policy': 'robocasa_delta80_slot18_joint_v1',
                                                  'embodiment_id': slot, 'action_dim': dim})

    def _check_legacy_entrypoint(self, mode_args, metadata_override=None):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'results'
            client = Mock()
            client.get_server_metadata.return_value = {
                'policy': 'robocasa_absolute47_joint_v1', 'execute_steps': 12,
                'inference_mode': 'joint_video_actions', 'seed': 7,
                'seed_mode': 'fixed_per_replan', 'embodiment_id': 18, 'action_dim': 47}
            client.get_server_metadata.return_value.update(metadata_override or {})
            runner = Mock(env=None)
            runner.run_simulation.return_value = ['test', [False]]
            modules = {
                'deployment.model_server.tools.websocket_policy_client': SimpleNamespace(
                    WebsocketClientPolicy=Mock(return_value=client)),
                'Robocasa_tabletop.eval_files.simulation_env': SimpleNamespace(
                    SimulationConfig=SimpleNamespace, MultiStepConfig=SimpleNamespace),
            }
            script = Path(__file__).parents[1] / 'tools/eval_robocasa_joint96.py'
            with patch.dict(sys.modules, modules), patch.dict(os.environ), \
                 patch.object(sys, 'path', list(sys.path)), \
                 patch.object(sys, 'argv', [str(script), '--output', str(output), '--env', 'test',
                     '--episodes', '1', '--sim-root', tmp, '--seed', '7'] + mode_args), \
                 patch.object(seed_runtime, 'ensure_python_hash_seed'), \
                 patch.object(seed_runtime, 'create_seeded_runner') as fixed, \
                 patch.object(seed_runtime, 'create_legacy_unseeded_runner', return_value=runner) as legacy:
                runpy.run_path(str(script), run_name='__main__')
            fixed.assert_not_called()
            legacy.assert_called_once()
            self.assertEqual(runner.run_simulation.call_args.args[0].seed, 7)
            saved = json.loads((output / 'config.json').read_text())
            self.assertEqual(saved['scene_seed'], seed_runtime.scene_seed_config(7, 'legacy_unseeded'))
            self.assertEqual(saved['server']['seed'], 7)
            client.close.assert_called_once()

    def test_explicit_reset_replays_all_rngs_and_preserves_sampler_reference(self):
        env = self.make_env()
        before_rng = env.env.retained_rng
        first, info = env.reset(seed=7)
        env.env.env.rng.random(100)
        np.random.random(17)
        random.random()
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertIs(before_rng, env.env.env.rng)
        self.assertEqual(env.env.env.seed, 7)
        self.assertEqual(info['scene_seed'], 7)

    def test_automatic_resets_advance_not_repeat_or_use_entropy(self):
        env = self.make_env()
        values = [env.reset(seed=7)[0], env.reset()[0], env.reset()[0]]
        self.assertEqual(env.env.received_seeds, [7, 8, 9])
        self.assertEqual(env.env.env.seed, 9)
        self.assertFalse(np.array_equal(values[0], values[1]))

    def test_next_scene_does_not_depend_on_random_consumption_in_previous_episode(self):
        first, second = self.make_env(), self.make_env()
        first.reset(seed=7)
        first.env.env.rng.random(1000)
        np.random.random(999)
        random.random()
        expected, _ = first.reset()
        second.reset(seed=7)
        actual, _ = second.reset()
        np.testing.assert_array_equal(expected, actual)

    def test_vector_workers_get_disjoint_episode_seed_sequences(self):
        a, b = self.make_env(7, 2), self.make_env(8, 2)
        for _ in range(3):
            a.reset()
            b.reset()
        self.assertEqual(a.env.received_seeds, [7, 9, 11])
        self.assertEqual(b.env.received_seeds, [8, 10, 12])

    def test_sync_vector_autoreset_does_not_drop_seed(self):
        vector = gym.vector.SyncVectorEnv([self.make_env])
        try:
            vector.reset(seed=7)
            for _ in range(4):
                vector.step(np.zeros((1, 1), np.float32))
            observed = vector.envs[0].env.received_seeds
            self.assertGreater(len(observed), 1)
            self.assertEqual(observed, list(range(7, 7 + len(observed))))
        finally:
            vector.close()

    def test_seed_zero_and_uint32_wrap_are_supported(self):
        env = self.make_env(2**32 - 1)
        env.reset()
        env.reset()
        self.assertEqual(env.env.received_seeds, [2**32 - 1, 0])

    def test_unseeded_constructor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'constructor'):
            seed_runtime.SeededRoboCasaEnv(FakeRoboCasa(None), seed=7)

    def test_reset_cannot_silently_discard_backend_seed(self):
        env = self.make_env()
        original = env.env.reset
        def broken_reset(**kwargs):
            value = original(**kwargs)
            env.env.env.seed = None
            return value
        with patch.object(env.env, 'reset', side_effect=broken_reset):
            with self.assertRaisesRegex(RuntimeError, 'lost the scene seed'):
                env.reset(seed=7)

    def test_factory_refuses_uncontrolled_hash_seed_before_constructing_scene(self):
        config = SimpleNamespace(seed=7, n_envs=1, env_name='test')
        with patch.dict(os.environ, PYTHONHASHSEED='123'), patch.object(seed_runtime.gym, 'make') as make:
            with self.assertRaisesRegex(ValueError, 'ensure_python_hash_seed'):
                seed_runtime.make_seeded_env(config, 0)
            make.assert_not_called()

    def test_invalid_seed_is_rejected(self):
        for value in (-1, 2**32, None, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                seed_runtime.validate_seed(value)

    def test_factory_passes_constructor_seed(self):
        config = SimpleNamespace(seed=7, n_envs=2, env_name='test')
        with patch.dict(os.environ, PYTHONHASHSEED='7'), \
             patch.object(seed_runtime.gym, 'make', side_effect=lambda name, **kw: FakeRoboCasa(kw['seed'])) as make, \
             patch.object(seed_runtime, '_wrap_simulation_env', side_effect=lambda env, cfg: env):
            env = seed_runtime.make_seeded_env(config, 1, enable_render=False)
        make.assert_called_once_with('test', enable_render=False, seed=8)
        self.assertEqual(env.reset()[1]['scene_seed'], 8)
        self.assertEqual(env.reset()[1]['scene_seed'], 10)

    def test_hash_seed_mismatch_reexecs_before_task_imports(self):
        with patch.dict(os.environ, PYTHONHASHSEED='123'), \
             patch.object(seed_runtime.os, 'execve', side_effect=SystemExit) as execute:
            with self.assertRaises(SystemExit):
                seed_runtime.ensure_python_hash_seed(7)
        self.assertEqual(execute.call_args.args[2]['PYTHONHASHSEED'], '7')
        with patch.dict(os.environ, PYTHONHASHSEED='7'), patch.object(seed_runtime.os, 'execve') as execute:
            seed_runtime.ensure_python_hash_seed(7)
            execute.assert_not_called()

    def test_reset_audit_records_seed_and_real_initial_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'env_0.jsonl'
            env = seed_runtime.SeededRoboCasaEnv(FakeRoboCasa(7), seed=7, audit_path=path)
            with patch.object(seed_runtime, 'scene_fingerprint', return_value={'backend_seed': 7, 'qpos': 'hash'}) as fingerprint:
                obs, _ = env.reset()
            fingerprint.assert_called_once()
            self.assertIs(fingerprint.call_args.args[1], obs)
            data = json.loads(path.read_text())
            self.assertEqual((data['scene_seed'], data['backend_seed'], data['reset_index']), (7, 7, 0))
            self.assertEqual(data['protocol'], seed_runtime.SEED_PROTOCOL)

    def test_actual_eval_entrypoint_uses_seeded_runner_and_records_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'results'
            metadata = {'policy': 'robocasa_delta80_joint_v1', 'execute_steps': 12,
                        'inference_mode': 'joint_video_actions', 'seed': 7, 'seed_mode': 'fixed_per_replan'}
            client = Mock()
            client.get_server_metadata.return_value = metadata
            runner = Mock(env=None)
            runner.run_simulation.return_value = ['test', [False]]
            modules = {
                'deployment.model_server.tools.websocket_policy_client': SimpleNamespace(
                    WebsocketClientPolicy=Mock(return_value=client)),
                'Robocasa_tabletop.eval_files.simulation_env': SimpleNamespace(
                    SimulationConfig=SimpleNamespace, MultiStepConfig=SimpleNamespace),
            }
            script = Path(__file__).parents[1] / 'tools/eval_robocasa_joint96.py'
            with patch.dict(sys.modules, modules), patch.dict(os.environ), \
                 patch.object(sys, 'path', list(sys.path)), \
                 patch.object(sys, 'argv', [str(script), '--output', str(output), '--env', 'test',
                     '--episodes', '1', '--sim-root', tmp, '--seed', '7', '--scene-seed-mode', 'fixed']), \
                 patch.object(seed_runtime, 'ensure_python_hash_seed') as ensure, \
                 patch.object(seed_runtime, 'create_seeded_runner', return_value=runner) as create:
                runpy.run_path(str(script), run_name='__main__')
            ensure.assert_called_once_with(7)
            self.assertEqual(create.call_args.kwargs['audit_dir'], output / 'scene_seeds')
            self.assertEqual(runner.run_simulation.call_args.args[0].seed, 7)
            saved = json.loads((output / 'config.json').read_text())
            self.assertEqual(saved['scene_seed'], seed_runtime.scene_seed_config(7))
            client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
