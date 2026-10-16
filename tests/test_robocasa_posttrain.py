import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

from wam.datasets.robocasa_delta import (
    CONTRACT, RAW, JOINTS, project, decode, masks, RoboCasaNormalizer, observation_state, prompt_for,
)
from wam.datasets.robocasa_posttrain import select_episodes
from wam.datasets.robocasa_rotations import _rotvec_to_matrix


def raw_pose(n=4):
    state = torch.zeros(n, 161)
    action = torch.zeros_like(state)
    for side in ("l", "r"):
        p, q = RAW[f"wrist_{side}_pos"], RAW[f"wrist_{side}_rot6d"]
        r = _rotvec_to_matrix(torch.tensor([[.4, -.2, .1]]).repeat(n, 1))
        d = _rotvec_to_matrix(torch.tensor([[.02, .03, -.04]]).repeat(n, 1))
        state[:, p[0]:p[1]] = torch.arange(n)[:, None]*.1
        action[:, p[0]:p[1]] = state[:, p[0]:p[1]] + torch.tensor([.01, -.02, .03])
        state[:, q[0]:q[1]] = r[:, :, :2].transpose(-1, -2).reshape(n, 6)
        action[:, q[0]:q[1]] = (r@d)[:, :, :2].transpose(-1, -2).reshape(n, 6)
    for i, key in enumerate(JOINTS):
        a, b = RAW[key]
        state[:, a:b] = i*.1
        action[:, a:b] = i*.1+.03
    return state, action


def norm_stats():
    return {"contract": CONTRACT, "mode": "minmax", **{
        kind: {"min": [-2.]*80, "max": [4.]*80} for kind in ("state", "action")}}


class RoboCasaTests(unittest.TestCase):
    def test_native_delta_not_chunk_anchor_or_double_shift(self):
        state, action = raw_pose()
        s, a = project(state, action)
        torch.testing.assert_close(a[:, 7:10], torch.tensor([[.01, -.02, .03]]).repeat(4, 1), atol=1e-6, rtol=0)
        torch.testing.assert_close(a[:, 10:13], torch.tensor([[.02, .03, -.04]]).repeat(4, 1), atol=1e-6, rtol=0)
        for key, (begin, end) in JOINTS.items():
            x, y = RAW[key]
            self.assertTrue(torch.equal(a[:, begin:end], action[:, x:y]))

    def test_roundtrip_noncommuting_rotations(self):
        state, action = raw_pose()
        _, a = project(state, action)
        result = decode(a, state)
        for key, value in result.items():
            start, end = RAW[key.removeprefix("action.")]
            torch.testing.assert_close(value, action[:, start:end], atol=1e-6, rtol=1e-5)

    def test_structural_masks(self):
        state, action = masks()
        self.assertEqual(int(state.sum()), 47)
        self.assertEqual(int(action.sum()), 41)
        self.assertFalse(bool(action[13:17].any()))
        self.assertTrue(bool(action[17:23].all()))
        self.assertTrue(bool(action[29:32].all()))
        self.assertTrue(bool(action[57:63].all()))

    def test_normalization_inverse(self):
        s, a = project(*raw_pose())
        norm = RoboCasaNormalizer(norm_stats())
        for kind, value in (("state", s), ("action", a)):
            torch.testing.assert_close(norm.backward(norm.forward(value, kind), kind), value, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(norm.forward(s, "state")[:, 10:16], s[:, 10:16])

    def test_missing_wrist_is_error(self):
        state, action = raw_pose()
        obs = {"state."+k: state[:1, x:y] for k, (x, y) in RAW.items()}
        observation_state(obs)
        del obs["state.wrist_r_pos"]
        with self.assertRaises(KeyError): observation_state(obs)

    def test_degenerate_rotations_rejected(self):
        state, action = raw_pose()
        state[:, 47:53] = 0
        with self.assertRaisesRegex(ValueError, "Degenerate"): project(state, action)

    def test_nonfinite_required_data_rejected(self):
        state, action = raw_pose()
        action[0, 0] = float("nan")
        with self.assertRaises(ValueError): project(state, action)

    def test_split_fewshot_stable_no_leakage(self):
        plan = {"scope": "full24_all_episodes", "episodes": [
            {"task": task, "episode": i} for task in ("a", "b") for i in range(1000)]}
        one = select_episodes(plan, fewshot_ratio=.1)
        reverse = select_episodes({**plan, "episodes": list(reversed(plan["episodes"]))}, fewshot_ratio=.1)
        self.assertEqual(one, reverse)
        self.assertEqual(sum(e["split"] == "train" for e in one), 198)
        self.assertEqual(sum(e["split"] == "val" for e in one), 20)
        full = select_episodes(plan)
        self.assertEqual([e for e in one if e["split"] == "val"], [e for e in full if e["split"] == "val"])
        self.assertEqual(len({(e['task'], e['episode']) for e in one}), len(one))

    def test_prompt_prefix_consistent(self):
        self.assertEqual(prompt_for("locked_waist: pick up cup"), prompt_for("pick up cup"))
        with self.assertRaises(ValueError): prompt_for("")

    def test_policy_joint_mode_and_controller_mapping(self):
        from wam.inference.robocasa_joint96 import RoboCasaJoint96Policy
        s, a = project(*raw_pose(n=32))
        norm = RoboCasaNormalizer(norm_stats())
        class Model:
            device = "cpu"
        class Text:
            def get(self, prompt):
                return torch.zeros(128,4096), torch.ones(128, dtype=torch.bool)
        policy = RoboCasaJoint96Policy(Model(), norm, Text())
        with patch("wam.inference.robocasa_joint96.infer_frame_joint_window", return_value={"action":norm.forward(a, "action")}) as infer:
            commands = policy.predict(np.zeros((256,256,3), dtype=np.uint8), raw_pose(n=1)[0], "pick up cup")
            self.assertEqual(infer.call_args.kwargs["embodiment_ids"].tolist(), [13])
            self.assertFalse(infer.call_args.kwargs["decode_video"])
            self.assertEqual(infer.call_args.kwargs["action_horizon"], 32)
            self.assertEqual(infer.call_args.kwargs["num_inference_steps"], 10)
            self.assertEqual(set(commands), {"action."+k for k in JOINTS})
            for key, (begin, end) in JOINTS.items():
                np.testing.assert_allclose(commands["action."+key][0], a[:12,begin:end], atol=1e-6)
            self.assertEqual(infer.call_args.kwargs["seed"], 7)
            policy.predict(np.zeros((256,256,3), dtype=np.uint8), raw_pose(n=1)[0], "pick up cup")
            self.assertEqual(infer.call_args.kwargs["seed"], 7)
            self.assertEqual(policy.replans, 2)
            policy.reset()
            self.assertEqual(policy.replans, 0)
            policy.predict(np.zeros((256,256,3), dtype=np.uint8), raw_pose(n=1)[0], "pick up cup")
            self.assertEqual([call.kwargs["seed"] for call in infer.call_args_list], [7, 7, 7])

    def test_policy_custom_seed_is_fixed_across_replans_and_reset(self):
        from wam.inference.robocasa_joint96 import RoboCasaJoint96Policy
        _, a = project(*raw_pose(n=32))
        norm = RoboCasaNormalizer(norm_stats())
        class Model:
            device = "cpu"
        class Text:
            def get(self, prompt):
                return torch.zeros(128,4096), torch.ones(128, dtype=torch.bool)
        policy = RoboCasaJoint96Policy(Model(), norm, Text(), execute_steps=5, seed=23)
        with patch("wam.inference.robocasa_joint96.infer_frame_joint_window", return_value={"action": norm.forward(a, "action")}) as infer:
            for reset in (False, False, True):
                if reset:
                    policy.reset()
                commands = policy.predict(np.zeros((256,256,3), dtype=np.uint8), raw_pose(n=1)[0], "pick up cup")
                for key, (begin, end) in JOINTS.items():
                    np.testing.assert_allclose(commands["action."+key][0], a[:5,begin:end], atol=1e-6)
            self.assertEqual([call.kwargs["seed"] for call in infer.call_args_list], [23, 23, 23])

    def test_checkpoint_rejects_wrong_training_contract(self):
        from wam.inference.robocasa_joint96 import validate_checkpoint
        from wam.datasets.robocasa_visual_cache import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "text").mkdir()
            (root / "text/COMPLETE.json").write_text("{}")
            package = {"sha256": "package", "norm_sha256": "norm", "plan_sha256": "visual"}
            contract = {"domain": "robocasa", "frame_contract": CONTRACT, "embodiment_id": 13,
                "package_sha256": "package", "normalization_sha256": "norm", "visual_plan_sha256": "visual",
                "visual_attention_backend": "math_sdpa", "pretrain_step": 72500,
                "text_manifest_sha256": file_hash(root / "text/COMPLETE.json")}
            payload = {"joint96_metadata": {"soft_prompt": {"training_contract": contract}}}
            validate_checkpoint(payload, package, root)
            contract["pretrain_step"] = 75000
            validate_checkpoint(payload, package, root, expected_pretrain_step=75000)
            with self.assertRaisesRegex(ValueError, "pretrain_step"):
                validate_checkpoint(payload, package, root)
            contract["pretrain_step"] = 72500
            for key in ("embodiment_id", "pretrain_step"):
                bad = copy.deepcopy(payload)
                bad["joint96_metadata"]["soft_prompt"]["training_contract"][key] = "wrong"
                with self.assertRaisesRegex(ValueError, key): validate_checkpoint(bad, package, root)

    def test_cache_reuse_requires_identical_encoder_and_step(self):
        from wam.training.robocasa_posttrain import validate_cached_initialization
        identity = {"teacher_sha256": "dino", "stvae_checkpoint_sha256": "stvae"}
        plan = {"encoder_identity": identity, "resources": {"policy": {"path": "/old/72500.pt"}}}
        payload = {"step": 75000, "joint96_metadata": {}}
        with patch("wam.training.checkpoint._joint96_identity", return_value=identity):
            self.assertEqual(validate_cached_initialization(payload, plan, 75000), identity)
            with self.assertRaisesRegex(ValueError, "step"):
                validate_cached_initialization(payload, plan, 72500)
            bad = copy.deepcopy(plan)
            bad["encoder_identity"]["teacher_sha256"] = "different_dino"
            validate_cached_initialization(payload, bad, 75000)
            bad["encoder_identity"]["action_dim"] = 7
            with self.assertRaisesRegex(ValueError, "action_dim"):
                validate_cached_initialization(payload, bad, 75000)


if __name__ == "__main__":
    unittest.main()
