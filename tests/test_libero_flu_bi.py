import copy
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from omegaconf import OmegaConf
import torch
from wam.datasets.lerobot.transforms.libero import _matrix_to_rotation_6d, _rotation_6d_to_matrix
from wam.datasets.lerobot.transforms.libero_flu_bi import LiberoRight80FLUBIdentityTransform, LIBERO_FLU_A, LIBERO_FLU_B
from wam.datasets.lerobot.transforms.libero_ab import LiberoRight80ABTransform
from wam.datasets.lerobot.utils.rotation import axis_angle_to_matrix


class FLUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.t = LiberoRight80FLUBIdentityTransform()
        self.cmd = torch.rand(4096, 7, dtype=torch.float64) * 2 - 1
        self.ee = torch.randn(4096, 6, dtype=torch.float64)

    def test_doc_matrices(self):
        a = torch.tensor(LIBERO_FLU_A, dtype=torch.float64)
        torch.testing.assert_close(a @ torch.tensor([1., 2., 3.], dtype=torch.float64), torch.tensor([2., -1., 3.], dtype=torch.float64))
        torch.testing.assert_close(a @ a.T, torch.eye(3, dtype=torch.float64))
        self.assertEqual(LIBERO_FLU_B, ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))

    def test_action_formula_roundtrip(self):
        encoded = self.t.encode_commands(self.cmd, self.ee)
        a = self.cmd.new_tensor(LIBERO_FLU_A)
        r = axis_angle_to_matrix(self.ee[..., 3:])
        torch.testing.assert_close(encoded[..., :3], .05 * self.cmd[..., :3] @ a.T)
        torch.testing.assert_close(encoded[..., 3:6], (r.transpose(-1, -2) @ (.5 * self.cmd[..., 3:6, None])).squeeze(-1))
        torch.testing.assert_close(self.t.decode_commands(encoded, self.ee), self.cmd, atol=1e-12, rtol=1e-12)

    def test_state_and_column_rot6d(self):
        raw = {"state": {"joint": self.cmd.clone(), "ee": self.ee.clone(), "gripper": self.cmd[..., :2].clone()}, "action": {"default": self.cmd.clone()}}
        b = self.t.forward(copy.deepcopy(raw))
        a = self.ee.new_tensor(LIBERO_FLU_A)
        expected = a @ axis_angle_to_matrix(self.ee[..., 3:])
        torch.testing.assert_close(_rotation_6d_to_matrix(b["state"]["ee"][..., 3:]), expected)
        torch.testing.assert_close(b["state"]["ee"][..., 3:], expected[..., :, :2].transpose(-1, -2).reshape(-1, 6))
        inverse = self.t.backward(b)
        torch.testing.assert_close(inverse["state"]["ee"][..., :3], self.ee[..., :3])
        torch.testing.assert_close(axis_angle_to_matrix(inverse["state"]["ee"][..., 3:]), axis_angle_to_matrix(self.ee[..., 3:]))
        torch.testing.assert_close(inverse["action"]["default"], self.cmd)

    def test_extra_state_and_clipping(self):
        cmd = torch.tensor([[2., -2., 0., 0., 0., 0., .25]])
        ee = torch.zeros(2, 6)
        encoded = self.t.encode_commands(cmd, ee)
        sim = self.t.decode_simulator_commands(encoded, ee[:1])
        torch.testing.assert_close(sim, torch.tensor([[1., -1., 0., 0., 0., 0., .5]]))
        with self.assertRaises(ValueError):
            self.t.encode_commands(cmd, torch.zeros(3, 6))

    def test_current_rotation_at_each_step(self):
        cmd = torch.tensor([[0., 0., 0., .2, 0., 0., .5]])
        old = torch.zeros(1, 6)
        new = old.clone(); new[0, 5] = torch.pi / 2
        memory = self.t.encode_commands(cmd, old)
        decoded = self.t.decode_simulator_commands(memory, new)
        torch.testing.assert_close(decoded[0, 3:6], torch.tensor([0., .2, 0.]), atol=1e-6, rtol=1e-6)

    def test_distinct_from_old_ab(self):
        old = LiberoRight80ABTransform().encode_commands(self.cmd, self.ee)
        new = self.t.encode_commands(self.cmd, self.ee)
        self.assertFalse(torch.allclose(old[..., :6], new[..., :6]))

    def test_checkpoint_frame_and_stats_guard(self):
        from wam.inference.libero_joint96_flu_bi import validate_evaluation_contract
        from wam.datasets.lerobot.transforms.libero_flu_bi import LIBERO_FLU_BI_CONTRACT
        cfg = OmegaConf.create({"joint96_posttrain": {"libero_frame_contract": LIBERO_FLU_BI_CONTRACT["version"], "expected_pretrain_step": 25000}, "data": {split: {"processor": {"action_state_transforms": [{"_target_": "wam.datasets.lerobot.transforms.libero_flu_bi.LiberoRight80FLUBIdentityTransform"}]}} for split in ("train", "val")}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(json.dumps({"frame_contract": LIBERO_FLU_BI_CONTRACT}))
            contract = {"frame_contract": LIBERO_FLU_BI_CONTRACT["version"], "normalization_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "embodiment_id": 0, "domain": "libero"}
            payload = {"joint96_metadata": {"soft_prompt": {"training_contract": contract}}}
            validate_evaluation_contract(cfg, payload, path)
            contract["frame_contract"] = "libero_osc_ab_v1"
            with self.assertRaises(ValueError):
                validate_evaluation_contract(cfg, payload, path)
            contract["frame_contract"] = LIBERO_FLU_BI_CONTRACT["version"]
            contract["normalization_sha256"] = "wrong"
            validate_evaluation_contract(cfg, payload, path)


if __name__ == "__main__":
    unittest.main()
