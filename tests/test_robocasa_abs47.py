import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from wam.datasets import robocasa_posttrain as data
from wam.datasets.robocasa_delta import RAW, prompt_for


class ConfigurationTests(unittest.TestCase):
    def compose_task(self, task):
        from hydra import compose, initialize_config_dir
        from wam.utils.config_resolvers import register_default_resolvers
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "configs"), version_base=None):
            return compose(config_name="train", overrides=["task=" + task])

    def test_delta80_requested_learning_rate_noise_and_loss_weights(self):
        with patch.dict('os.environ', {'EXPECTED_PRETRAIN_STEP': '110000'}):
            cfg = self.compose_task("robocasa_posttrain")
            cfg.joint96_posttrain.expected_pretrain_step = int(cfg.joint96_posttrain.expected_pretrain_step)
        self.assertEqual(cfg.learning_rate, 1e-4)
        for scheduler, shift in ((cfg.model.video_scheduler, 6.), (cfg.model.action_scheduler, 1.)):
            self.assertEqual((scheduler.train_shift, scheduler.infer_shift), (shift, shift))
            self.assertEqual(scheduler.num_train_timesteps, 1000)
        self.assertEqual(dict(cfg.model.loss), dict(lambda_video=1., lambda_action=1.,
                                                  lambda_vae=.5, lambda_stvae=.5))
        self.assertEqual(cfg.model.action_dit_config.action_dim, 80)
        self.assertEqual(cfg.joint96_posttrain.robocasa.embodiment_id, 18)
        self.assertEqual(cfg.joint96_posttrain.expected_pretrain_step, 110000)
        self.assertEqual((cfg.min_learning_rate, cfg.weight_decay), (5e-7, .01))

    def test_unsupported_historical_recipes_not_exposed(self):
        from hydra.errors import MissingConfigException
        for task in ('robocasa_joint96_abs47_slot18_75000', 'robocasa_joint96_cached_robot37_75000'):
            with self.assertRaises(MissingConfigException):
                self.compose_task(task)


class InstructionTests(unittest.TestCase):
    def test_prepare_uses_each_episode_remarks_not_constant_task_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = root / "task/meta"
            meta.mkdir(parents=True)
            (meta / "modality.json").write_text("{}")
            (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 1, "task": "pick squash"}) + "\n")
            (meta / "episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "remarks": text})
                for i, text in enumerate(("pick pear", "pick lemon"))) + "\n")
            plan = {"scope": "smoke", "data_root": str(root), "sha256": "visual",
                    "episodes": [{"task": "task", "episode": i} for i in range(2)]}
            # Exercise the actual package writer; only numeric/video IO is excluded.
            with patch.object(data, "load_plan", return_value=plan), \
                 patch.object(data, "validate_modality"), \
                 patch.object(data, "_iter_episode_statistics", return_value=iter([({}, [1]), ({}, [1])])):
                package = data.prepare_package(root / "plan.json", root / "package", smoke=True)
            self.assertEqual(package["prompts"], sorted(map(prompt_for, ("pick pear", "pick lemon"))))
            self.assertIn(str(meta / "episodes.jsonl"), package["metadata"])
            self.assertEqual(package["prompt_source"], "episodes.jsonl:remarks")


class Absolute47Tests(unittest.TestCase):
    def test_projection_is_exact_legacy_order_without_delta_or_rotation_rewrite(self):
        from wam.datasets import robocasa_abs47 as space
        state = torch.arange(3 * 161).float().reshape(3, 161) / 100
        action = state + .75
        s, a = space.project(state, action)
        expected_order = ("left_arm", "right_arm", "left_hand", "right_hand", "waist",
                          "wrist_l_pos", "wrist_l_rot6d", "wrist_r_pos", "wrist_r_rot6d")
        for value, raw in ((s, state), (a, action)):
            self.assertTrue(torch.equal(value, torch.cat([raw[:, slice(*RAW[k])] for k in expected_order], -1)))
        self.assertEqual(a.shape, (3, 47))
        for key, value in space.decode(a).items():
            self.assertTrue(torch.equal(value, action[:, slice(*RAW[key.removeprefix("action.")])]))
        self.assertEqual(sum(v.shape[-1] for v in space.decode(a).values()), 29)

    def test_zscore_all_47_dimensions_and_inverse(self):
        from wam.datasets import robocasa_abs47 as space
        stats = {"contract": space.CONTRACT, "mode": "z-score", **{
            kind: {"mean": [2.] * 47, "std": [.5] * 47} for kind in ("state", "action")}}
        norm = space.RoboCasaNormalizer(stats)
        x = torch.full((2, 47), 3.)
        for kind in ("state", "action"):
            self.assertTrue(torch.equal(norm.forward(x, kind), torch.full_like(x, 2.)))
            torch.testing.assert_close(norm.backward(norm.forward(x, kind), kind), x)
        stats["action"]["std"][0] = -1
        with self.assertRaises(ValueError):
            space.RoboCasaNormalizer(stats)

    def test_new_registry_only_activates_reserved_slot18(self):
        from wam.training.robocasa_posttrain import activate_robocasa_slot
        from wam.embodiment_registry import EmbodimentRegistry
        source = json.loads((Path(__file__).parents[1] / "configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json").read_text())
        target = activate_robocasa_slot(source)
        EmbodimentRegistry.validate(target)
        self.assertEqual(source["slot_status"][18], "reserved")
        self.assertEqual(target["slot_status"][18], "assigned")
        self.assertEqual(target["embodiments"][18], "robocasa_gr1_abs47")
        self.assertEqual(target["embodiments"][:18], source["embodiments"][:18])
        self.assertEqual(target["embodiments"][19:], source["embodiments"][19:])
        with self.assertRaises(ValueError):
            activate_robocasa_slot(target)

    def test_absolute_policy_uses_slot18_and_executes_joint29(self):
        import numpy as np
        from wam.datasets import robocasa_abs47 as space
        from wam.inference.robocasa_joint96 import RoboCasaJoint96Policy
        from test_robocasa_posttrain import raw_pose
        stats = {"contract": space.CONTRACT, "mode": "z-score", **{
            kind: {"mean": [0.] * 47, "std": [1.] * 47} for kind in ("state", "action")}}
        norm = space.RoboCasaNormalizer(stats)
        _, action = space.project(*raw_pose(n=32))
        text = SimpleNamespace(get=lambda _: (torch.zeros(128,4096), torch.ones(128,dtype=torch.bool)))
        policy = RoboCasaJoint96Policy(SimpleNamespace(device="cpu"), norm, text, action_space="absolute47")
        with patch("wam.inference.robocasa_joint96.infer_frame_joint_window", return_value={"action": norm.forward(action, "action")}) as infer:
            commands = policy.predict(np.zeros((256,256,3), dtype=np.uint8), raw_pose(n=1)[0], "pick pear")
        self.assertEqual(infer.call_args.kwargs["embodiment_ids"].tolist(), [18])
        self.assertEqual(infer.call_args.kwargs["action_dim_is_pad"].shape, (1,47))
        self.assertEqual(sum(v.shape[-1] for v in commands.values()), 29)
        for key, value in commands.items():
            torch.testing.assert_close(torch.tensor(value[0]), space.decode(action)[key][:12])


class MigrationTests(unittest.TestCase):
    def model(self, dim, registry):
        from wam.model.modules.conditioning.embodiment_prompt import EmbodimentPrompt
        action = torch.nn.Module()
        action.action_dim = dim
        action.action_encoder = torch.nn.Linear(dim, 4)
        action.head = torch.nn.Linear(4, dim)
        action.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        action.embodiment_prompt = EmbodimentPrompt(28, 8, 4, str(registry))
        mot = torch.nn.Module()
        mot.mixtures = torch.nn.ModuleDict({"action": action, "video": torch.nn.Linear(4,4)})
        return SimpleNamespace(joint96_three_view=True, mot=mot, action_expert=action,
            proprio_dim=dim, proprio_encoder=torch.nn.Linear(dim,4), action_proprio_encoder=None,
            model_paths={"joint_visual_encoder": {"source_sha256": "code", "stvae": {
                "stvae_step":150000, "teacher_sha256":"teacher", "stvae_checkpoint_sha256":"stvae"}}})

    def test_migration_inherits_all_io_and_activates_only_prompt_slot18(self):
        from wam.training.checkpoint import joint96_checkpoint_metadata, load_joint96_initialization
        from wam.training.robocasa_posttrain import load_robocasa_delta80_slot18_initialization as load_initialization
        root = Path(__file__).parents[1] / "configs/embodiment_registry"
        source = self.model(80, root / "local_robot37_arm18_capacity28_v1.json")
        target = self.model(80, root / "robocasa_gr1_delta80_slot18_v1.json")
        before = {k: v.clone() for k,v in target.mot.state_dict().items()}
        prop_before = copy.deepcopy(target.proprio_encoder.state_dict())
        payload = {"step":75000, "mot":source.mot.state_dict(), "proprio_encoder":source.proprio_encoder.state_dict(),
                   "joint96_metadata":joint96_checkpoint_metadata(source)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.pt"
            torch.save(payload, path)
            receipt = load_initialization(target, path, expected_step=75000)
            self.assertEqual(receipt["new_prompt_slot"], 18)
            for key, value in target.mot.state_dict().items():
                expected = payload["mot"][key]
                if key == "mixtures.action.embodiment_prompt.embedding.weight":
                    expected = expected.clone()
                    expected[18] = before[key][18]
                self.assertTrue(torch.equal(value, expected), key)
            for key,value in target.proprio_encoder.state_dict().items():
                self.assertTrue(torch.equal(value, payload["proprio_encoder"][key]))
            self.assertGreater(int(target.action_expert.embodiment_prompt.embedding.weight[18].count_nonzero()), 0)
            # A newly trained checkpoint uses the strict normal reload path, not migration.
            saved = {"step":1, "mot":target.mot.state_dict(), "proprio_encoder":target.proprio_encoder.state_dict(),
                     "joint96_metadata":joint96_checkpoint_metadata(target)}
            torch.save(saved, Path(tmp) / "trained.pt")
            reload = self.model(80, root / "robocasa_gr1_delta80_slot18_v1.json")
            load_joint96_initialization(reload, Path(tmp) / "trained.pt", expected_step=1)
            self.assertTrue(torch.equal(reload.action_expert.head.weight, target.action_expert.head.weight))
            # A broken backbone cannot be silently skipped.
            payload["mot"]["mixtures.video.weight"] = torch.zeros(1)
            torch.save(payload, path)
            snapshot = copy.deepcopy(target.mot.state_dict())
            with self.assertRaisesRegex(ValueError, "shape"):
                load_initialization(target, path, expected_step=75000)
            self.assertTrue(all(torch.equal(v,target.mot.state_dict()[k]) for k,v in snapshot.items()))

    def test_vae_runtime_keeps_fp32_scale_computation(self):
        from wam.model.world_action_model import WAM
        vae = torch.nn.Linear(1,1)
        vae.mean = torch.tensor([.12345])
        vae.std = torch.tensor([1.038])
        expected = (1/vae.std).to(torch.bfloat16)
        model = SimpleNamespace(vae=vae, torch_dtype=torch.bfloat16, joint_visual_encoder=None,
                                text_encoder=None, understanding=None)
        model.set_runtime_device = lambda device: WAM.set_runtime_device(model, device)
        from wam.training.robocasa_posttrain import install_robocasa_runtime
        install_robocasa_runtime(model)
        model.set_runtime_device("cpu")
        self.assertTrue(torch.equal(vae.scale[1].to(torch.bfloat16), expected))
        self.assertEqual(vae.std.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
