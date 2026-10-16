"""CPU regressions for stable window identity, sharding and immutable readback."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from wam.datasets import robocasa_visual_cache as c


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def episode(self, number=0, length=40):
        result = dict(task="fixture", episode=number, length=length, fps=20.0, windows=length-4)
        for key in ("video", "parquet"):
            path = self.root / f"{number}.{key}"
            path.write_bytes(b"fixture")
            result[key] = str(path)
            result[key + "_identity"] = c.file_identity(path)
        return result

    def fixture(self):
        episode = self.episode()
        episode.update(starts=[0, 1, 35], windows=3)
        plan = dict(schema=c.SCHEMA, episodes=[episode], windows=3)
        plan["sha256"] = c.digest(plan)
        path = self.root / "plan.json"
        c.atomic_json(path, plan)
        output = c.episode_output(self.root, episode)
        output.mkdir(parents=True)
        value = torch.arange(3 * np.prod(c.SHAPE), dtype=torch.float32).reshape(3, *c.SHAPE).to(torch.bfloat16)
        value.view(torch.uint16).numpy().tofile(output / "features.bf16.bin")
        receipt = dict(plan_sha256=plan["sha256"], episode_sha256=c.digest(episode), rows=3,
                       feature_shape=list(c.SHAPE), features_sha256=c.file_hash(output / "features.bf16.bin"))
        c.atomic_json(output / "COMPLETE.json", receipt)
        return path, episode, value

    def test_slide_one_not_four(self):
        self.assertEqual(c.window_mapping(40, 0)["frame_indices"], list(range(0, 33, 4)))
        self.assertEqual(c.window_mapping(40, 1)["frame_indices"], list(range(1, 34, 4)))

    def test_tail_is_repeat_with_mask(self):
        mapping = c.window_mapping(40, 35)
        self.assertEqual(mapping["frame_indices"], [35, 39, 39, 39, 39, 39, 39, 39, 39])
        self.assertEqual(mapping["image_is_pad"], [False, False, True, True, True, True, True, True, True])
        with self.assertRaises(ValueError):
            c.window_mapping(40, 36)

    def test_short_episodes(self):
        for length in [0, 1, 4]:
            with self.assertRaises(ValueError):
                c.window_mapping(length, 0)
        self.assertEqual(c.window_mapping(5, 0)["frame_indices"][:2], [0, 4])

    def test_128_partitions_exact_coverage(self):
        episodes = [dict(episode=i) for i in range(1003)]
        result = [e["episode"] for r in range(128) for e in c.assigned_episodes({"episodes": episodes}, r, 128)]
        self.assertEqual(sorted(result), list(range(1003)))
        self.assertEqual(len(set(result)), len(result))

    def test_invalid_partition(self):
        for rank, world in [(0, 0), (-1, 2), (2, 2)]:
            with self.assertRaises(ValueError):
                c.assigned_episodes({"episodes": []}, rank, world)

    def test_readback_and_padding(self):
        path, episode, value = self.fixture()
        cache = c.RoboCasaVisualCache(self.root, path)
        for row, start in enumerate(episode["starts"]):
            item = cache.get("fixture", 0, start)
            self.assertTrue(torch.equal(item["video_latents"][..., :16], value[row]))
            self.assertEqual(tuple(item["video_latents"].shape), (96, 3, 16, 48))
            self.assertEqual(int(item["video_latents"][..., 16:].count_nonzero()), 0)
            self.assertEqual(item["camera_is_pad"].tolist(), [False, True, True])

    def test_incomplete_rejected(self):
        path, episode, _ = self.fixture()
        (c.episode_output(self.root, episode) / "COMPLETE.json").unlink()
        with self.assertRaises(FileNotFoundError):
            c.RoboCasaVisualCache(self.root, path).get("fixture", 0, 0)

    def test_corrupt_feature_rejected(self):
        path, episode, _ = self.fixture()
        features = c.episode_output(self.root, episode) / "features.bf16.bin"
        with features.open("r+b") as stream:
            stream.write(b"\xff\xff")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            c.RoboCasaVisualCache(self.root, path).get("fixture", 0, 0)

    def test_truncated_feature_rejected(self):
        path, episode, _ = self.fixture()
        (c.episode_output(self.root, episode) / "features.bf16.bin").write_bytes(b"0")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            c.RoboCasaVisualCache(self.root, path).get("fixture", 0, 0)

    def test_wrong_episode_rejected(self):
        path, episode, _ = self.fixture()
        episode["length"] += 1
        c.validate_complete(self.root, episode, c.load_plan(path)["sha256"])

    def test_modified_plan_rejected(self):
        path, episode, _ = self.fixture()
        plan = c.read_json(path)
        plan["windows"] += 1
        c.atomic_json(path, plan)
        with self.assertRaisesRegex(ValueError, "count"):
            c.load_plan(path)

    def test_changed_source_rejected(self):
        path, episode, _ = self.fixture()
        Path(episode["video"]).write_bytes(b"different")
        c.RoboCasaVisualCache(self.root, path).get("fixture", 0, 0)
        Path(episode["video"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing source"):
            c.RoboCasaVisualCache(self.root, path).get("fixture", 0, 0)

    def test_rgb_range_and_slots(self):
        images = np.full((40, 256, 256, 3), 255, dtype=np.uint8)
        result = c.pack_rgb(images, c.window_mapping(40, 35))
        self.assertEqual(result.shape, (1, 3, 3, 9, 256, 256))
        self.assertTrue(bool((result[:, 0] == 1).all()))
        self.assertTrue(bool((result[:, 1:] == -1).all()))

    def test_bitwise_comparison(self):
        a = torch.tensor([1., 2.], dtype=torch.bfloat16)
        self.assertTrue(c.compare(a, a)["passed"])
        self.assertFalse(c.compare(a, a+1)["passed"])

    def test_torch24_restricted_loading(self):
        from wam.utils.torch24_compat import install_safe_serialization_compat
        install_safe_serialization_compat()
        path = self.root / "numpy.pt"
        torch.save({"rng": np.arange(5, dtype=np.uint32), "tensor": torch.ones(2)}, path)
        unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        self.assertLessEqual(unsafe, {"numpy.core.multiarray._reconstruct", "numpy.ndarray", "numpy.dtype"})
        allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
        with torch.serialization.safe_globals(allowed):
            result = torch.load(path, weights_only=True)
        np.testing.assert_array_equal(result["rng"], np.arange(5, dtype=np.uint32))

    def test_torch24_unknown_global_stays_rejected(self):
        from wam.utils.torch24_compat import install_safe_serialization_compat
        install_safe_serialization_compat()
        path = self.root / "unknown.pt"
        torch.save({"object": Path("/not/a/command")}, path)
        self.assertIn("pathlib.PosixPath", torch.serialization.get_unsafe_globals_in_checkpoint(path))
        with self.assertRaises(Exception):
            torch.load(path, weights_only=True)

    def test_ppu_only_extraction_entrypoint_is_not_exposed(self):
        # This port consumes verified caches; it does not expose the PPU encoder launcher.
        root = Path(c.__file__).resolve().parents[3]
        self.assertFalse((root / "scripts/extract_robocasa_joint96_cache_128gpu.sh").exists())


if __name__ == "__main__":
    unittest.main()
