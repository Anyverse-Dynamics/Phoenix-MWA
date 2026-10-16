from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from wam.datasets.robocasa_posttrain import RoboCasaTextCache, text_name
from wam.datasets.robocasa_visual_cache import atomic_json, file_hash, digest
from wam.inference.robocasa_text import RoboCasaInferenceText


class InferenceTextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.text = self.root / "text"
        self.text.mkdir()
        self.known = "pick the squash from the placemat and place it in the bowl"
        self.unseen = "pick the sweet potato from the placemat and place it in the bowl"
        self.path = self.text / text_name(self.known)
        torch.save({"prompt": self.known, "context": torch.ones(128, 4096, dtype=torch.bfloat16),
                    "mask": torch.arange(128) < 10}, self.path)
        atomic_json(self.text / "COMPLETE.json", {"package_sha256": "package", "context_len": 128,
                    "files": {self.path.name: file_hash(self.path)}})
        self.cache = RoboCasaTextCache(self.root, {"sha256": "package"})
        self.runtime = RoboCasaInferenceText(self.cache, {}, "cpu")

    def test_known_prompt_uses_original_cache(self):
        with patch.object(self.runtime, "_encode", side_effect=AssertionError("Must use cache")):
            context, mask = self.runtime.get(self.known)
        self.assertTrue(mask.all())
        self.assertEqual(context[:10].sum().item(), 10 * 4096)
        self.assertEqual(context[10:].count_nonzero().item(), 0)

    def test_unseen_exact_prompt_encoded_once_without_package_changes(self):
        before = {p.name: p.read_bytes() for p in self.text.iterdir()}
        values = torch.ones(128, 4096, dtype=torch.bfloat16)
        valid = torch.arange(128) < 12
        with patch.object(self.runtime, "_encode", return_value=(values, valid)) as encode:
            context, mask = self.runtime.get(self.unseen)
            context.zero_()
            mask.zero_()
            again, again_mask = self.runtime.get(self.unseen)
        encode.assert_called_once_with(self.unseen)
        self.assertTrue(again_mask.all())
        self.assertEqual(again[:12].sum().item(), 12 * 4096)
        self.assertEqual(again[12:].count_nonzero().item(), 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.text.iterdir()})

    def test_missing_committed_file_is_not_hidden_by_fallback(self):
        self.path.unlink()
        with patch.object(self.runtime, "_encode", side_effect=AssertionError("Must not fallback")):
            with self.assertRaises(FileNotFoundError):
                self.runtime.get(self.known)

    def test_corrupt_committed_file_is_not_hidden_by_fallback(self):
        self.path.write_bytes(b"corrupt")
        with patch.object(self.runtime, "_encode", side_effect=AssertionError("Must not fallback")):
            with self.assertRaises((ValueError, RuntimeError, __import__('pickle').UnpicklingError)):
                self.runtime.get(self.known)

    def test_nonfinite_runtime_context_rejected(self):
        with patch.object(self.runtime, "_encode", return_value=(
                torch.full((128, 4096), float("nan"), dtype=torch.bfloat16), torch.ones(128, dtype=torch.bool))):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                self.runtime.get(self.unseen)

    def test_bad_context_length_rejected(self):
        with self.assertRaisesRegex(ValueError, "context length"):
            RoboCasaInferenceText(self.cache, {"tokenizer_max_len": 256}, "cpu")

    def test_different_dynamic_instructions_not_substituted(self):
        with patch.object(self.runtime, "_encode", return_value=(
                torch.ones(128, 4096, dtype=torch.bfloat16), torch.ones(128, dtype=torch.bool))) as encode:
            self.runtime.get(self.unseen)
            self.runtime.get(self.unseen.replace("sweet potato", "lemon"))
        self.assertEqual([call.args[0] for call in encode.call_args_list],
                         [self.unseen, self.unseen.replace("sweet potato", "lemon")])

    def test_wrong_encoder_or_tokenizer_refused_before_loading(self):
        weights = self.root / "t5.pt"
        weights.write_bytes(b"test-weights")
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "config.json").write_text("{}")
        self.runtime.model_config.update(model_id="test", tokenizer_model_id="test")
        configs = (None, SimpleNamespace(path=str(weights), download_if_necessary=lambda: None),
                   None, SimpleNamespace(path=str(tokenizer), download_if_necessary=lambda: None))
        for bad_key, message in (("t5_sha256", "weights differ"), ("tokenizer_sha256", "tokenizer differs")):
            with self.subTest(key=bad_key):
                self.cache.manifest.update(t5_sha256=file_hash(weights),
                    tokenizer_sha256=digest({"config.json": file_hash(tokenizer / "config.json")}))
                self.cache.manifest[bad_key] = "incorrect-hash"
                with patch("wam.model.backbones.wan22.helpers.loader._resolve_configs", return_value=configs), \
                     patch("wam.model.backbones.wan22.helpers.loader._load_registered_model") as load, \
                     patch("wam.model.backbones.wan22.wan_video_text_encoder.HuggingfaceTokenizer"):
                    self.runtime._load_encoder()
                    load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
