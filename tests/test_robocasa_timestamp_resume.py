"""Timestamp regression and narrow legacy-plan compatibility checks."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wam.datasets import robocasa_visual_cache as c


class TimestampResumeTests(unittest.TestCase):
    def setUp(self):
        self.episode = {"task": "fixture", "episode": 189, "length": 2230}
        self.pts = np.arange(2230, dtype=np.float64) * 0.05
        self.stamps = np.arange(2230, dtype=np.float64) * 0.04999995231628418

    def test_reproduced_numeric_clock_drift_is_accepted(self):
        self.assertFalse(np.allclose(self.pts, self.stamps, atol=1e-4, rtol=0))
        c.validate_video_timestamps(self.pts, self.stamps, self.episode)

    def test_normal_timestamps_are_accepted(self):
        c.validate_video_timestamps(self.pts, self.pts, self.episode)

    def test_one_frame_offset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "timestamps differ"):
            c.validate_video_timestamps(self.pts + 0.05, self.stamps, self.episode)

    def test_shift_above_one_millisecond_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "timestamps differ"):
            c.validate_video_timestamps(self.pts + 0.0011, self.pts, self.episode)

    def test_missing_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame count"):
            c.validate_video_timestamps(self.pts[:-1], self.stamps, self.episode)

    def test_repeated_frame_timestamp_is_rejected(self):
        pts = self.pts.copy()
        pts[30] = pts[29]
        with self.assertRaisesRegex(ValueError, "20Hz"):
            c.validate_video_timestamps(pts, self.stamps, self.episode)

    def test_nonfinite_video_timestamp_is_rejected(self):
        pts = self.pts.copy()
        pts[30] = np.nan
        with self.assertRaisesRegex(ValueError, "20Hz"):
            c.validate_video_timestamps(pts, self.stamps, self.episode)

    def compatibility_fixture(self, root):
        key = "datasets/robocasa_visual_cache.py"
        old = {key: c.LEGACY_CACHE_IMPLEMENTATION, "encoder.py": "unchanged"}
        current = {key: "reviewed-fix-hash", "encoder.py": "unchanged"}
        plan = {"sha256": "original-plan", "implementation": old}
        value = {
            "schema": "wam.robocasa.timestamp_resume.v1",
            "plan_sha256": plan["sha256"],
            "old_implementation": old,
            "new_implementation": current,
            "timestamp_policy": c.TIMESTAMP_POLICY,
        }
        path = Path(root) / "compatibility.json"
        c.atomic_json(path, value)
        return plan, current, value, path

    def test_exact_implementation_does_not_need_compatibility(self):
        with patch.object(c, "implementation_identity", return_value={"test": "hash"}):
            result = c.validate_implementation({"implementation": {"test": "hash"}})
        self.assertEqual(result["mode"], "exact")

    def test_legacy_requires_explicit_compatibility(self):
        with tempfile.TemporaryDirectory() as root:
            plan, current, _, _ = self.compatibility_fixture(root)
            with patch.object(c, "implementation_identity", return_value=current):
                with self.assertRaisesRegex(ValueError, "reviewed"):
                    c.validate_implementation(plan)

    def test_reviewed_fix_preserves_old_plan(self):
        with tempfile.TemporaryDirectory() as root:
            plan, current, _, path = self.compatibility_fixture(root)
            before = copy.deepcopy(plan)
            with patch.object(c, "implementation_identity", return_value=current):
                result = c.validate_implementation(plan, path)
            self.assertEqual(plan, before)
            self.assertEqual(result["compatibility_sha256"], c.file_hash(path))

    def test_encoder_change_is_rejected_even_if_record_claims_it(self):
        with tempfile.TemporaryDirectory() as root:
            plan, current, value, path = self.compatibility_fixture(root)
            current["encoder.py"] = "different"
            c.atomic_json(path, value)
            with patch.object(c, "implementation_identity", return_value=current):
                with self.assertRaisesRegex(ValueError, "only change"):
                    c.validate_implementation(plan, path)

    def test_other_plan_or_code_or_policy_is_rejected(self):
        for field in ["plan_sha256", "new_implementation", "timestamp_policy"]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                plan, current, value, path = self.compatibility_fixture(root)
                value[field] = "wrong"
                c.atomic_json(path, value)
                with patch.object(c, "implementation_identity", return_value=current):
                    with self.assertRaises(ValueError):
                        c.validate_implementation(plan, path)


if __name__ == "__main__":
    unittest.main()
