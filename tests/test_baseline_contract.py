"""Reject wrong evidence/time provenance before interpreting model suggestions."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.local_baseline import load_frames, validate_answer


class AnswerContract(unittest.TestCase):
    frames = [{"i": i, "source_timestamp_s": t} for i, t in enumerate([1.0, 1.0, 2.0, 3.0])]

    def answer(self, **changes):
        value = {"d": 1, "h": 0, "r": 2, "e": 3, "next_skill": "continue_harvest"}
        value.update(changes)
        return validate_answer(json.dumps(value), self.frames)

    def test_valid_order_is_only_structural_not_visual_or_executable(self):
        result = self.answer()
        self.assertTrue(result["valid"])
        self.assertFalse(result["visual_evidence_audited"])
        self.assertFalse(result["executable"])

    def test_same_time_two_views_cannot_prove_a_transition(self):
        self.assertFalse(self.answer(r=1)["valid"])

    def test_invented_frame_boolean_and_unsupported_skill_rejected(self):
        for changes in ({"h": 100}, {"d": True}, {"next_skill": "move_joints"}):
            with self.subTest(changes=changes):
                self.assertFalse(self.answer(**changes)["valid"])

    def test_unknown_does_not_carry_confirmed_evidence(self):
        self.assertFalse(self.answer(d=-1)["valid"])
        self.assertTrue(self.answer(d=-1, h=None, r=None, e=None, next_skill="observe")["valid"])


class ReplayContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest_path = self.root / "manifest.json"
        self.rows = []
        for i, view in enumerate(("head", "left_wrist", "right_wrist", "head_crop")):
            path = self.root / (view + ".png")
            Image.new("RGB", (32, 32)).save(path)
            self.rows.append({"i": i, "view": view, "path": path.name,
                              "source_timestamp_s": 1.0, "source_frame": 30,
                              "source_camera": "cam0" if view in ("head", "head_crop") else view,
                              "width": 32, "height": 32,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.rows[-1]["crop"] = {"parent_path": "head.png", "source_size": [32, 32],
                                  "xyxy_pixels": [0, 0, 32, 32], "mode": "fixed_diagnostic",
                                  "uses_future_coordinates": False}

    def load(self):
        self.manifest_path.write_text(json.dumps({"cutoff_s": 1.0, "frames": self.rows}))
        return load_frames(self.manifest_path, True)

    def test_current_head_crop_has_related_source(self):
        self.assertEqual(len(self.load()[1]), 4)

    def test_future_frame_rejected(self):
        self.rows[0]["source_timestamp_s"] = 1.1
        with self.assertRaisesRegex(ValueError, "future"):
            self.load()

    def test_future_coordinates_rejected(self):
        self.rows[-1]["crop"]["uses_future_coordinates"] = True
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.load()

    def test_repeated_id_and_changed_pixels_rejected(self):
        self.rows[1]["i"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()
        self.rows[1]["i"] = 1
        (self.root / "head.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.load()


if __name__ == "__main__":
    unittest.main()
