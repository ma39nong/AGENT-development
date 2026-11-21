"""Protect source references and offline-only split boundary without API calls."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.annotate_video_api import run, validate
from scripts.summarize_teacher_pilot import summarize


class TeacherContract(unittest.TestCase):
    frames = [{"t": 0}, {"t": .5}, {"t": 1}]

    def answer(self):
        return {"events": [{"detached_i": 0, "delivered_i": 2, "result": "delivered",
                            "evidence_i": [0, 2], "reason": "candidate"}],
                "observed_adjustments": [], "uncertain_ranges": []}

    def test_valid_candidate_requires_separate_review(self):
        answer = self.answer()
        self.assertIs(validate(answer, self.frames), answer)
        self.assertNotIn("training_ready", answer)

    def test_unknown_detachment_allowed_for_boundary_delivery(self):
        answer = self.answer()
        answer["events"][0]["detached_i"] = None
        validate(answer, self.frames)

    def test_bad_evidence_or_contradiction_rejected(self):
        for patch in ({"delivered_i": None}, {"evidence_i": [20]},
                      {"detached_i": True}, {"detached_i": 2, "delivered_i": 0},
                      {"result": "unknown"}, {"evidence_i": []}):
            answer = self.answer()
            answer["events"][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate(answer, self.frames)

    def test_nontrain_rejected_before_key_or_video_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            for split in ("val", "historical_test", "blind_test"):
                path.write_text(json.dumps({"split": split}))
                with self.subTest(split=split), self.assertRaisesRegex(ValueError, "train only"):
                    run(argparse.Namespace(review=path))

    def test_candidate_index_flags_phase_and_duplicates_without_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(json.dumps({"source_recording_id": "source", "split": "train",
                                                       "prompt_sha256": "test"}))
            inventory = {"recordings": [{"source_recording_id": "source", "split": "train", "frames": 4,
                                          "fps": 1, "coarse_segments": [
                                              {"label": "pickup", "start_frame": 0, "end_frame": 2},
                                              {"label": "harvest", "start_frame": 2, "end_frame": 4}]}]}
            record = {"start": 0, "end": 3, "raw": {"synthetic": True}, "api_s": 1,
                      "status": "complete_structural_only", "answer": self.answer(),
                      "frames": [{"t": t, "sources": {"cam0": {"frame": t}}} for t in (1, 2, 3)]}
            for i in range(2):
                (root / f"chunk-{i:03d}.json").write_text(json.dumps(record))
            result = summarize(inventory, [root])
            self.assertIsNone(result["recordings"]["source"]["unique_success_count"])
            self.assertFalse(result["training_ready"])
            self.assertTrue(result["recordings"]["source"]["requested_ranges_cover_recording"])
            for event in result["candidates"]:
                self.assertIn("detached_s_outside_demonstrated_harvest", event["warnings"])
                self.assertIn("possible_cross_chunk_duplicate", event["warnings"])


if __name__ == "__main__":
    unittest.main()
