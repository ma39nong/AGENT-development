"""Data intake must preserve heldout isolation and source identity."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_subtask_data import audit, digest, review_queue


class DataIntake(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack = self.root / "A/package"
        (self.pack / "meta").mkdir(parents=True)
        info = {"fps": 30, "chunks_size": 1000,
                "features": {"observation.images." + c: {} for c in ("cam0", "cam1", "cam2")},
                "data_path": "data/{episode_index}.parquet",
                "video_path": "videos/{video_key}/{episode_index}.mp4"}
        (self.pack / "meta/info.json").write_text(json.dumps(info))
        self.records, episodes = [], []
        for i, split in enumerate(("train", "val", "blind_test")):
            group = f"source-{i}"
            episodes.append({"episode_index": i, "source_recording_id": group, "length": 100})
            p = self.pack / "annotations/subtasks/coarse" / f"episode_{i:06d}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            if split == "blind_test":
                p.write_text("not valid JSON: sealed labels must not be read")
            else:
                p.write_text(json.dumps({"source": {"source_recording_id": group, "episode_index": i,
                                                   "length": 100, "fps": 30},
                                         "segments": [{"label": "harvest", "start_frame": 0,
                                                       "end_frame": 100, "reviewed": True}]}))
            self.records.append({"source_recording_id": group, "split": split,
                                 "package": "A/package", "source_episode_index": i,
                                 "length": 100, "annotation_sha256": digest(p)})
            for relative in [f"data/{i}.parquet", *[f"videos/observation.images.{c}/{i}.mp4"
                                                     for c in ("cam0", "cam1", "cam2")]]:
                f = self.pack / relative
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"fixture, not actual decoded video/parquet")
        (self.pack / "meta/episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
        self.lock = self.root / "lock.json"
        self.write_lock()

    def write_lock(self):
        self.lock.write_text(json.dumps({"recordings": self.records,
                                        "counts": {"train": 1, "val": 1, "blind_test": 1}}))

    def test_sealed_labels_not_read_and_review_queue_train_only(self):
        result = audit(self.root, self.lock)
        self.assertEqual(result["summary"]["heldout_recordings_sealed"], 1)
        self.assertFalse(result["summary"]["video_pixels_decoded"])
        queue = review_queue(result, 5)
        self.assertEqual([r["source_recording_id"] for r in queue], ["source-0"])
        self.assertEqual(queue[0]["events"], [])
        self.assertIsNone(queue[0]["task_goal_count"])
        self.assertEqual(queue[0]["review_status"], "unreviewed")

    def test_duplicate_source_cannot_cross_split(self):
        self.records[1]["source_recording_id"] = self.records[0]["source_recording_id"]
        self.write_lock()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit(self.root, self.lock)

    def test_missing_camera_reported_without_inventing_ready_sample(self):
        (self.pack / "videos/observation.images.cam2/0.mp4").unlink()
        result = audit(self.root, self.lock)
        self.assertEqual(result["summary"]["train_val_recordings_with_all_three_videos"], 1)
        self.assertEqual(review_queue(result, 5), [])
        self.assertFalse(result["summary"]["ready_for_qwen_training"])

    def test_changed_annotation_rejected(self):
        (self.pack / "annotations/subtasks/coarse/episode_000000.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "frozen lock"):
            audit(self.root, self.lock)

    def test_fallback_requires_matching_video_digest(self):
        missing = self.pack / "videos/observation.images.cam2/0.mp4"
        missing.unlink()
        donor = self.root / "donors/raw"
        relative = "videos/observation.images.cam2/0.mp4"
        f = donor / relative
        f.parent.mkdir(parents=True)
        f.write_bytes(b"original video")
        annpath = self.pack / "annotations/subtasks/coarse/episode_000000.json"
        ann = json.loads(annpath.read_text())
        ann["source"].update(dataset=str(donor), sha256={relative: digest(f)})
        annpath.write_text(json.dumps(ann))
        self.records[0]["annotation_sha256"] = digest(annpath)
        self.write_lock()
        result = audit(self.root, self.lock, self.root / "donors")
        self.assertEqual(result["summary"]["fallback_video_hashes_verified"], 1)
        self.assertEqual(len(review_queue(result, 5)), 1)
        f.write_bytes(b"wrong bytes")
        with self.assertRaisesRegex(ValueError, "Fallback video bytes"):
            audit(self.root, self.lock, self.root / "donors")


if __name__ == "__main__":
    unittest.main()
