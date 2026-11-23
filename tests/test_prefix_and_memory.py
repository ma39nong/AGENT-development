import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_coarse_prefix_sft import clipped_segments
from scripts.build_coarse_sft import target_json
from scripts.observation_memory import ObservationMemory
from scripts.train_coarse_lora import current_phase_metrics


class PrefixTargets(unittest.TestCase):
    def test_cutoff_clips_current_stage_and_excludes_future(self):
        segments = [{"label": label, "start_frame": a, "end_frame": b, "reviewed": True}
                    for label, a, b in [("pickup", 0, 30), ("harvest", 30, 120), ("adjust", 120, 150)]]
        clipped = clipped_segments(segments, 75)
        answer = json.loads(target_json(clipped, 75, 30))["segments"]
        self.assertEqual([s["label"] for s in answer], ["pickup", "harvest"])
        self.assertEqual(answer[-1]["end_seconds"], 2.5)
        self.assertEqual(segments[1]["end_frame"], 120)
        self.assertEqual(len(clipped_segments(segments, 30)), 1)

    def test_current_phase_is_not_confused_with_history_correctness(self):
        truth = '{"segments":[{"phase_index":2,"label":"harvest"}]}'
        actual = '{"segments":[{"phase_index":4,"label":"harvest"}]}'
        self.assertEqual(current_phase_metrics(actual, truth),
                         {"current_label_correct": True, "current_phase_index_correct": False})
        self.assertFalse(current_phase_metrics("broken", truth)["current_label_correct"])


class MemoryPersistence(unittest.TestCase):
    def test_restart_late_results_time_filter_and_task_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"memory.sqlite"
            m = ObservationMemory(path)
            def add(task, request, cutoff, received):
                return m.append(task, request, cutoff, received, [{"source_last_seconds": cutoff-.1}],
                                {"phase": "harvest"}, "model-v1")
            self.assertTrue(add("task1", "new", 20, 22))
            self.assertTrue(add("task1", "late", 10, 25))
            self.assertFalse(add("task1", "late", 10, 25))
            add("task2", "new", 30, 31)
            self.assertEqual(m.query("task1", cutoff=20, available_at=21), [])
            m.close()
            m = ObservationMemory(path)
            found = m.query("task1", cutoff=30, available_at=26)
            self.assertEqual([r["request"] for r in found], ["new", "late"])
            self.assertFalse(found[0]["control_eligible"])
            self.assertEqual(m.query("task1", cutoff=15, available_at=26)[0]["request"], "late")
            with self.assertRaises(ValueError):
                m.append("task1", "new", 20, 22, [{"source_last_seconds": 19.9}], {"phase": "return"}, "model-v1")
            with self.assertRaises(ValueError):
                m.append("task1", "future", 20, 22, [{"source_last_seconds": 21}], {}, "model-v1")
            m.close()


if __name__ == "__main__":
    unittest.main()
