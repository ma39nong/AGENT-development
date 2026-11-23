import json
import unittest

from scripts.infer_coarse_unlabeled import schema_check


class UnlabeledSchema(unittest.TestCase):
    def test_unseen_video_does_not_require_five_phases_or_invent_accuracy(self):
        answer = {"segments": [
            {"phase_index": 1, "label": "pickup", "start_seconds": 0, "end_seconds": 1},
            {"phase_index": 2, "label": "harvest", "start_seconds": 1, "end_seconds": 5},
            {"phase_index": 3, "label": "return", "start_seconds": 5, "end_seconds": 7}]}
        result = schema_check(json.dumps(answer), 7)
        self.assertEqual(result, {"valid": True})
        self.assertFalse(schema_check(json.dumps(answer), 8)["valid"])
        answer["segments"][1]["start_seconds"] = 2
        self.assertFalse(schema_check(json.dumps(answer), 7)["valid"])


if __name__ == "__main__":
    unittest.main()
