import unittest

from scripts.evaluate_coarse_test import summarize


class TestSummary(unittest.TestCase):
    def test_invalid_outputs_are_counted_but_not_scored_as_zero_error(self):
        rows = [{"generation_seconds": 2, "metrics": {"valid": True,
                 "boundary_errors_s": [0, .5, 1, 2], "mean_segment_iou": .8}},
                {"generation_seconds": 4, "metrics": {"valid": False, "error": "wrong stage count"}}]
        result = summarize(rows)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["valid_outputs"], 1)
        self.assertEqual(result["boundary_mae_s_valid_only"], .875)
        self.assertEqual(result["boundary_within_0_5s_valid_only"], .5)
        self.assertEqual(result["boundary_within_1s_valid_only"], .75)
        self.assertEqual(result["generation_seconds_median"], 3)

    def test_all_invalid_is_not_reported_as_perfect_accuracy(self):
        result = summarize([{"generation_seconds": 2, "metrics": {"valid": False}}])
        self.assertEqual(result["valid_outputs"], 0)
        self.assertIsNone(result["boundary_mae_s_valid_only"])
        self.assertIsNone(result["mean_segment_iou_valid_only"])


if __name__ == "__main__":
    unittest.main()
