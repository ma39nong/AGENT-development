import unittest

from scripts.compare_bounded_history import parse_observation, select_history, summarize


class BoundedHistory(unittest.TestCase):
    def record(self, cutoff, phase, index):
        return {"observation_cutoff_seconds": cutoff,
                "prediction": {"valid": True, "phase": phase, "phase_index": index, "uncertain": True},
                "evidence": [{"source_last_seconds": cutoff-.1}]}

    def test_history_budget_distinctness_and_window_cutoff(self):
        records = [self.record(40, "return", 5), self.record(20, "harvest", 2),
                   self.record(15, "harvest", 2), self.record(10, "pickup", 1)]
        selected = select_history(records, 30, len, budget=10000)
        self.assertEqual([r[0]["observed_until"] for r in selected], [20, 10])
        self.assertEqual(select_history(records, 30, len, budget=1), [])

    def test_parser_and_invalid_results_do_not_disappear_from_accuracy(self):
        self.assertEqual(parse_observation('{"phase":"harvest","phase_index":null,"uncertain":true}')["phase"], "harvest")
        with self.assertRaises(ValueError):
            parse_observation('{"phase":"harvest","phase_index":true,"uncertain":true}')
        rows = [{"valid": True, "label_correct": True, "index_correct": False, "prepare_generate_s": 1,
                 "input_tokens": 100, "history": []},
                {"valid": False, "label_correct": False, "index_correct": False, "prepare_generate_s": 3,
                 "input_tokens": 200, "history": []}]
        result = summarize(rows)
        self.assertEqual(result["current_label_accuracy_all"], .5)
        self.assertEqual(result["valid_outputs"], 1)


if __name__ == "__main__":
    unittest.main()
