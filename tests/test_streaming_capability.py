"""Never accept future evidence or silently turn candidate memory into truth."""
import copy
import unittest

from scripts.streaming_capability import validate


class StreamingContract(unittest.TestCase):
    def answer(self):
        return {"phase": "harvest", "delivery_seconds": [36.5], "confirmed_total": 2, "uncertain": False,
                "target4": {"request": "harvest", "reason": "candidate"},
                "two_then_two": {"request": "adjust", "reason": "candidate"}}

    def test_consistent_memory_does_not_add_control_authority(self):
        answer = self.answer()
        self.assertIs(validate(answer, 27, 37, "recent_memory", [26.5]), answer)
        self.assertNotIn("control_eligible", answer)

    def test_future_duplicate_invalid_count_and_prompt_rejected(self):
        patches = [{"delivery_seconds": [38]}, {"delivery_seconds": [36, 36]},
                   {"delivery_seconds": [26]}, {"delivery_seconds": [True]},
                   {"confirmed_total": True}, {"confirmed_total": 3},
                   {"target4": {"request": "move_joints", "reason": "no"}}]
        for patch in patches:
            answer = copy.deepcopy(self.answer())
            answer.update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate(answer, 27, 37, "recent_memory", [26.5])

    def test_unknown_total_allowed_without_prior_history(self):
        answer = self.answer()
        answer.update(confirmed_total=None, uncertain=True)
        validate(answer, 27, 37, "recent_only", [])

    def test_prefix_count_must_match_all_past_events(self):
        answer = self.answer()
        with self.assertRaisesRegex(ValueError, "ledger"):
            validate(answer, 0, 37, "prefix", [])
        answer["delivery_seconds"] = [26.5, 36.5]
        validate(answer, 0, 37, "prefix", [])


if __name__ == "__main__":
    unittest.main()
