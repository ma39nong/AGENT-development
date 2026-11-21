import unittest

from scripts.prompt_plan_guard import check_plan


class PromptPlanGuard(unittest.TestCase):
    def test_synthetic_four_without_adjustment_is_allowed(self):
        result = check_plan([1., 2., 3., 4.], 4, "return", uncertain=False)
        self.assertTrue(result["logical_conditions_satisfied"])
        self.assertFalse(result["control_eligible"])

    def test_synthetic_two_then_two_uses_execution_timestamps(self):
        result = check_plan([1., 2., 5., 6.], 4, "return", uncertain=False,
                            required_first_batch=2, adjustment_started_s=3., adjustment_finished_s=4.)
        self.assertTrue(result["logical_conditions_satisfied"])

    def test_three_then_one_cannot_be_relabelled_two_then_two(self):
        result = check_plan([26., 36., 45., 76.], 4, "return", uncertain=False,
                            required_first_batch=2, adjustment_started_s=46.9, adjustment_finished_s=61.267)
        self.assertIn("first_batch_count_violates_instruction", result["issues"])

    def test_unknown_short_target_and_missing_adjustment_block_completion(self):
        result = check_plan([1.], 4, "return", uncertain=True, required_first_batch=2)
        self.assertEqual(set(result["issues"]), {"perception_or_history_uncertain", "target_not_reached", "required_adjustment_missing"})

    def test_quota_is_task_parameter_and_gates_next_harvest(self):
        result = check_plan([1., 2., 3.], 5, "harvest", uncertain=False, required_first_batch=3)
        self.assertIn("required_adjustment_before_next_harvest", result["issues"])
        result = check_plan([1., 2., 3., 6., 7.], 5, "return", uncertain=False,
                            required_first_batch=3, adjustment_started_s=4., adjustment_finished_s=5.)
        self.assertTrue(result["logical_conditions_satisfied"])


if __name__ == "__main__":
    unittest.main()
