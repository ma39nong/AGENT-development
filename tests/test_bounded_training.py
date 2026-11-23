import json
import unittest
from scripts.train_coarse_lora import evaluate_answer, current_phase_metrics, observation_report

class BoundedTraining(unittest.TestCase):
    def answer(self, phase, index=None):
        return json.dumps(dict(phase=phase, phase_index=index, uncertain=False, reason=''))

    def test_nullable_harvest_and_invalid_output(self):
        truth = self.answer('harvest')
        self.assertTrue(evaluate_answer(truth, truth)['valid'])
        self.assertTrue(current_phase_metrics(truth, truth)['current_label_correct'])
        self.assertFalse(evaluate_answer(self.answer('nonsense'), truth)['valid'])
        self.assertFalse(evaluate_answer(self.answer('harvest', True), truth)['valid'])
        self.assertFalse(evaluate_answer('broken', truth)['valid'])
        self.assertFalse(evaluate_answer('[]', truth)['valid'])
        self.assertFalse(current_phase_metrics('[]', truth)['current_label_correct'])

    def test_recall_invalid_denominator_false_return_and_ordinal(self):
        rows = []
        for target, raw, reference in [('pickup', self.answer('return', 5), 1),
                                       ('harvest', self.answer('harvest'), 2),
                                       ('adjust', 'broken', 3)]:
            truth = self.answer(target, None if target == 'harvest' else reference)
            rows.append(dict(target=truth, raw_output=raw, metrics=evaluate_answer(raw, truth), reference_phase_index=reference))
        report = observation_report(rows)
        self.assertEqual(report['false_return_count'], 1)
        self.assertEqual(report['per_class_recall']['adjust'], 0)
        self.assertEqual(report['macro_recall_present_classes'], 1/3)
        self.assertEqual(report['reference_full_five_stage_index_accuracy'], 0)
        self.assertEqual(report['non_null_target_index_samples'], 2)

if __name__ == '__main__':
    unittest.main()
