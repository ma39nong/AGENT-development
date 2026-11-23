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


class BoundarySampling(unittest.TestCase):
    def test_only_one_harvest_position_changes_with_fixed_budget(self):
        from scripts.compare_bounded_history import training_cutoffs
        segments=[dict(label=label,start_frame=i*300,end_frame=(i+1)*300) for i,label in enumerate(['pickup','harvest','adjust','harvest','return'])]
        for identity in ['a','b','c','d']:
            original=training_cutoffs(segments,30,identity)
            focused=training_cutoffs(segments,30,identity,True)
            self.assertEqual(len(focused),5)
            self.assertEqual(len(original-focused),1)
            self.assertTrue(focused-original <= {330,930})
            self.assertTrue({150,750,1350}<=focused)

class BoundaryGrouping(unittest.TestCase):
    def test_close_handoffs_use_exact_probe_origin(self):
        from scripts.summarize_bounded_replay import boundary_group
        segments=[{'end_frame':257},{'end_frame':1224},{'end_frame':1282},{'end_frame':2441},{'end_frame':2554}]
        self.assertEqual(boundary_group(segments,1252,30),'B3_before_1s')
        self.assertEqual(boundary_group(segments,1254,30),'B2_after_1s')
        self.assertEqual(boundary_group(segments,1253,30),'ambiguous_boundary_probe')

if __name__ == '__main__':
    unittest.main()
