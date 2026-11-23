import json
import unittest

import torch
import torch.nn.functional as F

from scripts.build_coarse_sft import target_json
from scripts.train_coarse_lora import assistant_start, evaluate_answer, suffix_loss


class CoarseTargets(unittest.TestCase):
    def segments(self):
        return [{"label": label, "start_frame": a, "end_frame": b, "reviewed": True}
                for label, a, b in (("pickup", 0, 30), ("harvest", 30, 120), ("return", 120, 150))]

    def test_whole_video_target_does_not_force_five_stages_or_add_counts(self):
        raw = target_json(self.segments(), 150, 30)
        self.assertIn('"start_seconds":0.00', raw)
        answer = json.loads(raw)
        self.assertEqual(set(answer), {"segments"})
        self.assertEqual(len(answer["segments"]), 3)
        self.assertEqual(answer["segments"][-1]["end_seconds"], 5.)

    def test_gaps_and_unreviewed_targets_rejected(self):
        for key, value in (("start_frame", 31), ("reviewed", False), ("label", "fine_grasp")):
            segments = self.segments()
            segments[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                target_json(segments, 150, 30)

    def test_generation_checks_continuity_and_boundary_error(self):
        truth = target_json(self.segments(), 150, 30)
        self.assertEqual(evaluate_answer(truth, truth)["boundary_mae_s"], 0.)
        prediction = json.loads(truth)
        prediction["segments"][0]["end_seconds"] = 1.5
        prediction["segments"][1]["start_seconds"] = 1.5
        self.assertEqual(evaluate_answer(json.dumps(prediction), truth)["boundary_mae_s"], .25)
        prediction["segments"][1]["start_seconds"] = 1.6
        self.assertFalse(evaluate_answer(json.dumps(prediction), truth)["valid"])


class AssistantLoss(unittest.TestCase):
    def test_suffix_loss_matches_full_masked_loss_and_gradients(self):
        torch.manual_seed(1)
        full = torch.randn(2, 8, 11, requires_grad=True)
        target = torch.randint(0, 11, (2, 8))
        labels = target.clone()
        labels[:, :5] = -100
        expected = F.cross_entropy(full[:, :-1].contiguous().view(-1, 11), labels[:, 1:].reshape(-1))
        actual = suffix_loss(full[:, 4:], target[:, 5:])
        torch.testing.assert_close(expected, actual)
        g1 = torch.autograd.grad(expected, full, retain_graph=True)[0]
        g2 = torch.autograd.grad(actual, full)[0]
        torch.testing.assert_close(g1, g2)
        self.assertEqual(torch.count_nonzero(g2[:, :4]), 0)

    def test_mask_requires_exactly_one_answer_header(self):
        self.assertEqual(assistant_start([1, 2, 3, 4, 5], [3, 4]), 4)
        for ids in ([1, 2, 3], [3, 4], [3, 4, 1, 3, 4, 2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                assistant_start(ids, [3, 4])


if __name__ == "__main__":
    unittest.main()
