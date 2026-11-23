import unittest
from scripts.cloud_video_replay import available_history, summarize

class CloudVideoReplay(unittest.TestCase):
    def row(self,cutoff,received,phase='pickup',valid=True):
        return dict(cutoff_seconds=cutoff,received_source_seconds=received,answer=dict(phase=phase,uncertain=False),valid=valid)

    def test_only_returned_candidates_and_newest_two(self):
        rows=[self.row(2,3),self.row(4,5),self.row(6,7),self.row(8,12)]
        self.assertEqual([r['observed_until'] for r in available_history(rows,8)],[6,4])
        self.assertEqual(available_history(rows,2),[])

    def test_unknown_and_invalid_not_used_as_memory(self):
        self.assertEqual(available_history([self.row(2,3,'unknown'),self.row(4,5,valid=False)],10),[])

    def test_invalid_stays_in_accuracy_denominator(self):
        rows=[dict(target_phase='harvest',answer={'phase':'return'},valid=True,correct=False,api_seconds=1,request_cycle_seconds=2),
              dict(target_phase='harvest',answer={},valid=False,correct=False,api_seconds=None,request_cycle_seconds=3)]
        report=summarize(rows)
        self.assertEqual(report['false_return_count'],1)
        self.assertEqual(report['confusion']['harvest']['invalid'],1)
        self.assertEqual(report['accuracy'],0)


class CloudOutputNormalization(unittest.TestCase):
    def test_only_whole_fenced_json_is_recovered(self):
        from scripts.report_cloud_video_replay import normalize_answer
        data='{"phase":"harvest","phase_index":null,"uncertain":false}'
        self.assertEqual(normalize_answer('```json\n'+data+'\n```')['phase'],'harvest')
        with self.assertRaises(ValueError):normalize_answer('Explanation '+data)

if __name__=='__main__':unittest.main()
