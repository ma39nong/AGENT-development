import unittest
from scripts.report_cloud_full_prefix import interpret_current

class CloudPrefixInterpretation(unittest.TestCase):
    def test_complete_observation_after_prose_is_scored_separately(self):
        data='{"phase":"adjust","phase_index":3,"uncertain":false,"reason":"右手旋转"}'
        self.assertEqual(interpret_current(data)[1],'strict_or_fenced_json')
        answer,kind=interpret_current('动作分析：右手旋转。\n'+data)
        self.assertEqual(answer['phase'],'adjust')
        self.assertEqual(kind,'single_json_object_with_extra_prose')
        with self.assertRaises(ValueError):interpret_current(data+'\n'+data)
        with self.assertRaises(ValueError):interpret_current('看起来是调整')

if __name__=='__main__':unittest.main()
