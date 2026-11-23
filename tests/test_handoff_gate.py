import json
import unittest
from scripts.handoff_gate import parse_decision,evaluate_gate

class HandoffGate(unittest.TestCase):
    def setUp(self):
        self.d={'completion':'yes','decision':'handoff','next_skill':'adjust'}
        self.r={'task_version':1,'skill_epoch':2,'current_skill':'harvest','planned_next':'adjust','stop_requested':True}
        self.e={'task_version':1,'skill_epoch':2,'current_skill':'harvest','connected':True,'safe_point_confirmed':True,'pending_actions':0,'transition_verified':True}
    def gate(self,**kw):return evaluate_gate(kw.get('d',self.d),kw.get('r',self.r),kw.get('e',self.e),kw.get('age',1),2)
    def test_only_ready_request_allowed(self):self.assertTrue(self.gate()['allow'])
    def test_all_external_failures_block(self):
        for key,value in [('task_version',2),('skill_epoch',3),('current_skill','return'),('connected',False),('safe_point_confirmed',False),('pending_actions',1),('transition_verified',False)]:
            with self.subTest(key=key):self.assertFalse(self.gate(e={**self.e,key:value})['allow'])
    def test_unknown_late_invalid_and_no_stop_block(self):
        self.assertFalse(self.gate(d=None)['allow'])
        self.assertFalse(self.gate(d={})['allow'])
        self.assertFalse(self.gate(e={})['allow'])
        self.assertFalse(self.gate(age=3)['allow'])
        self.assertFalse(self.gate(age=float('nan'))['allow'])
        self.assertFalse(self.gate(r={**self.r,'stop_requested':False})['allow'])
        self.assertFalse(self.gate(d={'completion':'unknown','decision':'hold','next_skill':'none'})['allow'])
    def test_wrong_transition_blocked(self):self.assertFalse(self.gate(d={**self.d,'next_skill':'return'})['allow'])
    def test_schema_no_prose_or_conflicting_fields(self):
        self.assertEqual(parse_decision(json.dumps(self.d)),self.d)
        for raw in ['```json\n'+json.dumps(self.d)+'\n```',json.dumps({**self.d,'completion':'unknown'}),json.dumps({**self.d,'extra':1})]:
            with self.assertRaises(ValueError):parse_decision(raw)

if __name__=='__main__':unittest.main()
