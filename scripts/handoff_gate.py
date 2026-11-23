"""Fail-closed offline handoff request gate. No robot/control transport."""
import json
import math


def parse_decision(raw):
    d=json.loads(raw)
    if not isinstance(d,dict) or set(d)!={'completion','decision','next_skill'}:raise ValueError('exact short schema required')
    if d['completion'] not in {'yes','no','unknown'} or d['decision'] not in {'continue','hold','handoff','scene_changed'} or d['next_skill'] not in {'harvest','adjust','return','none'}:raise ValueError('unsupported enum')
    if d['decision']=='handoff':
        if d['completion']!='yes' or d['next_skill']=='none':raise ValueError('contradictory handoff')
    elif d['next_skill']!='none':raise ValueError('next skill without handoff')
    return d


def evaluate_gate(decision, request, executor, observation_age_s, max_age_s):
    if decision is not None:
        try:decision=parse_decision(json.dumps(decision))
        except (ValueError,TypeError):return {'allow':False,'reason':'invalid_output'}
    required={'task_version','skill_epoch','current_skill'}
    if not required.issubset(request) or not required.issubset(executor) or 'planned_next' not in request:
        return {'allow':False,'reason':'invalid_execution_state'}
    checks=[(decision is not None,'invalid_output'),
      (all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (observation_age_s,max_age_s)),'invalid_clock'),
      (request['task_version']==executor['task_version'] and request['skill_epoch']==executor['skill_epoch'] and request['current_skill']==executor['current_skill'],'stale_execution'),
      (executor.get('connected') is True,'disconnected'),
      (request.get('stop_requested') is True,'no_stop_request'),
      (executor.get('safe_point_confirmed') is True,'no_safe_point'),
      (executor.get('pending_actions')==0 and type(executor.get('pending_actions')) is int,'pending_actions'),
      (executor.get('transition_verified') is True,'unverified_transition')]
    for passed,reason in checks:
        if not passed:return {'allow':False,'reason':reason}
    if observation_age_s>max_age_s:return {'allow':False,'reason':'expired_observation'}
    if decision['decision']!='handoff' or decision['completion']!='yes':return {'allow':False,'reason':'completion_not_ready'}
    if decision['next_skill']!=request['planned_next']:return {'allow':False,'reason':'wrong_transition'}
    return {'allow':True,'reason':'offline_request_eligible'}
