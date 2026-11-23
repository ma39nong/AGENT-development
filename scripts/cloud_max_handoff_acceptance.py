"""Fixed Max acceptance only: 3 truncations + 4 handoffs before/after, two repeats."""
import argparse
import base64
import json
from pathlib import Path
import time

if __package__:
    from .annotate_video_api import digest,save
    from .cloud_full_prefix_baseline import prepare_prefix
    from .streaming_capability import cloud_call
    from .handoff_gate import parse_decision,evaluate_gate
else:
    from annotate_video_api import digest,save
    from cloud_full_prefix_baseline import prepare_prefix
    from streaming_capability import cloud_call
    from handoff_gate import parse_decision,evaluate_gate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=False)
    old=root/'runs/cloud_full_prefix_baseline_20261123_v2';prior=json.loads((old/'protocol.json').read_text())
    row=next(r for r in map(json.loads,(root/'data/derived/coarse_sft_20261123_v1/train.jsonl').read_text().splitlines()) if r['source_recording_id']==prior['source_recording_id'])
    source=Path(row['video']);fps=row['fps']
    if digest(source)!=row['video_sha256'] or digest(Path(row['annotation']))!=row['annotation_sha256']:raise ValueError('source changed')
    seg=json.loads(Path(row['annotation']).read_text())['segments'];cases=[]
    for nominal in (30,40,80):
        j=next(j for j in prior['jobs'] if j['mode']=='prefix' and j['nominal_seconds']==nominal)
        if digest(Path(j['path']))!=j['video_sha256']:raise ValueError('original prefix changed')
        index=next(i for i,s in enumerate(seg,1) if s['start_frame']<=j['actual_frames']-1<s['end_frame'])
        cases.append({'id':f'truncated-{nominal}','group':'original_truncation','path':j['path'],'actual_end_seconds':j['actual_end_seconds'],
          'sha256':j['video_sha256'],'stage':index,'current_skill':seg[index-1]['label'],'planned_next':seg[index]['label'],'stop_requested':False})
    for i,(before,after) in enumerate(zip(seg,seg[1:]),1):
        boundary=after['start_frame']/fps
        for offset in (-1,1):
            nominal=boundary+offset;path=args.output/f'B{i}-{offset:+d}.mp4';covered,decoded=prepare_prefix(source,nominal,path,fps)
            cases.append({'id':f'B{i}-{offset:+d}','group':'handoff_before' if offset<0 else 'handoff_after','path':str(path),'actual_end_seconds':covered/fps,
              'decoded_frames':decoded,'sha256':digest(path),'stage':i,'current_skill':before['label'],'planned_next':after['label'],'stop_requested':True,
              'reference_boundary_seconds':boundary,'completion_reference':'unknown_requires_visual_review' if offset<0 else 'next_phase_started_by_coarse_reference'})
    cases.sort(key=lambda c:c['actual_end_seconds']);prompt=(root/'prompts/cloud_max_handoff_zh.txt').read_text()
    save(args.output/'protocol.json',{'model':'qwen3.8-max','enable_thinking':False,'response_format':{'type':'json_object'},'max_tokens':96,
         'prompt':prompt,'prompt_sha256':digest(root/'prompts/cloud_max_handoff_zh.txt'),'cases':cases,'repeats':2,'requests_maximum':22,
         'source_recording_id':row['source_recording_id'],'input':'causal cumulative video prefix, no reencoding, no model-history text',
         'scope':'fixed skill-completion conditions, no counting or model/prompt sweep','executor_context':'explicit simulated current skill/plan/stop request; not real hardware feedback',
         'gate_age_limit_seconds':2,'gate_age_limit_status':'offline screening parameter, not measured robot deadline',
         'visual_reference_warning':'coarse phase onset is not independent prior-skill completion ground truth; before cases require review',
         'robot_control':False,'no_retries':True})
    results=[]
    try:
        for repeat in (1,2):
            for c in cases:
                request={'task_version':1,'skill_epoch':c['stage'],'current_skill':c['current_skill'],'planned_next':c['planned_next'],'stop_requested':c['stop_requested']}
                context={**request,'observation_cutoff_seconds':c['actual_end_seconds']}
                begin=time.perf_counter()
                payload={'model':'qwen3.8-max','enable_thinking':False,'stream':False,'max_tokens':96,'response_format':{'type':'json_object'},
                  'messages':[{'role':'system','content':prompt},{'role':'user','content':[
                    {'type':'video_url','video_url':{'url':'data:video/mp4;base64,'+base64.b64encode(Path(c['path']).read_bytes()).decode()}},
                    {'type':'text','text':json.dumps(context,ensure_ascii=False)}]}]}
                response={};d=None;error=None;api=None
                try:
                    response,api=cloud_call(payload);ch=response['choices'][0]
                    if ch['finish_reason']!='stop':raise ValueError('incomplete output: '+str(ch['finish_reason']))
                    d=parse_decision(ch['message']['content'])
                except Exception as exc:error=type(exc).__name__+': '+str(exc)
                age=time.perf_counter()-begin
                # These are fault-test fixtures, never real executor acknowledgements.
                executor={'task_version':1,'skill_epoch':c['stage']+(c['group']=='handoff_after'),
                  'current_skill':c['planned_next'] if c['group']=='handoff_after' else c['current_skill'],
                  'connected':True,'safe_point_confirmed':False,'pending_actions':0,'transition_verified':False}
                gate=evaluate_gate(d,request,executor,age,2)
                r={'case_id':c['id'],'group':c['group'],'repeat':repeat,'request':request,'answer':d,'valid':d is not None,'error':error,
                   'api_seconds':api,'observation_age_seconds':age,'raw_response':response,'simulated_executor':executor,'offline_gate':gate,
                   'frame_cutoff':c['actual_end_seconds'],'gate_fixture':'no real safe-point or capability acknowledgement; post cases simulate advanced skill epoch'}
                results.append(r)
                with (args.output/'predictions.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
                save(args.output/'status.json',{'status':'running','completed':len(results),'total':22})
                print(json.dumps({'completed':len(results),'case':c['id'],'answer':d,'error':error,'api_s':api}),flush=True)
                if error and 'HTTP 400' in error:raise ValueError('structured-mode rejected; no silent fallback or retries')
                if len(results)>=3 and all(r['error'] for r in results[-3:]):raise ValueError('three invalid responses; stop fixed acceptance')
        save(args.output/'status.json',{'status':'complete','completed':len(results)})
    except Exception as exc:
        save(args.output/'status.json',{'status':'failed','completed':len(results),'error':str(exc)});raise

if __name__=='__main__':main()
