"""Separate strict output formatting from phase recognition; never rewrite replay memory."""
import argparse
import json
from pathlib import Path
import statistics

if __package__:
    from .cloud_video_replay import summarize
    from .compare_bounded_history import parse_observation
else:
    from cloud_video_replay import summarize
    from compare_bounded_history import parse_observation


def normalize_answer(content):
    text=content.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text=text[len('```json\n'):-len('\n```')]
    elif text.startswith('```\n') and text.endswith('\n```'):
        text=text[4:-4]
    return parse_observation(text)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if json.loads((args.run/'status.json').read_text())['status']!='complete':raise ValueError('incomplete replay')
    protocol=json.loads((args.run/'protocol.json').read_text())
    rows=[json.loads(s) for s in (args.run/'predictions.jsonl').read_text().splitlines()]
    normalized=[];wrapped=0
    for r in rows:
        item=dict(r)
        try:
            c=r['raw_response']['choices'][0]
            if c['finish_reason']!='stop':raise ValueError('incomplete')
            a=normalize_answer(c['message']['content'])
            wrapped+=not r['valid']
            item.update(answer=a,valid=True,correct=a['phase']==r['target_phase'])
        except (KeyError,ValueError,TypeError,AttributeError):pass
        normalized.append(item)
    root=Path(__file__).resolve().parents[1]
    source=next(r for r in map(json.loads,(root/'data/derived/coarse_sft_20261123_v1/train.jsonl').read_text().splitlines()) if r['source_recording_id']==protocol['source_recording_id'])
    segments=json.loads(Path(source['annotation']).read_text())['segments']
    transitions=[]
    for old,new in zip(segments,segments[1:]):
        b=new['start_frame']/source['fps'];end=new['end_frame']/source['fps']
        found=next((r for r in normalized if b<r['cutoff_seconds']<=end and r['valid'] and r['answer']['phase']==new['label']),None)
        early=[r for r in normalized if max(0,b-4)<=r['cutoff_seconds']<=b and r['valid'] and r['answer']['phase']==new['label']]
        transitions.append({'from':old['label'],'to':new['label'],'reference_boundary_seconds':b,
            'first_correct_observation_cutoff':found['cutoff_seconds'] if found else None,
            'estimated_result_source_seconds':found['received_source_seconds'] if found else None,
            'estimated_observed_to_result_after_boundary_s':found['received_source_seconds']-b if found else None,
            'premature_new_phase_probes_in_previous_4s':len(early)})
    report={'protocol':protocol,'strict_json':summarize(rows),'phase_after_removing_only_complete_json_fences':summarize(normalized),
        'fenced_json_recovered':wrapped,'transitions':transitions,
        'api_exceed_2s_count':sum(r['api_seconds'] is not None and r['api_seconds']>2 for r in rows),
        'cycle_exceed_2s_count':sum(r['request_cycle_seconds']>2 for r in rows),
        'video_tokens_reported_count':sum(r['raw_response'].get('usage',{}).get('prompt_tokens_details',{}).get('video_tokens',0)>0 for r in rows),
        'max_api_s':max(r['api_seconds'] for r in rows if r['api_seconds'] is not None),
        'limitations':['one previously inspected train recording, no general accuracy estimate','fixed-cadence offline source probes, actual HTTP calls sequential; not a live streaming scheduler',
                       'normalization is posthoc scoring only, original strict failures remained excluded from replay memory',
                       'result source times estimate independent launches; queueing/backpressure/robot execution not measured','counts not tested; no automatic control']}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='protocol'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
