"""Report simple cloud full/prefix baseline; preserve free-form full-video answer."""
import argparse
import json
from pathlib import Path
import statistics

if __package__:
    from .report_cloud_video_replay import normalize_answer
else:
    from report_cloud_video_replay import normalize_answer


def interpret_current(text):
    try:
        return normalize_answer(text), "strict_or_fenced_json"
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    decoder=json.JSONDecoder();candidates=[]
    for pos,char in enumerate(text):
        if char!='{':continue
        try:
            obj,_=decoder.raw_decode(text[pos:])
            if not isinstance(obj,dict) or 'phase' not in obj:continue
            candidates.append(normalize_answer(json.dumps(obj)))
        except (ValueError, KeyError, TypeError, AttributeError):pass
    if len(candidates)!=1:raise ValueError("no unique valid observation object")
    return candidates[0], "single_json_object_with_extra_prose"


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    status=json.loads((args.run/'status.json').read_text())
    if status['status']!='complete':raise ValueError('run incomplete')
    rows=list(map(json.loads,(args.run/'predictions.jsonl').read_text().splitlines()))
    protocol=json.loads((args.run/'protocol.json').read_text())
    root=Path(__file__).resolve().parents[1]
    source=next(r for r in map(json.loads,(root/'data/derived/coarse_sft_20261123_v1/train.jsonl').read_text().splitlines()) if r['source_recording_id']==protocol['source_recording_id'])
    segments=json.loads(Path(source['annotation']).read_text())['segments'];report={}
    for model in protocol['models']:
        rs=[r for r in rows if r['model']==model];full=next(r for r in rs if r['mode']=='full');cases=[]
        labels=('pickup','harvest','adjust','return');matrix={k:dict.fromkeys((*labels,'unknown','invalid'),0) for k in labels}
        for r in rs:
            if r['mode']!='prefix':continue
            a={};error=None;interpretation='invalid'
            try:
                ch=r['response']['choices'][0]
                if ch['finish_reason']!='stop':raise ValueError('incomplete output')
                a,interpretation=interpret_current(ch['message']['content'])
            except (ValueError,KeyError,TypeError,AttributeError) as exc:error=str(exc)
            expected_index=next(i for i,s in enumerate(segments,1) if s['start_frame']<=r['actual_frames']-1<s['end_frame'])
            matrix[r['target_phase']][a.get('phase','invalid')]+=1
            cases.append({'nominal_seconds':r['nominal_seconds'],'actual_end_seconds':r['actual_end_seconds'],'target_phase':r['target_phase'],
                          'target_index':expected_index,'answer':a,'interpretation':interpretation,'valid':error is None,'error':error,
                          'correct':error is None and a['phase']==r['target_phase'],'index_correct':error is None and a.get('phase_index')==expected_index,
                          'api_seconds':r['api_seconds'],'raw_response':r['response']})
        report[model]={'prefix_samples':len(cases),'prefix_valid':sum(c['valid'] for c in cases),'prefix_correct':sum(c['correct'] for c in cases),
            'prefix_index_correct':sum(c['index_correct'] for c in cases),'strict_or_fenced_json_count':sum(c['interpretation']=='strict_or_fenced_json' for c in cases),
            'extra_prose_with_unique_json_count':sum(c['interpretation']=='single_json_object_with_extra_prose' for c in cases),'confusion':matrix,
            'recall':{k:v[k]/sum(v.values()) if sum(v.values()) else None for k,v in matrix.items()},
            'prefix_api_median_s':statistics.median(c['api_seconds'] for c in cases if c['api_seconds'] is not None),
            'prefix_api_max_s':max(c['api_seconds'] for c in cases if c['api_seconds'] is not None),
            'full_video_api_s':full['api_seconds'],'full_video_response':full['response'],'prefix_cases':cases}
    report['reference_boundaries_s']=[s['end_frame']/source['fps'] for s in segments[:-1]]
    report['limitations']=['one previously inspected train recording','full-video response is offline segmentation, prefixes are current observation, not equivalent tasks',
                            'native MP4 prefix streamcopy PTS and endpoint pixels verified; server internal sampling unmeasured',
                            'no prior text or agent memory, no counts or robot control','API timing excludes clip preparation; no live queue simulation']
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({model:{k:v for k,v in result.items() if k not in ('prefix_cases','full_video_response')} for model,result in report.items() if isinstance(result,dict)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
