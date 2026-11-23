"""Small fixed-case latency summaries, with cache and request-order caveats."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    state=json.loads((args.run/'status.json').read_text())
    if state['status']!='complete':raise ValueError('run incomplete')
    rows=list(map(json.loads,(args.run/'predictions.jsonl').read_text().splitlines()))
    modes={}
    for mode in sorted({r['mode'] for r in rows}):
        rs=[r for r in rows if r['mode']==mode];valid=[r for r in rs if 'complete_s' in r['timing']]
        modes[mode]={'samples':len(rs),'strict_valid':sum(r['valid'] for r in rs),
             'median_s':statistics.median(r['timing']['complete_s'] for r in valid),'max_s':max(r['timing']['complete_s'] for r in valid)}
    cases={}
    for case in sorted({r['case'] for r in rows}):
        rs=[r for r in rows if r['case']==case];treatment=rs[1:3]
        cases[case]={'first_control_s':rs[0]['timing'].get('complete_s'),'last_control_s':rs[-1]['timing'].get('complete_s'),
           'middle_treatment_median_s':statistics.median(r['timing']['complete_s'] for r in treatment),
           'same_answer_all_four':len({json.dumps(r['answer'],sort_keys=True) for r in rs})==1,
           'requests':[{'mode':r['mode'],'seconds':r['timing'].get('complete_s'),'headers_s':r['timing'].get('headers_s'),
               'cached_tokens':r['response'].get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens'),
               'video_tokens':r['response'].get('usage',{}).get('prompt_tokens_details',{}).get('video_tokens'),
               'completion_tokens':r['response'].get('usage',{}).get('completion_tokens'),
               'payload_bytes':r.get('payload_bytes'),'socket_fingerprint':r['timing'].get('socket_fingerprint'),'answer':r['answer']} for r in rs]}
    report={'run_status':state,'mode_summary':modes,'cases':cases,
        'max_body_read_after_headers_s':max(r['timing']['complete_s']-r['timing']['headers_s'] for r in rows if 'complete_s' in r['timing']),
        'limitations':['tiny fixed-case ABBA test; no independent p95/p99 estimate','provider cache and queue are not controlled; first control vs middle treatments alone cannot prove speedup',
            'headers wait includes connection/upload/proxy/provider compute; cannot separate these without telemetry',
            'unchanged model answers do not independently establish visual completion accuracy','no robot control or gate-policy changes']}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'modes':modes,'cases':{k:{x:y for x,y in v.items() if x!='requests'} for k,v in cases.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
