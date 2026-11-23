"""Report fixed structured-output/gating acceptance without fabricating completion truth."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    state=json.loads((args.run/'status.json').read_text())
    rows=list(map(json.loads,(args.run/'predictions.jsonl').read_text().splitlines()))
    groups={}
    for group in sorted({r['group'] for r in rows}):
        rs=[r for r in rows if r['group']==group]
        groups[group]={'samples':len(rs),'strict_valid':sum(r['valid'] for r in rs),'truncated':sum(any(c.get('finish_reason')=='length' for c in r['raw_response'].get('choices',[])) for r in rs),
            'handoff_proposals':sum(r['answer'] is not None and r['answer']['decision']=='handoff' for r in rs),
            'scene_changed_reported':sum(r['answer'] is not None and r['answer']['decision']=='scene_changed' for r in rs),
            'gate_rejections':dict(Counter(r['offline_gate']['reason'] for r in rs))}
    repeated={}
    for identity in sorted({r['case_id'] for r in rows}):
        rs=[r for r in rows if r['case_id']==identity]
        repeated[identity]={'completion_consistent':len({r['answer']['completion'] if r['answer'] else None for r in rs})==1,
                            'decision_and_next_consistent':len({(r['answer']['decision'],r['answer']['next_skill']) if r['answer'] else None for r in rs})==1}
    report={'run_status':state,'requests':len(rows),'model':'qwen3.8-max','strict_valid':sum(r['valid'] for r in rows),'groups':groups,
            'median_api_s':statistics.median(r['api_seconds'] for r in rows if r['api_seconds'] is not None),
            'max_api_s':max(r['api_seconds'] for r in rows if r['api_seconds'] is not None),
            'over_screening_age_2s':sum(r['observation_age_seconds']>2 for r in rows),
            'completion_consistent_cases':sum(x['completion_consistent'] for x in repeated.values()),
            'decision_consistent_cases':sum(x['decision_and_next_consistent'] for x in repeated.values()),'repeat_cases':repeated,
            'release_count_in_fault_fixtures':sum(r['offline_gate']['allow'] for r in rows),
            'visual_completion_accuracy':None,
            'limitations':['coarse onset times are not skill-completion truth; before cases remain visual review pending',
                'executor fields/safe points/transition capability are simulated fixtures, not real acknowledgements',
                'all-blocked fixtures are not end-to-end success and do not demonstrate controller hold or pause',
                'two-second gate age is a screening parameter, not measured robot deadline',
                'after cases provide known old executor context as a stale-state challenge; not the new skill as an answer',
                'single previously inspected recording; repeated calls are not independent scenes; no counts or robot control']}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
