"""Pair two completed validation replays; retain regressions and boundary groups."""
import argparse
import json
from pathlib import Path
if __package__:
    from .summarize_bounded_replay import boundary_group
else:
    from summarize_bounded_replay import boundary_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    refs = {r['source_recording_id']: (json.loads(Path(r['annotation']).read_text())['segments'], r['fps'])
            for r in map(json.loads, (root/'data/derived/coarse_sft_20261123_v1/val.jsonl').read_text().splitlines())}
    mid = {(r['source_recording_id'], r['source_exclusive_end_frame']) for r in map(json.loads,
           (root/'data/derived/bounded_observation_sft_20261123_v1/val.jsonl').read_text().splitlines())}
    def read(directory):
        if json.loads((directory/'status.json').read_text())['status'] != 'complete':
            raise ValueError('cannot compare unfinished replay')
        rows = [json.loads(s) for s in (directory/'predictions.jsonl').read_text().splitlines()]
        result = {(r['source_recording_id'], r['source_exclusive_end_frame']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate probe or multiple modes')
        return result
    before, after = read(args.before), read(args.after)
    if before.keys() != after.keys():
        raise ValueError('probe sets differ')
    changed, groups = [], {}
    for key, a in before.items():
        b = after[key]
        for field in ('target_phase', 'target_index', 'source_first_frame', 'source_exclusive_end_frame', 'last_sampled_source_frame'):
            if a[field] != b[field]:
                raise ValueError('different reference or current source observation: '+field)
        segments, fps = refs[key[0]]
        group = 'midpoint' if key in mid else boundary_group(segments, key[1], fps)
        stat = groups.setdefault(group, dict(samples=0, before_correct=0, after_correct=0, improved=0, regressed=0))
        stat['samples'] += 1
        stat['before_correct'] += a['label_correct']
        stat['after_correct'] += b['label_correct']
        improve = not a['label_correct'] and b['label_correct']
        regress = a['label_correct'] and not b['label_correct']
        stat['improved'] += improve
        stat['regressed'] += regress
        if improve or regress:
            changed.append(dict(recording=key[0], cutoff_frame=key[1], group=group, target=b['target_phase'],
                                before=a['answer'].get('phase'), after=b['answer'].get('phase'), improved=improve, regressed=regress))
    report = dict(before=str(args.before.resolve()), after=str(args.after.resolve()), samples=len(before), groups=groups,
                  improved=sum(g['improved'] for g in groups.values()), regressed=sum(g['regressed'] for g in groups.values()), changed=changed,
                  scope='Same current observations and references; each model rolls its own candidate history; validation probes only')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'changed'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
