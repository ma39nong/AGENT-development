"""Summarize fixed validation midpoint/boundary probes without nearest-boundary ambiguity."""
import argparse
import json
from pathlib import Path
import statistics


def boundary_group(segments, cutoff, fps):
    # Two nearby handoffs can be less than two seconds apart. Match generated
    # +/-1s probes exactly rather than assigning to the nearest handoff.
    candidates = [(i, cutoff-s['end_frame']) for i, s in enumerate(segments[:-1], 1)
                  if abs(cutoff-s['end_frame']) == round(fps)]
    if len(candidates) != 1:
        return 'ambiguous_boundary_probe'
    index, offset = candidates[0]
    return f'B{index}_{"after" if offset > 0 else "before"}_1s'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root/'data/derived/coarse_sft_20261123_v1/val.jsonl'
    refs = {r['source_recording_id']: (json.loads(Path(r['annotation']).read_text())['segments'], r['fps'])
            for r in map(json.loads, source.read_text().splitlines())}
    mid = {(r['source_recording_id'], r['source_exclusive_end_frame']) for r in map(json.loads,
           (root/'data/derived/bounded_observation_sft_20261123_v1/val.jsonl').read_text().splitlines())}
    rows = [json.loads(s) for s in (args.run/'predictions.jsonl').read_text().splitlines()]
    if len({(r['source_recording_id'], r['source_exclusive_end_frame']) for r in rows}) != len(rows):
        raise ValueError('duplicate probes or multiple modes')
    if json.loads((args.run/'status.json').read_text())['status'] != 'complete':
        raise ValueError('replay incomplete')
    labels = ('pickup', 'harvest', 'adjust', 'return')
    matrix = {k: dict.fromkeys((*labels, 'unknown', 'invalid'), 0) for k in labels}
    groups, errors = {}, []
    for r in rows:
        matrix[r['target_phase']][r['answer'].get('phase', 'invalid') if r['valid'] else 'invalid'] += 1
        identity, cutoff = r['source_recording_id'], r['source_exclusive_end_frame']
        segments, fps = refs[identity]
        group = 'midpoint' if (identity, cutoff) in mid else boundary_group(segments, cutoff, fps)
        stat = groups.setdefault(group, {'samples': 0, 'correct': 0})
        stat['samples'] += 1
        stat['correct'] += r['label_correct']
        if not r['label_correct']:
            errors.append({'recording': identity, 'cutoff_frame': cutoff, 'group': group,
                           'target': r['target_phase'], 'prediction': r['answer'].get('phase')})
    recalls = {k: v[k]/sum(v.values()) if sum(v.values()) else None for k, v in matrix.items()}
    report = {'run': str(args.run.resolve()), 'samples': len(rows), 'valid': sum(r['valid'] for r in rows),
              'correct': sum(r['label_correct'] for r in rows), 'accuracy': sum(r['label_correct'] for r in rows)/len(rows),
              'per_class_recall': recalls, 'macro_recall': statistics.mean(v for v in recalls.values() if v is not None),
              'confusion': matrix, 'false_return_count': sum(v['return'] for k, v in matrix.items() if k != 'return'),
              'groups': groups, 'errors': errors, 'median_prepare_generate_s': statistics.median(r['prepare_generate_s'] for r in rows),
              'max_prepare_generate_s': max(r['prepare_generate_s'] for r in rows),
              'full_five_stage_index_correct': sum(r['index_correct'] for r in rows),
              'scope': 'Validation only, self-generated history and annotation-selected probes; not dense online boundary timing',
              'timing_excludes': ['physical trim', 'storage', 'queue', 'transport', 'robot control']}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('errors', 'confusion')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
