"""Sequential, fail-closed rollout -> smoke -> training; no concurrent GPU stages."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from train_coarse_lora import write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/derived/bounded_observation_sft_20261123_v1'
RUN = ROOT/'runs/bounded_observation_lora_20261123_v1'
STATE = ROOT/'runs/bounded_training_pipeline_20261123_v1'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-focus", action="store_true")
    parser.add_argument("--paired-boundary", action="store_true")
    args = parser.parse_args()
    if args.paired_boundary:
        args.boundary_focus = True
    data = ROOT/'data/derived/bounded_observation_boundary_sft_20261123_v1' if args.boundary_focus else DATA
    run = ROOT/'runs/bounded_observation_boundary_20261123_v1' if args.boundary_focus else RUN
    state = ROOT/'runs/bounded_boundary_pipeline_20261123_v1' if args.boundary_focus else STATE
    replay_output = "runs/bounded_history_boundary_allval_20261123_v1"
    if args.paired_boundary:
        data = ROOT/"data/derived/bounded_observation_paired_sft_20261123_v1"
        run = ROOT/"runs/bounded_observation_paired_20261123_v1"
        state = ROOT/"runs/bounded_paired_pipeline_20261123_v1"
        replay_output = "runs/bounded_history_paired_allval_20261123_v1"
    state.mkdir(parents=True, exist_ok=False)
    write_json(state/'launch.json', {'pid': os.getpid(), 'started_unix': time.time(), 'data': str(data), 'run': str(run)})
    stages = [
        ('candidate_history_preparation', ['scripts/compare_bounded_history.py', '--output', str(data), '--prepare-training']),
        ('smoke_training', ['scripts/train_coarse_lora.py', '--config', 'configs/bounded_observation_lora_v1.json', '--data', str(data), '--output', str(run)+'_smoke', '--smoke']),
        ('training', ['scripts/train_coarse_lora.py', '--config', 'configs/bounded_observation_lora_v1.json', '--data', str(data), '--output', str(run)]),
    ]
    if args.boundary_focus:
        stages[0][1].append("--paired-boundary" if args.paired_boundary else "--boundary-focus")
        stages.append(("causal_validation", ["scripts/compare_bounded_history.py", "--output", replay_output,
                       "--recordings", "21", "--only-mode", "memory8", "--adapter", str(run/"best")]))
    try:
        for stage, arguments in stages:
            if stage == 'smoke_training':
                summary = json.loads((data/'summary.json').read_text())
                if summary['counts'] != {'train': 755, 'val': 105}:
                    raise ValueError('unexpected window counts')
                for split, expected in summary['counts'].items():
                    if len((data/(split+'.jsonl')).read_text().splitlines()) != expected:
                        raise ValueError('incomplete manifest')
            write_json(state/'status.json', {'status': 'running', 'stage': stage, 'updated_unix': time.time()})
            with (state/(stage+'.log')).open('w') as log:
                subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(state/'status.json', {'status': 'complete', 'updated_unix': time.time()})
    except Exception as exc:
        write_json(state/'status.json', {'status': 'failed', 'error': str(exc), 'updated_unix': time.time()})
        raise

if __name__ == '__main__':
    main()
