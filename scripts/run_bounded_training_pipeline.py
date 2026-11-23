"""Sequential, fail-closed rollout -> smoke -> training; no concurrent GPU stages."""
import json
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
    STATE.mkdir(parents=True, exist_ok=False)
    write_json(STATE/'launch.json', {'pid': os.getpid(), 'started_unix': time.time(), 'data': str(DATA), 'run': str(RUN)})
    stages = [
        ('candidate_history_preparation', ['scripts/compare_bounded_history.py', '--output', str(DATA), '--prepare-training']),
        ('smoke_training', ['scripts/train_coarse_lora.py', '--config', 'configs/bounded_observation_lora_v1.json', '--data', str(DATA), '--output', str(RUN)+'_smoke', '--smoke']),
        ('training', ['scripts/train_coarse_lora.py', '--config', 'configs/bounded_observation_lora_v1.json', '--data', str(DATA), '--output', str(RUN)]),
    ]
    try:
        for stage, arguments in stages:
            if stage == 'smoke_training':
                summary = json.loads((DATA/'summary.json').read_text())
                if summary['counts'] != {'train': 755, 'val': 105}:
                    raise ValueError('unexpected window counts')
                for split, expected in summary['counts'].items():
                    if len((DATA/(split+'.jsonl')).read_text().splitlines()) != expected:
                        raise ValueError('incomplete manifest')
            write_json(STATE/'status.json', {'status': 'running', 'stage': stage, 'updated_unix': time.time()})
            with (STATE/(stage+'.log')).open('w') as log:
                subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(STATE/'status.json', {'status': 'complete', 'updated_unix': time.time()})
    except Exception as exc:
        write_json(STATE/'status.json', {'status': 'failed', 'error': str(exc), 'updated_unix': time.time()})
        raise

if __name__ == '__main__':
    main()
