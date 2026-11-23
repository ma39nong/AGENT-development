"""Bounded paid API replay: continuous source cadence, causal MP4 windows, no still-image input."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

if __package__:
    from .annotate_video_api import digest, save
    from .streaming_capability import cloud_call
    from .compare_bounded_history import parse_observation
else:
    from annotate_video_api import digest, save
    from streaming_capability import cloud_call
    from compare_bounded_history import parse_observation


def available_history(records, cutoff, maximum=2):
    return [{"observed_until":r['cutoff_seconds'],"candidate_phase":r['answer']['phase'],
             "uncertain":r['answer']['uncertain'],"status":"unverified"}
            for r in reversed(records) if r['valid'] and r['answer']['phase']!='unknown'
            and r['received_source_seconds']<=cutoff ][:maximum]


def summarize(rows):
    labels=('pickup','harvest','adjust','return')
    matrix={k:dict.fromkeys((*labels,'unknown','invalid'),0) for k in labels}
    for r in rows: matrix[r['target_phase']][r['answer'].get('phase','invalid') if r['valid'] else 'invalid']+=1
    return {'samples':len(rows),'valid':sum(r['valid'] for r in rows),
            'correct':sum(r['correct'] for r in rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),
            'per_class_recall':{k:v[k]/sum(v.values()) if sum(v.values()) else None for k,v in matrix.items()},
            'confusion':matrix,'false_return_count':sum(v['return'] for k,v in matrix.items() if k!='return'),
            'median_api_s':__import__('statistics').median(r['api_seconds'] for r in rows if r['api_seconds'] is not None) if any(r['api_seconds'] is not None for r in rows) else None,
            'max_request_cycle_s':max(r['request_cycle_seconds'] for r in rows),
            'median_request_cycle_s':__import__('statistics').median(r['request_cycle_seconds'] for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['qwen3.8-flash','qwen3.8-max'],default='qwen3.8-flash')
    p.add_argument('--prompt-file',type=Path)
    p.add_argument('--fixed-context-run',type=Path,help='Prompt-only comparison using exact prior unverified context at each cutoff')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    source=next(r for r in map(json.loads,(root/'data/derived/coarse_sft_20261123_v1/train.jsonl').read_text().splitlines())
                if r['source_recording_id']=='b0f7213da83547b99aac45f6438b511e')
    if digest(Path(source['video']))!=source['video_sha256'] or digest(Path(source['annotation']))!=source['annotation_sha256']:
        raise ValueError('source identity changed')
    seg=json.loads(Path(source['annotation']).read_text())['segments']
    fps=source['fps'];ends=list(range(round(2*fps),source['frames'],round(2*fps)))+[source['frames']]
    template=(root/'prompts/bounded_history_probe_zh.txt').read_text().replace('历史图片带有各自的源时间，不能当作当前画面。','历史候选只含以前的文字判断，不是当前画面。').replace('和历史原始图','')
    if args.prompt_file:
        template=args.prompt_file.read_text()
    fixed={r['cutoff_frame']:r for r in map(json.loads,(args.fixed_context_run/'predictions.jsonl').read_text().splitlines())} if args.fixed_context_run else None
    if fixed is not None and set(fixed)!=set(ends):raise ValueError('fixed replay cutoff mismatch')
    save(args.output/'protocol.json',{'model':args.model,'source_recording_id':source['source_recording_id'],'split':'train',
        'source_video_sha256':source['video_sha256'],'camera':'cam0','fps':fps,'window_seconds':8,'cadence_seconds':2,
        'fixed_context_run':str(args.fixed_context_run) if args.fixed_context_run else None,'prompt_sha256':__import__('hashlib').sha256(template.encode()).hexdigest(),
        'requests_maximum':len(ends),'enable_thinking':False,'prompt':template,'memory':'up to two actually returned unverified own candidates',
        'input':'native video_url MP4; no still-image inputs, no resizing, original frame clock; h264 re-encoding',
        'server_sampling':'provider controlled, not measured','no_retries':True,'control':False,'counting':False,
        'limitations':'Offline fixed-cadence replay. No real-time scheduler/backpressure; source-clock availability estimates assume requests could launch at cadence; no robot commands.'})
    import cv2
    results=[]
    try:
        for cutoff in ends:
            start=max(0,cutoff-round(8*fps));begin=time.perf_counter()
            clip=args.output/f'window-{cutoff:06d}.mp4'
            subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-threads','1','-i',source['video'],
                '-vf',f'trim=start_frame={start}:end_frame={cutoff},setpts=N/({fps}*TB)','-an','-r',str(fps),'-fps_mode','cfr',
                '-c:v','libx264','-threads','1','-preset','ultrafast','-crf','18',str(clip)],check=True)
            cap=cv2.VideoCapture(str(clip));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));actual_fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
            if count!=cutoff-start or abs(actual_fps-fps)>.01:raise ValueError('frame clock mismatch')
            if fixed is not None and digest(clip)!=fixed[cutoff]['clip_sha256']:raise ValueError('video differs from fixed-context baseline')
            history=fixed[cutoff]['history'] if fixed is not None else available_history(results,cutoff/fps)
            context={'window_start_seconds':start/fps,'observation_cutoff_seconds':cutoff/fps,'history_candidates':history}
            payload={'model':args.model,'enable_thinking':False,'stream':False,'max_tokens':160,
                     'messages':[{'role':'user','content':[{'type':'video_url','video_url':{'url':'data:video/mp4;base64,'+base64.b64encode(clip.read_bytes()).decode()}},
                                  {'type':'text','text':template+'\n本次观测：'+json.dumps(context,ensure_ascii=False)}]}]}
            response={};answer={};api_seconds=None;error=None
            try:
                response,api_seconds=cloud_call(payload)
                choice=response['choices'][0]
                if choice['finish_reason']!='stop':raise ValueError('incomplete output: '+str(choice['finish_reason']))
                answer=parse_observation(choice['message']['content'])
            except Exception as exc:
                error=type(exc).__name__+': '+str(exc)
            elapsed=time.perf_counter()-begin
            target=next(s['label'] for s in seg if s['start_frame']<=cutoff-1<s['end_frame'])
            result={'cutoff_frame':cutoff,'cutoff_seconds':cutoff/fps,'source_first_frame':start,'source_last_frame':cutoff-1,
                    'target_phase':target,'answer':answer,'valid':error is None,'correct':error is None and answer['phase']==target,
                    'error':error,'history':history,'api_seconds':api_seconds,'request_cycle_seconds':elapsed,
                    'received_source_seconds':cutoff/fps+elapsed,'clip_sha256':digest(clip),'raw_response':response}
            results.append(result)
            with (args.output/'predictions.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
            save(args.output/'summary.json',summarize(results))
            save(args.output/'status.json',{'status':'running','completed':len(results),'total':len(ends)})
            print(json.dumps({'completed':len(results),'total':len(ends),'phase':answer.get('phase'),'correct':result['correct'],'error':error,'cycle_s':elapsed}),flush=True)
            if len(results)>=3 and all(not r['raw_response'].get('choices') for r in results[-3:]):raise ValueError('three consecutive transport/service failures; stop paid requests')
        save(args.output/'status.json',{'status':'complete','completed':len(results)})
    except Exception as exc:
        save(args.output/'status.json',{'status':'failed','completed':len(results),'error':str(exc)})
        raise


if __name__=='__main__':main()
