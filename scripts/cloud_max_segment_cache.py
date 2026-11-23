"""8 fixed-case calls: full prefix vs lossless segment packaging, same visual frames."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

if __package__:
    from .annotate_video_api import ENDPOINT,KEY_PATH,digest,save
    from .handoff_gate import parse_decision
    from .cloud_max_transport_latency import timed_post
else:
    from annotate_video_api import ENDPOINT,KEY_PATH,digest,save
    from handoff_gate import parse_decision
    from cloud_max_transport_latency import timed_post


def decoded_hashes(path):
    import cv2
    cap=cv2.VideoCapture(str(path));hashes=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
    cap.release();return hashes


def main():
    import requests
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    source=root/'runs/cloud_max_handoff_acceptance_20261123_v1';proto=json.loads((source/'protocol.json').read_text())
    records=list(map(json.loads,(source/'predictions.jsonl').read_text().splitlines()))
    cases=[dict(next(c for c in proto['cases'] if c['id']==identity)) for identity in ('B4--1','B4-+1')]
    first_hash=None;first_frames=None;first_path=None
    for n,c in enumerate(cases):
        if digest(Path(c['path']))!=c['sha256']:raise ValueError('source changed')
        pattern=str(args.output/f'case{n}-%02d.mp4')
        subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',c['path'],'-map','0:v:0','-c:v','copy',
          '-f','segment','-segment_times','60','-reset_timestamps','1',pattern],check=True)
        parts=sorted(args.output.glob(f'case{n}-*.mp4'))
        if len(parts)!=2:raise ValueError('expected exactly two keyframe-aligned segments')
        original=decoded_hashes(Path(c['path']));parts_frames=[decoded_hashes(p) for p in parts]
        if original!=sum(parts_frames,[]):raise ValueError('decoded frame sequence changed; no API calls')
        checksum=digest(parts[0])
        if first_hash is None:
            first_hash=checksum;first_frames=parts_frames[0];first_path=parts[0]
        else:
            if first_frames!=parts_frames[0]:raise ValueError('historical visual frames changed')
            # Container metadata can differ despite identical frames. Reuse the
            # already verified immutable history file, without transcoding.
            parts[0]=first_path
        c['segments']=[str(p) for p in parts];c['split_seconds']=len(parts_frames[0])/30
        c['decoded_frame_sequence_sha256']=hashlib.sha256(''.join(original).encode()).hexdigest();c['frame_equality_verified']=True
    save(args.output/'protocol.json',{'model':'qwen3.8-max','cases':cases,'max_requests':8,'order_per_case':['full','segmented','segmented','full'],
         'prompt_sha256':proto['prompt_sha256'],'same_frames':True,'historical_segment_sha256':first_hash,'transport':'same persistent requests.Session',
         'only_change':'video packaging with necessary per-video source time ranges; no frame omission or reencoding',
         'cache_caveat':'cache hits measured, never guaranteed; no remote object store; Base64 still uploads all bytes',
         'no_retries':True,'robot_control':False})
    headers={'Authorization':'Bearer '+KEY_PATH.read_text().strip(),'Content-Type':'application/json','User-Agent':'qwen-fixed-acceptance/1.0'}
    session=requests.Session();results=[]
    try:
        for c in cases:
            request=next(r['request'] for r in records if r['case_id']==c['id'])
            for mode in ('full','segmented','segmented','full'):
                paths=[c['path']] if mode=='full' else c['segments']
                ranges=[[0,c['actual_end_seconds']]] if mode=='full' else [[0,c['split_seconds']],[c['split_seconds'],c['actual_end_seconds']]]
                content=[]
                for path,source_range in zip(paths,ranges):
                    content.append({'type':'text','text':'连续视频片段，原任务时间范围：'+json.dumps(source_range)+'秒。按视频顺序连续观看，最后一段末尾是当前时刻。'})
                    content.append({'type':'video_url','video_url':{'url':'data:video/mp4;base64,'+base64.b64encode(Path(path).read_bytes()).decode()}})
                content.append({'type':'text','text':json.dumps({**request,'observation_cutoff_seconds':c['actual_end_seconds']},ensure_ascii=False)})
                payload={'model':proto['model'],'enable_thinking':False,'stream':False,'max_tokens':96,'response_format':{'type':'json_object'},
                         'messages':[{'role':'system','content':proto['prompt']},{'role':'user','content':content}]}
                encoded=json.dumps(payload,ensure_ascii=False).encode();resp={};timing={};answer=None;error=None
                try:
                    resp,timing=timed_post(session,encoded,headers);choice=resp['choices'][0]
                    if choice['finish_reason']!='stop':raise ValueError('incomplete output')
                    answer=parse_decision(choice['message']['content'])
                except Exception as exc:error=type(exc).__name__
                session.cookies.clear()
                result={'case':c['id'],'mode':mode,'timing':timing,'answer':answer,'valid':answer is not None,'error':error,
                        'payload_bytes':len(encoded),'response':resp}
                results.append(result)
                with (args.output/'predictions.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                save(args.output/'status.json',{'status':'running','completed':len(results),'total':8})
                print(json.dumps({'completed':len(results),'case':c['id'],'mode':mode,'seconds':timing.get('complete_s'),'error':error}),flush=True)
                if len(results)>=3 and all(r['error'] for r in results[-3:]):raise ValueError('three failures; stop')
        save(args.output/'status.json',{'status':'complete','completed':len(results)})
    except Exception as exc:
        save(args.output/'status.json',{'status':'failed','completed':len(results),'error':type(exc).__name__});raise
    finally:session.close()

if __name__=='__main__':main()
