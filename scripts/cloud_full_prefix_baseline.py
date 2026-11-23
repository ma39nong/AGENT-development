"""Simple full-video and cumulative-prefix cloud baseline; streamcopy only, no textual memory."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time

if __package__:
    from .annotate_video_api import digest,save
    from .streaming_capability import cloud_call
else:
    from annotate_video_api import digest,save
    from streaming_capability import cloud_call


def prepare_prefix(source, nominal, target, fps):
    import cv2
    import numpy as np
    # Streamcopy may retain a few extra reordered packets. Leave a small guard,
    # then validate actual decoded duration and original endpoint pixels.
    subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',str(source),'-t',str(nominal-.2),
                    '-map','0:v:0','-c:v','copy','-an',str(target)],check=True)
    cap=cv2.VideoCapture(str(target));count=0;last=None
    while True:
        ok,frame=cap.read()
        if not ok:break
        count+=1;last=frame
    outfps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if count<=0 or count>int(nominal*fps) or abs(outfps-fps)>.01:raise ValueError('noncausal or invalid copied prefix')
    timestamps=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(target)]))['frames']
    last_source_frame=round(max(float(x['best_effort_timestamp_time']) for x in timestamps)*fps)
    if last_source_frame>=round(nominal*fps):raise ValueError('future source frame in copied prefix')
    original=cv2.VideoCapture(str(source));original.set(cv2.CAP_PROP_POS_FRAMES,last_source_frame);ok,frame=original.read();original.release()
    if not ok or not np.array_equal(frame,last):raise ValueError('copied prefix last frame differs from source')
    return last_source_frame+1, count


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=False)
    row=next(r for r in map(json.loads,(root/'data/derived/coarse_sft_20261123_v1/train.jsonl').read_text().splitlines()) if r['source_recording_id']=='b0f7213da83547b99aac45f6438b511e')
    source=Path(row['video']);fps=row['fps']
    if digest(source)!=row['video_sha256'] or digest(Path(row['annotation']))!=row['annotation_sha256']:raise ValueError('source changed')
    segments=json.loads(Path(row['annotation']).read_text())['segments']
    online=(root/'prompts/cloud_simple_five_phases_zh.txt').read_text()
    common=online[online.index('任务是：'):online.index('视频可能只是')]
    full='请完整观看视频，按实际动作列出各粗阶段的开始和结束秒数；缺失阶段如实说明，不要硬凑。\n'+common+'\n用简短表格列出阶段、开始秒数、结束秒数。时间从视频开头0秒计算，看不清的边界说明不确定。'
    jobs=[{'mode':'full','path':str(source),'actual_frames':row['frames'],'nominal_seconds':row['frames']/fps,'prompt':full}]
    for nominal in [*range(10,121,10),124]:
        target=args.output/f'prefix-{nominal:03d}.mp4';count,decoded=prepare_prefix(source,nominal,target,fps)
        jobs.append({'mode':'prefix','path':str(target),'actual_frames':count,'decoded_frames':decoded,'nominal_seconds':nominal,
                     'prompt':online+'\n这段视频从任务开始连续记录到当前，请判断最后可见时刻；视频当前长度为'+f'{count/fps:.3f}秒。'})
    for j in jobs:
        j['actual_end_seconds']=j['actual_frames']/fps;j['video_sha256']=digest(Path(j['path']))
        j['target_phase']=next(s['label'] for s in segments if s['start_frame']<=j['actual_frames']-1<s['end_frame'])
    save(args.output/'protocol.json',{'source_recording_id':row['source_recording_id'],'split':'train','camera':'cam0',
         'models':['qwen3.8-flash','qwen3.8-max'],'requests_max':len(jobs)*2,'thinking':False,'textual_memory':False,
         'reencoding':False,'prefix_validation':'all decoded frame PTS before nominal cutoff; last RGB exactly equals corresponding original source frame; truncated GOP can omit trailing reordered frames',
         'limitations':'one known train recording; full-video offline segmentation differs from causal current-phase task; service sampling unmeasured',
         'jobs':jobs})
    results=[]
    try:
        for model in ['qwen3.8-flash','qwen3.8-max']:
            for j in jobs:
                begin=time.perf_counter();blob=Path(j['path']).read_bytes()
                payload={'model':model,'enable_thinking':False,'stream':False,'max_tokens':1200 if j['mode']=='full' else 320,
                         'messages':[{'role':'user','content':[{'type':'video_url','video_url':{'url':'data:video/mp4;base64,'+base64.b64encode(blob).decode()}},
                         {'type':'text','text':j['prompt']}]}]}
                response={};error=None;api=None
                try:response,api=cloud_call(payload)
                except Exception as exc:error=type(exc).__name__+': '+str(exc)
                r={'model':model,**{k:v for k,v in j.items() if k!='prompt'},'response':response,'error':error,'api_seconds':api,'prepare_request_seconds':time.perf_counter()-begin}
                results.append(r)
                with (args.output/'predictions.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
                save(args.output/'status.json',{'status':'running','completed':len(results),'total':len(jobs)*2,'model':model,'mode':j['mode']})
                print(json.dumps({'completed':len(results),'total':len(jobs)*2,'model':model,'mode':j['mode'],'cutoff':j['actual_end_seconds'],'api_s':api,'error':error}),flush=True)
                if len(results)>=3 and all(r['error'] for r in results[-3:]):raise ValueError('three request failures; stop')
        save(args.output/'status.json',{'status':'complete','completed':len(results)})
    except Exception as exc:
        save(args.output/'status.json',{'status':'failed','completed':len(results),'error':str(exc)});raise

if __name__=='__main__':main()
