"""12-call fixed Max transport-only ABBA test, no prompt/model/input sweep."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

if __package__:
    from .annotate_video_api import ENDPOINT,KEY_PATH,digest,save
    from .handoff_gate import parse_decision
else:
    from annotate_video_api import ENDPOINT,KEY_PATH,digest,save
    from handoff_gate import parse_decision


def timed_post(session,payload_bytes,headers):
    begin=time.perf_counter();response=session.post(ENDPOINT,data=payload_bytes,headers=headers,stream=True,timeout=(15,90),allow_redirects=False)
    header_s=time.perf_counter()-begin
    connection=getattr(response.raw,'connection',None)
    sock=getattr(connection,'sock',None)
    fingerprint=None
    if sock is not None:
        try:fingerprint=hashlib.sha256(str((sock.fileno(),sock.getsockname()[1],sock.getpeername()[1])).encode()).hexdigest()[:16]
        except OSError:pass
    try:
        if response.status_code!=200:raise ValueError('HTTP '+str(response.status_code))
        blob=bytearray();first_body=None
        for chunk in response.iter_content(chunk_size=8192):
            if first_body is None:first_body=time.perf_counter()-begin
            blob.extend(chunk)
            if len(blob)>2*1024*1024:raise ValueError('response exceeds cap')
        elapsed=time.perf_counter()-begin
        return json.loads(blob),{'complete_s':elapsed,'headers_s':header_s,'first_body_chunk_s':first_body,'socket_fingerprint':fingerprint,'response_connection_header':response.headers.get('Connection')}
    finally:response.close()


def main():
    import requests
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    original=root/'runs/cloud_max_handoff_acceptance_20261123_v1';proto=json.loads((original/'protocol.json').read_text())
    prior=list(map(json.loads,(original/'predictions.jsonl').read_text().splitlines()))
    ids=('B1--1','B2--1','B4-+1');cases=[next(c for c in proto['cases'] if c['id']==key) for key in ids]
    key=KEY_PATH.read_text().strip()
    if not key:raise ValueError('missing key')
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','User-Agent':'qwen-fixed-acceptance/1.0'}
    save(args.output/'protocol.json',{'source_run':str(original),'model':proto['model'],'cases':ids,'order_per_case':['fresh','session','session','fresh'],
        'requests_maximum':12,'transport_backend':'requests.Session with existing environment proxy settings and TLS verification',
        'same_video_prompt_payload_per_case':True,'no_retries':True,'gate_unchanged':True,
        'cache_control':'record cached tokens; ABBA order; cache/time variance not guaranteed controlled',
        'limitations':'3 fixed cases, not stable p95/p99 or service-side queue/prefill decomposition'})
    shared=requests.Session();shared.cookies.clear();results=[]
    try:
        for c in cases:
            if digest(Path(c['path']))!=c['sha256']:raise ValueError('source clip changed')
            request=next(r['request'] for r in prior if r['case_id']==c['id'])
            payload={'model':proto['model'],'enable_thinking':False,'stream':False,'max_tokens':96,'response_format':{'type':'json_object'},
              'messages':[{'role':'system','content':proto['prompt']},{'role':'user','content':[
                {'type':'video_url','video_url':{'url':'data:video/mp4;base64,'+base64.b64encode(Path(c['path']).read_bytes()).decode()}},
                {'type':'text','text':json.dumps({**request,'observation_cutoff_seconds':c['actual_end_seconds']},ensure_ascii=False)}]}]}
            encoded=json.dumps(payload,ensure_ascii=False).encode()
            for number,mode in enumerate(('fresh','session','session','fresh'),1):
                session=requests.Session() if mode=='fresh' else shared
                response={};timing={};answer=None;error=None;started=time.perf_counter()
                try:
                    response,timing=timed_post(session,encoded,headers);choice=response['choices'][0]
                    if choice['finish_reason']!='stop':raise ValueError('incomplete output')
                    answer=parse_decision(choice['message']['content'])
                except Exception as exc:
                    # Avoid requests exception strings, which can contain proxy URLs.
                    error=type(exc).__name__
                finally:
                    session.cookies.clear()
                    if mode=='fresh':session.close()
                result={'case':c['id'],'position':number,'mode':mode,'payload_sha256':hashlib.sha256(encoded).hexdigest(),
                        'timing':timing,'wall_s':time.perf_counter()-started,'valid':answer is not None,'answer':answer,'error':error,'response':response}
                results.append(result)
                with (args.output/'predictions.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                save(args.output/'status.json',{'status':'running','completed':len(results),'total':12})
                print(json.dumps({'completed':len(results),'case':c['id'],'mode':mode,'seconds':timing.get('complete_s'),'valid':answer is not None,'error':error}),flush=True)
                if len(results)>=3 and all(r['error'] for r in results[-3:]):raise ValueError('three failures; stop without fallback')
        save(args.output/'status.json',{'status':'complete','completed':len(results)})
    except Exception as exc:
        save(args.output/'status.json',{'status':'failed','completed':len(results),'error':type(exc).__name__});raise
    finally:shared.close()

if __name__=='__main__':main()
