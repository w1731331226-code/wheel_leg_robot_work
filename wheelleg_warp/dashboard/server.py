#!/usr/bin/env python3
"""本地只读训练仪表盘、状态历史、ETA和检查点物理录像调度。"""
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'wheelleg_warp/results/formal_cpu_warp_fast_v1_20260921'
DATA = HERE/'local_data'
LOCK = threading.Lock()
STATE = {}
HISTORY = []


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def stamp(text):
    return datetime.fromisoformat(text).timestamp()


def estimate(total, done, elapsed, cycles):
    if done <= 0 or elapsed <= 0:
        return dict(rate=None, remaining_seconds=None, confidence='等待有效进度')
    rate = done/elapsed
    remaining = max(0,total-done)/rate
    return dict(rate=rate, remaining_seconds=remaining, range_seconds=[remaining*(.75 if cycles>=3 else .5),remaining*(1.5 if cycles>=3 else 2)],
                confidence='基于完整评估周期' if cycles>=3 else '早期粗估，评估/课程/资源竞争可能显著改变速度')


def status(now=None):
    now = now or time.time()
    protocol = read(RUN/'protocol.json')
    queue = read(RUN/'status.json', {})
    per_seed = protocol['config']['policy_steps']
    result = dict(now=now, queue=queue, backends={}, history=HISTORY[::max(1,len(HISTORY)//1440)], total_budget=2*len(protocol['seeds'])*per_seed,
                  run_directory=str(RUN), final_evaluation_cases=len(protocol['final_cases']))
    for backend in ('cpu','warp'):
        done=0; inherited_steps=0; all_points=[]; cycles=[]; current=None; starts=[]
        for index,seed in enumerate(protocol['seeds']):
            folder=RUN/f'{backend}_{seed}'
            config=folder/'run_config.json'
            if config.exists():starts.append(config.stat().st_mtime)
            start_steps=read(config,{}).get('start_policy_steps',0);inherited_steps+=start_steps
            completion=read(folder/'completed.json')
            progress=read(folder/'progress.json', {})
            steps=per_seed if completion and completion.get('passed') else progress.get('policy_steps',start_steps)
            done+=steps
            selected=read(folder/'selection.json',{})
            for row in selected.get('checkpoints',[]):
                point=dict(step=index*per_seed+row['steps'],seed=seed,success=100*row['success_count']/row['total'],
                           yaw=row['mean_yaw_score_deg'])
                all_points.append(point)
                check=folder/f"step_{row['steps']}.json"
                if check.exists():cycles.append((point['step'],check.stat().st_mtime))
            if current is None and not (completion and completion.get('passed')):
                current=dict(seed=seed,steps=steps,stage=progress.get('stage',max(c['stage'] for c in protocol['config']['curriculum'] if steps>=c['start'])),updated=progress.get('updated'),
                             error=read(folder/'failed.json'),directory=str(folder))
        current=current or dict(seed=protocol['seeds'][-1],steps=per_seed,stage=3,updated=None,error=None,directory=str(folder))
        total=per_seed*len(protocol['seeds']);started=min(starts) if starts else now
        speed_started=started
        if inherited_steps:
            anchor=RUN/f'{backend}_{protocol["seeds"][0]}'/f'step_{inherited_steps}.json'
            if anchor.exists():speed_started=max(started,anchor.stat().st_mtime)
        # 完整检查点评估周期包含选模开销；未有检查点时仅能用早期吞吐粗估。
        if cycles and cycles[-1][0]>inherited_steps:
            work, end = cycles[-1]
            speed=estimate(total-inherited_steps,work-inherited_steps,end-speed_started,len(cycles))
            if speed['rate']:
                speed['remaining_seconds']=max(0,total-done)/speed['rate']
                factor=(.75,1.5) if len(cycles)>=3 else (.5,2)
                speed['range_seconds']=[speed['remaining_seconds']*x for x in factor]
        else:
            speed=estimate(total-inherited_steps,done-inherited_steps,(stamp(current['updated']) if current['updated'] else now)-speed_started,0)
        pid=queue.get('pids',{}).get(backend)
        alive=bool(pid and Path(f'/proc/{pid}').exists())
        stale=bool(current['updated'] and now-stamp(current['updated'])>900)
        phase='completed' if done==total else 'failed' if current['error'] else 'running' if alive else 'stopped'
        pending_evaluation=any(Path(str(p.with_suffix(''))+'.pkl').exists() and not p.with_suffix('.json').exists()
                               for p in Path(current['directory']).glob('step_*.zip'))
        if phase=='running' and pending_evaluation:phase='evaluating'
        if phase in ('failed','stopped') or stale:
            speed.update(remaining_seconds=None,confidence='队列停止或进度过期，暂停预计时间')
        archive=[]
        for meta in sorted((DATA/'captures'/backend).glob('*/metadata.json'))[-60:]:
            value=read(meta)
            if value and ((meta.parent/'animation.gif').exists() or (meta.parent/'animation_50.gif').exists()):
                rel=meta.parent.relative_to(DATA).as_posix()
                archive.append(dict(**value, id=rel, live=read(meta.parent/'live.json',{}),
                                    gif='/media/'+rel+('/animation_50.gif' if (meta.parent/'animation_50.gif').exists() else '/animation.gif'),
                                    fps=50 if (meta.parent/'animation_50.gif').exists() else 25,
                                    webp='/media/'+rel+'/animation_50.webp' if (meta.parent/'animation_50.webp').exists() else None,trace='/media/'+rel+'/trajectory.npz',
                                    metadata='/media/'+rel+'/metadata.json', still='/media/'+rel+'/current.jpg'))
        result['backends'][backend]=dict(phase=phase,current=current,total_steps=done,inherited_steps=inherited_steps,budget=total,started=started,
                     elapsed_seconds=now-started,estimate=speed,stale=stale,selection=all_points,archives=archive,live=read(DATA/'live'/f'{backend}.json'))
    times=[b['estimate']['remaining_seconds'] for b in result['backends'].values()]
    smoke=read(RUN/'smoke_cpu_1609/completed.json',{})
    overhead=max(0,smoke.get('total_seconds',0)-smoke.get('train_seconds',0))
    # 两后端×三种子×两个检查点，每个64场景；smoke共同CPU四场景实测外推。
    evaluation_reserve=overhead/4*len(protocol['final_cases'])*len(protocol['seeds'])*4
    result['evaluation_reserve_seconds']=evaluation_reserve
    result['all_remaining_seconds']=max(times)+evaluation_reserve if all(t is not None for t in times) else None
    if queue.get('status')=='evaluating':result['all_remaining_seconds']=max(0,evaluation_reserve-(now-(RUN/'status.json').stat().st_mtime))
    if queue.get('status')=='completed':result['all_remaining_seconds']=0
    result['total_done']=sum(b['total_steps'] for b in result['backends'].values())
    return result


def collect():
    global STATE
    while True:
        try:
            current=status()
            row=dict(time=current['now'], **{b:current['backends'][b]['total_steps'] for b in ('cpu','warp')})
            HISTORY.append(row)
            with (DATA/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            with LOCK:STATE=current
        except Exception as exc:
            print('状态采集失败：',repr(exc),flush=True)
        time.sleep(10)



class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def do_GET(self):
        path=unquote(urlparse(self.path).path)
        if path=='/api/status':
            with LOCK:body=json.dumps(STATE,ensure_ascii=False,allow_nan=False).encode()
            return self.send_bytes(body,'application/json; charset=utf-8')
        if path.startswith('/live/') and path.endswith('.mjpg'):
            backend=path.split('/')[2].split('.')[0]
            if backend not in ('cpu','warp'):return self.send_error(404)
            self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control','no-store');self.end_headers()
            last=None;last_sent=0.
            try:
                while True:
                    files=[p for p in [DATA/'live'/f'{backend}.jpg'] if p.exists()]
                    if files:
                        image=max(files,key=lambda p:p.stat().st_mtime_ns)
                        key=(str(image),image.stat().st_mtime_ns)
                        if key!=last or time.monotonic()-last_sent>1:
                            last_sent=time.monotonic()
                            content=image.read_bytes();last=key
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(content)).encode()+b'\r\n\r\n'+content+b'\r\n')
                            self.wfile.flush()
                    time.sleep(1/60)
            except (BrokenPipeError,ConnectionResetError):pass
            return
        base=DATA if path.startswith('/media/') else HERE
        name=path[len('/media/'):] if base==DATA else ('index.html' if path=='/' else path.lstrip('/'))
        target=(base/name).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():return self.send_error(404)
        if base==HERE and target.name not in ('index.html','style.css','app.js'):return self.send_error(404)
        return self.send_bytes(target.read_bytes(),mimetypes.guess_type(str(target))[0] or 'application/octet-stream')

    def send_bytes(self,body,kind):
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-cache');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765)
    args=p.parse_args();DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'history.jsonl').exists():
        for line in (DATA/'history.jsonl').read_text().splitlines()[-1440:]:
            try:HISTORY.append(json.loads(line))
            except json.JSONDecodeError:pass
    STATE=status()
    threading.Thread(target=collect,daemon=True).start()
    # 渲染器为单独systemd进程，避免训练、HTTP服务和GL上下文互相阻塞。
    print(f'本地训练前端 http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
