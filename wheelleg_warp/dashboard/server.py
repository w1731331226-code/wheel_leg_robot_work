#!/usr/bin/env python3
"""单一1024原生基线的只读状态、真实训练帧与本地轨迹。"""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import argparse,json,mimetypes,threading,time
from pathlib import Path
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
RUN=ROOT/'wheelleg_warp/results/formal_native_1024_20260921';DATA=HERE/'local_data'
STATE={};LOCK=threading.Lock()


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def status():
    p=read(RUN/'protocol.json',{});current=read(RUN/'status.json',{'status':'validating','round':0,'new_steps':0})
    selection=read(RUN/'selection.json',{'rounds':[],'stagnant_rounds':0})
    rate=current.get('round_steps',0)/current['train_seconds'] if current.get('train_seconds',0)>0 else None
    if rate is None and selection['rounds']:
        last=selection['rounds'][-1];rate=p['steps_per_round']/last['train_seconds']
    eta=None
    if current['status']=='training' and rate:eta=max(0,p['steps_per_round']-current.get('round_steps',0))/rate
    archives=[]
    for meta in sorted((DATA/'captures/native').glob('*/metadata.json'),key=lambda x:x.stat().st_mtime,reverse=True)[:20]:
        item=read(meta,{})
        if not (meta.parent/'animation_50.webp').exists():continue
        rel=meta.parent.relative_to(DATA).as_posix()
        archives.append({**item,'webp':'/media/'+rel+'/animation_50.webp','gif':'/media/'+rel+'/animation_50.gif',
            'trace':'/media/'+rel+'/trajectory.npz','metadata':'/media/'+rel+'/metadata.json'})
    validation=None
    if not p:
        test=ROOT/'wheelleg_warp/results/convergence_1024_rollout50_20260921'
        phase=read(test/'status.json',{})
        progress=read(test/phase.get('current','')/'progress.json',{})
        validation=dict(phase=phase.get('status'),current=phase.get('current'),policy_steps=progress.get('policy_steps'),target=2048000)
    live=read(DATA/'live/native.json')
    return dict(now=time.time(),current=current,validation=validation,requested_environment=read(DATA/'selected_environment.json',{'environment':0}).get('environment',0),protocol={k:p.get(k) for k in ('environments','n_steps','max_rounds','patience','steps_per_round','inherited_steps','bootstrap_summary')},
        selection={**selection,'rounds':[{k:r[k] for k in ('round','summary','policy_steps','train_seconds','total_seconds','updates')} for r in selection['rounds']]},rate=rate,round_remaining_training_seconds=eta,live=live,archives=archives,
        final_evaluation=read(RUN/'final_evaluation.json'),run_directory=str(RUN))


def collect():
    global STATE
    while True:
        try:
            value=status()
            with LOCK:STATE=value
        except Exception as exc:print('状态读取失败',repr(exc),flush=True)
        time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        if self.path!='/api/environment':return self.send_error(404)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=256:raise ValueError('length')
            index=json.loads(self.rfile.read(length))['environment']
            if type(index) is not int or not 0<=index<1024:raise ValueError('environment')
        except (ValueError,KeyError,TypeError):return self.send_error(400)
        path=DATA/'selected_environment.json';DATA.mkdir(parents=True,exist_ok=True)
        with LOCK:
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps({'environment':index}));temp.replace(path)
        return self.send_bytes(json.dumps({'environment':index}).encode(),'application/json')

    def do_GET(self):
        path=unquote(urlparse(self.path).path)
        if path=='/api/overview':
            paths=list(RUN.glob('round_*/live/overview.bin'))
            if not paths:
                for name in ('native_live_preflight_20260921','native_formal_probe_20260921'):paths.extend((ROOT/'wheelleg_warp/results'/name).glob('round_*/live/overview.bin'))
            if not paths:return self.send_error(404)
            target=max(paths,key=lambda p:p.stat().st_mtime)
            tag=str(target.stat().st_mtime_ns)
            if self.headers.get('If-None-Match')==tag:
                self.send_response(304);self.end_headers();return
            body=target.read_bytes();self.send_response(200);self.send_header('Content-Type','application/octet-stream');self.send_header('ETag',tag)
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
            return
        if path=='/api/best.zip':
            best=read(RUN/'selection.json',{}).get('best')
            if not best:return self.send_error(404)
            target=Path(best['path']+'.zip').resolve()
            if not target.is_relative_to(RUN.resolve()) or not target.is_file():return self.send_error(404)
            return self.send_bytes(target.read_bytes(),'application/zip')
        if path=='/api/frame':
            target=DATA/'live/native.frame.json'
            if not target.exists():return self.send_error(404)
            tag=str(target.stat().st_mtime_ns)
            if self.headers.get('If-None-Match')==tag:
                self.send_response(304);self.end_headers();return
            body=target.read_bytes();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('ETag',tag)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-cache');self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
            return
        if path=='/api/status':
            with LOCK:body=json.dumps(STATE,ensure_ascii=False,allow_nan=False).encode()
            return self.send_bytes(body,'application/json; charset=utf-8')
        if path=='/live/native.mjpg':
            self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control','no-store');self.end_headers();last=None
            try:
                while True:
                    image=DATA/'live/native.jpg'
                    if image.exists():
                        stamp=image.stat().st_mtime_ns
                        if stamp!=last:
                            content=image.read_bytes();last=stamp
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(content)).encode()+b'\r\n\r\n'+content+b'\r\n');self.wfile.flush()
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
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);a=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);STATE=status()
    threading.Thread(target=collect,daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
