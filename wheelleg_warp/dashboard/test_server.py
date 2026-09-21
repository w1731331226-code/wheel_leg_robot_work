"""ETA边界及本地HTTP路径限制；不依赖训练终评数据。"""
from pathlib import Path
import tempfile
import threading
import urllib.request
import urllib.error
import server

assert server.estimate(6000000,0,100,0)['remaining_seconds'] is None
x=server.estimate(6000000,20000,200,1)
assert x['rate']==100 and x['remaining_seconds']==59800
assert server.estimate(20000,20000,200,3)['remaining_seconds']==0
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);private=root/'private';private.write_text('secret')
    safe=root/'safe';safe.mkdir();(safe/'clip.gif').write_bytes(b'GIF89a')
    (safe/'escape').symlink_to(private)
    server.DATA=safe;server.STATE={'ok':True}
    http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
    threading.Thread(target=http.serve_forever,daemon=True).start()
    base='http://127.0.0.1:'+str(http.server_port)
    assert urllib.request.urlopen(base+'/api/status').read()==b'{"ok": true}'
    assert urllib.request.urlopen(base+'/media/clip.gif').read()==b'GIF89a'
    for path in ('/media/%2e%2e/private','/media/escape','/server.py'):
        try:urllib.request.urlopen(base+path)
        except urllib.error.HTTPError as e:assert e.code==404
        else:raise AssertionError('禁止任意本地文件读取')
    http.shutdown();http.server_close()
print('PASS：ETA空进度/完成边界，API，越界/符号链接/源文件访问拒绝')

# 恢复检查点的旧步数和一次性补评估不能算成新速度，也不能拉低稳态预测。
import json,os
from datetime import datetime,timezone
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);server.RUN=root/'run';server.DATA=root/'data';server.RUN.mkdir()
    protocol={'seeds':[1609,1610,1611],'config':{'policy_steps':2000000,'curriculum':[{'start':0,'stage':1},{'start':20000,'stage':2}]},'final_cases':[]}
    (server.RUN/'protocol.json').write_text(json.dumps(protocol))
    (server.RUN/'status.json').write_text(json.dumps({'status':'training','pids':{'cpu':os.getpid(),'warp':os.getpid()}}))
    t=1700000000
    for backend in ('cpu','warp'):
        folder=server.RUN/f'{backend}_1609';folder.mkdir()
        config=folder/'run_config.json';config.write_text(json.dumps({'start_policy_steps':60000}));os.utime(config,(t,t))
        (folder/'progress.json').write_text(json.dumps({'policy_steps':62000,'stage':2,'updated':datetime.fromtimestamp(t+140,timezone.utc).isoformat()}))
        (folder/'selection.json').write_text(json.dumps({'checkpoints':[{'steps':60000,'success_count':1,'total':1,'mean_yaw_score_deg':0}]}))
        anchor=folder/'step_60000.json';anchor.write_text('{}');os.utime(anchor,(t+120,t+120))
    state=server.status(now=t+141)
    assert state['backends']['cpu']['estimate']['rate']==100
    assert state['backends']['cpu']['total_steps']==62000
    assert state['backends']['cpu']['estimate']['remaining_seconds']==59380
print('PASS：续训继承步数与启动补评估从吞吐估算中剔除')
