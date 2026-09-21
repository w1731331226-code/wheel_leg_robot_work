"""单基线API、任意环境选择和文件边界；无需GPU。"""
from pathlib import Path
import json,tempfile,threading,urllib.request,urllib.error
import server
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);server.RUN=root/'run';server.RUN.mkdir();server.DATA=root/'data';server.DATA.mkdir()
    state=server.status();assert state['current']['status']=='validating' and state['archives']==[]
    private=root/'private';private.write_text('secret');(server.DATA/'escape').symlink_to(private)
    (server.DATA/'clip.gif').write_bytes(b'GIF89a');server.STATE={'ok':True}
    http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
    threading.Thread(target=http.serve_forever,daemon=True).start();base='http://127.0.0.1:'+str(http.server_port)
    assert urllib.request.urlopen(base+'/api/status').read()==b'{"ok": true}'
    assert urllib.request.urlopen(base+'/media/clip.gif').read()==b'GIF89a'
    for path in ('/media/%2e%2e/private','/media/escape','/server.py','/api/best.zip'):
        try:urllib.request.urlopen(base+path)
        except urllib.error.HTTPError as e:assert e.code==404
        else:raise AssertionError(path)
    for value in (0,1023,-1,1024,True,'1'):
        request=urllib.request.Request(base+'/api/environment',data=json.dumps({'environment':value}).encode(),headers={'Content-Type':'application/json'},method='POST')
        if type(value) is int and 0<=value<1024:
            assert json.loads(urllib.request.urlopen(request).read())['environment']==value
        else:
            try:urllib.request.urlopen(request)
            except urllib.error.HTTPError as e:assert e.code==400
            else:raise AssertionError(value)
    assert json.loads((server.DATA/'selected_environment.json').read_text())['environment']==1023
    http.shutdown();http.server_close()
print('PASS: single-baseline status, 0/1023 selection, invalid input and file boundaries')
