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
