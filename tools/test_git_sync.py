"""标准库集成检查：自动提交/推送、钩子拒绝、暂存区保护、远程分叉保护。"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from unittest.mock import patch
import git_sync as sync


def run(*args, ok=True):
    p = subprocess.run(args, cwd=sync.ROOT, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, p.stderr + p.stdout
    else:
        assert p.returncode != 0
    return p


def main():
    original = sync.ROOT
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sync.ROOT = root / 'work'
        sync.ROOT.mkdir()
        run('git', 'init', '--bare', str(root / 'remote.git'))
        run('git', 'init', '-b', 'main')
        run('git', 'config', 'user.name', '同步测试')
        run('git', 'config', 'user.email', 'test@example.invalid')
        (sync.ROOT / '.githooks').mkdir()
        shutil.copy2(original / '.githooks/commit-msg', sync.ROOT / '.githooks/commit-msg')
        run('git', 'config', 'core.hooksPath', '.githooks')
        (sync.ROOT / sync.MEMORY).write_text('# 测试记忆\n')
        run('git', 'add', '.')
        run('git', 'commit', '-m', '初始化：同步测试')
        run('git', 'remote', 'add', 'origin', str(root / 'remote.git'))
        run('git', 'push', '-u', 'origin', 'main')
        (sync.ROOT / 'demo.txt').write_text('变更\n')
        sync.sync(sync.changes())
        head = sync.git('rev-parse', 'HEAD').strip()
        assert sync.git('ls-remote', 'origin', 'refs/heads/main').split()[0] == head
        assert not sync.git('status', '--porcelain')
        (sync.ROOT / 'demo.txt').write_text('下一项\n')
        run('git', 'add', 'demo.txt')
        run('git', 'commit', '-m', 'English only', ok=False)
        run('git', 'commit', '-m', '中文但未更新记忆', ok=False)
        staged = sync.git('diff', '--cached')
        sync.sync(sync.changes())
        assert sync.git('diff', '--cached') == staged
        assert sync.git('rev-parse', 'HEAD').strip() == head
        run('git', 'reset', '--hard', 'HEAD')
        heads=[]
        for number in range(2):
            (sync.ROOT/'pending.txt').write_text(str(number))
            with (sync.ROOT/sync.MEMORY).open('a') as stream:stream.write(f'\n待传{number}\n')
            run('git','add','.');run('git','commit','-m',f'待传测试{number}');heads.append(sync.git('rev-parse','HEAD').strip())
        sync.sync(sync.changes());assert sync.git('ls-remote','origin','refs/heads/main').split()[0]==heads[0]
        budget=sync.PUSH_BYTES;sync.PUSH_BYTES=1
        try:
            try:sync.sync(sync.changes())
            except RuntimeError:pass
            else:raise AssertionError('Oversized pending commit pushed')
            assert sync.git('ls-remote','origin','refs/heads/main').split()[0]==heads[0]
        finally:sync.PUSH_BYTES=budget
        sync.sync(sync.changes());assert sync.git('ls-remote','origin','refs/heads/main').split()[0]==heads[1]
        budget=sync.SNAPSHOT_BYTES;sync.SNAPSHOT_BYTES=80
        try:
            for number in range(3):(sync.ROOT/f'batch{number}.bin').write_bytes(os.urandom(64))
            sync.sync(sync.changes())
            remaining=sync.changes();assert sum(name.startswith('batch') for name in remaining)==2
            sync.sync(remaining);sync.sync(sync.changes())
            assert not sync.git('status','--porcelain')
        finally:sync.SNAPSHOT_BYTES=budget
        run('git', 'clone', str(root / 'remote.git'), str(root / 'other'), '-b', 'main')
        other = root / 'other'
        for key, value in [('user.name', '测试'), ('user.email', 'test@example.invalid')]:
            run('git', '-C', str(other), 'config', key, value)
        (other / 'remote.txt').write_text('远程独立变动')
        run('git', '-C', str(other), 'add', '.')
        run('git', '-C', str(other), 'commit', '-m', '远程提交')
        run('git', '-C', str(other), 'push')
        remote_head = sync.git('ls-remote', 'origin', 'refs/heads/main').split()[0]
        (sync.ROOT / 'local.txt').write_text('本地独立变动')
        try:
            sync.sync(sync.changes())
        except subprocess.CalledProcessError:
            pass
        else:
            raise AssertionError('远程分叉必须拒绝推送')
        assert sync.git('ls-remote', 'origin', 'refs/heads/main').split()[0] == remote_head
        assert (sync.ROOT / 'local.txt').exists()
    sync.ROOT = original
    with tempfile.TemporaryDirectory() as tmp:
        directory=Path(tmp);childfile=directory/'child.pid';fake=directory/'git'
        fake.write_text('#!/usr/bin/python3\nimport os,time\npid=os.fork()\nif pid==0:\n open(os.environ["TEST_SYNC_CHILD_PID"],"w").write(str(os.getpid()))\n time.sleep(60)\nelse:\n time.sleep(60)\n')
        fake.chmod(0o755);real_popen=subprocess.Popen
        def short_process(*args,**kwargs):
            process=real_popen(*args,**kwargs);communicate=process.communicate
            process.communicate=lambda input=None,timeout=None:communicate(input,timeout=.2 if timeout is not None else None)
            return process
        with patch.dict(os.environ,PATH=str(directory)+os.pathsep+os.environ['PATH'],TEST_SYNC_CHILD_PID=str(childfile)),patch.object(sync.subprocess,'Popen',short_process):
            try:sync.git('timeout-fixture')
            except subprocess.TimeoutExpired:pass
            else:raise AssertionError('Timeout not raised')
        child=int(childfile.read_text());status=Path(f'/proc/{child}/stat')
        assert not status.exists() or status.read_text().split()[2]=='Z','Pack child survived timeout'
    print('PASS：超时清理整个Git进程组')
    print('PASS：自动同步、中文/记忆约束、暂存保护、分叉不强推')


if __name__ == '__main__':
    main()
