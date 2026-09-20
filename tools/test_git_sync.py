"""标准库集成检查：自动提交/推送、钩子拒绝、暂存区保护、远程分叉保护。"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
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
    print('PASS：自动同步、中文/记忆约束、暂存保护、分叉不强推')


if __name__ == '__main__':
    main()
