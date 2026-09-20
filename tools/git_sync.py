#!/usr/bin/env python3
"""稳定文件自动快照、中文提交、推送当前分支；失败保留本地提交并重试。"""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MEMORY = 'PROJECT_MEMORY.md'


def git(*args, input=None):
    return subprocess.check_output(['git', '--literal-pathspecs', *args], cwd=ROOT, input=input, stderr=subprocess.STDOUT, timeout=120)


def changes():
    paths = git('ls-files', '-m', '-d', '-o', '--exclude-standard', '-z').split(b'\0')
    snapshot = {}
    for raw in set(paths) - {b''}:
        path = os.fsdecode(raw)
        try:
            stat = (ROOT / path).lstat()
            snapshot[path] = (stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            snapshot[path] = None
    return snapshot


def sync(previous):
    gitdir = Path(os.fsdecode(git('rev-parse', '--absolute-git-dir')).strip())
    with (gitdir / 'project-write.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {}
        if any((gitdir / p).exists() for p in ('index.lock', 'MERGE_HEAD', 'rebase-merge', 'rebase-apply', 'CHERRY_PICK_HEAD')):
            return {}
        branch = git('symbolic-ref', '--quiet', '--short', 'HEAD').decode().strip()
        remote = git('config', f'branch.{branch}.remote').decode().strip()
        target = git('config', f'branch.{branch}.merge').decode().strip()
        if remote == '.' or not target.startswith('refs/heads/'):
            raise RuntimeError('需要显式远程分支上游')
        current = changes()
        stable = sorted(p for p in current if p in previous and previous[p] == current[p])
        # 用户暂存区优先；不提交他人正在准备的索引。
        if stable and not git('diff', '--cached', '--name-only', '-z'):
            if MEMORY in current and MEMORY not in stable:
                return current
            large = [p for p in stable if current[p] and current[p][0] >= 95 * 1024**2]
            if large:
                raise RuntimeError(f'文件超过95MiB，需单独决定存储方式：{large}')
            now = datetime.now().astimezone().isoformat(timespec='seconds')
            with (ROOT / MEMORY).open('a') as f:
                f.write(f'\n### 自动同步快照 {now}\n\n')
                f.write('稳定落盘变动自动归档；本条仅记录文件变化，不代表测试或研究验收通过。\n\n')
                for p in stable:
                    f.write(f'- `{p}`\n')
            paths = sorted(set(stable) | {MEMORY})
            git('add', '-A', '--pathspec-from-file=-', '--pathspec-file-nul',
                input=b'\0'.join(os.fsencode(p) for p in paths) + b'\0')
            if git('diff', '--cached', '--name-only', '-z'):
                result = git('commit', '-m', f'同步：自动归档 {len(stable)} 项项目变更',
                             '-m', '记录稳定文件快照并更新长期项目记忆；未额外运行验收。')
                print(result.decode(), flush=True)
        # 普通 push 不改写历史；远程冲突/认证/网络失败留在日志并重试，不自动合并。
        head = git('rev-parse', 'HEAD').strip()
        remote_head = git('ls-remote', remote, target).split()
        if not remote_head or remote_head[0] != head:
            print(git('push', remote, f'HEAD:{target}').decode(), flush=True)
        return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error('同步间隔至少5秒')
    previous = {}
    while True:
        try:
            previous = sync(previous)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
            detail = exc.output.decode(errors='replace') if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            print(f'同步失败，保留本地状态，下轮重试：{detail}', flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
