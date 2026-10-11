"""Local bare-remote test: bounded auxiliary commits, hooks, original history preserved."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import git_preseed as transfer


def main():
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory);root=base/'work';root.mkdir();remote=base/'remote.git'
        transfer.git(base,'init','--bare',str(remote));transfer.git(root,'init','-b','main')
        transfer.git(root,'config','user.name','同步测试');transfer.git(root,'config','user.email','test@example.invalid')
        (root/'.githooks').mkdir();shutil.copy2(transfer.ROOT/'.githooks/commit-msg',root/'.githooks/commit-msg')
        transfer.git(root,'config','core.hooksPath','.githooks')
        (root/'PROJECT_MEMORY.md').write_text('# 初态\n');transfer.git(root,'add','.');transfer.git(root,'commit','-m','初始化测试')
        transfer.git(root,'remote','add','origin',str(remote));transfer.git(root,'push','-u','origin','main')
        for i in range(4):(root/f'data{i}.bin').write_bytes(os.urandom(8192))
        (root/'PROJECT_MEMORY.md').write_text('# 大于单批预算的原提交\n');transfer.git(root,'add','.');transfer.git(root,'commit','-m','保存大批数据')
        head=transfer.git(root,'rev-parse','HEAD').strip();before=transfer.git(root,'status','--porcelain')
        transfer.run(root,chunk_bytes=12000)
        assert transfer.git(root,'rev-parse','HEAD').strip()==head
        assert transfer.git(root,'status','--porcelain')==before
        assert transfer.git(root,'ls-remote','origin','refs/heads/main').split()[0]==head
        assert len(transfer.git(root,'ls-remote','--heads','origin').splitlines())==1
        state=json.loads((root/'.git/sync-transfer.json').read_text());assert state['status']=='complete' and state['chunks']>=4 and state['temporary_ref_removed']
        for i in range(4):assert transfer.git(remote,'show',f'main:data{i}.bin')==(root/f'data{i}.bin').read_bytes()
        auxiliary=Path(state['helper_repository'])
        messages=transfer.git(auxiliary,'log','--format=%s').decode().splitlines();assert all(m.startswith('同步：') for m in messages)
        print('PASS preseed boundedchunks/hookedcommits/originalHEAD+worktree/remote blobs/temprefcleanup')


if __name__=='__main__':main()
