"""Local bare-remote test: bounded auxiliary commits, hooks, original history preserved."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
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
        original_git=transfer.git;calls=[]
        def fail_first_push(cwd,*args,**kwargs):
            if args[0]=='push' and any('refs/heads/codex/sync-preseed-' in str(x) for x in args):
                calls.append(Path(cwd));raise RuntimeError('Simulated transport failure after auxiliary commit')
            return original_git(cwd,*args,**kwargs)
        with patch.object(transfer,'git',fail_first_push):
            try:transfer.run(root,chunk_bytes=12000)
            except RuntimeError:pass
            else:raise AssertionError('Injected transport failure missed')
        assert calls==[root],'Push must use source repository authentication context'
        state=json.loads((root/'.git/sync-transfer.json').read_text());assert state['status']=='failed' and state['completed_chunks']==0
        pending=state['auxiliary_head']
        measured=transfer.pack_report
        reports=[]
        def indexed_report(cwd,tip,excluded,limit=transfer.CHUNK):
            # Test-only packing makes a bitmap after all auxiliary metadata is present.
            original_git(cwd,'repack','-a','-d','--write-bitmap-index')
            report=measured(cwd,tip,excluded,limit);reports.append(report);return report
        with patch.object(transfer,'pack_report',indexed_report):
            transfer.run(root,chunk_bytes=12000,resume=True)
        assert len(reports)==1 and reports[0]['bytes']<4096
        try:measured(root,head.decode(),[],limit=4096)
        except RuntimeError as error:assert 'budget' in str(error)
        else:raise AssertionError('Oversized pack accepted')
        assert transfer.git(root,'rev-parse','HEAD').strip()==head
        assert transfer.git(root,'status','--porcelain')==before
        assert transfer.git(root,'ls-remote','origin','refs/heads/main').split()[0]==head
        assert len(transfer.git(root,'ls-remote','--heads','origin').splitlines())==1
        state=json.loads((root/'.git/sync-transfer.json').read_text());assert state['status']=='complete' and state['chunks']>=4 and state['temporary_ref_removed']
        for i in range(4):assert transfer.git(remote,'show',f'main:data{i}.bin')==(root/f'data{i}.bin').read_bytes()
        auxiliary=Path(state['helper_repository'])
        messages=transfer.git(auxiliary,'log','--format=%s').decode().splitlines();assert all(m.startswith('同步：') for m in messages)
        assert len(messages)==state['chunks'] and transfer.git(auxiliary,'rev-list','--max-parents=0','HEAD').decode().strip()==pending
        print('PASS preseed measured small bitmap pack/oversize rejection/hooked commits/original HEAD+worktree/remote data/resume')


if __name__=='__main__':main()
