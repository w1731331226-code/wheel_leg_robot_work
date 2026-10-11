"""Bounded local object packing for reachability bitmaps; never rewrite commits."""
import fcntl
import json
import shutil
import tempfile
from pathlib import Path
from git_preseed import ROOT, git, save

BATCH=256*1024**2


def prepare(root=ROOT,budget=BATCH):
    root=Path(root);gitdir=Path(git(root,'rev-parse','--absolute-git-dir').decode().strip())
    with (gitdir/'project-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=git(root,'rev-parse','HEAD');refs=git(root,'show-ref');status=git(root,'status','--porcelain','-uno')
        git(root,'fsck','--connectivity-only','--no-dangling')
        paths=sorted(p for p in (gitdir/'objects').glob('[0-9a-f][0-9a-f]/*')
                     if len(p.name)==38 and all(c in '0123456789abcdef' for c in p.name))
        ids=[p.parent.name+p.name for p in paths]
        sizes=git(root,'cat-file','--batch-check=%(objectsize)',input=('\n'.join(ids)+'\n').encode()).decode().splitlines() if ids else []
        assert len(sizes)==len(ids)
        items=list(zip(ids,map(int,sizes)))
        if any(size>budget for _,size in items):raise RuntimeError('A loose object exceeds the local packing budget')
        batches=[];batch=[];total=0
        for item in items:
            if batch and total+item[1]>budget:batches.append(batch);batch=[];total=0
            batch.append(item);total+=item[1]
        if batch:batches.append(batch)
        statefile=gitdir/'bitmap-prepare-state.json'
        state=dict(status='packing',head=before.decode().strip(),loose_objects=len(items),raw_bytes=sum(s for _,s in items),batches=len(batches),completed=[])
        save(statefile,state)
        try:
            for index,batch in enumerate(batches):
                if shutil.disk_usage(gitdir).free<2*budget+1024**3:raise RuntimeError('Insufficient free space for bounded object packing')
                name=git(root,'pack-objects',str(gitdir/'objects/pack/pack'),input=('\n'.join(oid for oid,_ in batch)+'\n').encode()).decode().strip()
                assert len(name)==40 and all(c in '0123456789abcdef' for c in name)
                idx=gitdir/f'objects/pack/pack-{name}.idx';pack=idx.with_suffix('.pack')
                assert pack.stat().st_size<=budget+1024**2
                git(root,'verify-pack',str(idx))
                # Git removes only loose copies already present in a pack; verified new packs remain intact.
                git(root,'prune-packed')
                state['completed'].append(dict(batch=index+1,objects=len(batch),pack=name,bytes=pack.stat().st_size));save(statefile,state)
                print('BITMAP-PREP',index+1,'/',len(batches),flush=True)
            git(root,'multi-pack-index','write','--bitmap')
            git(root,'multi-pack-index','verify')
            git(root,'fsck','--connectivity-only','--no-dangling')
            assert before==git(root,'rev-parse','HEAD') and refs==git(root,'show-ref') and status==git(root,'status','--porcelain','-uno')
            state.update(status='complete',head_refs_and_tracked_worktree_unchanged=True,connectivity_before_after=True)
            save(statefile,state);return state
        except BaseException as error:
            state.update(status='failed',error=str(error));save(statefile,state);raise


def self_check():
    import os
    from git_preseed import pack_report
    with tempfile.TemporaryDirectory(prefix='wheelleg-bitmap-test-') as td:
        root=Path(td);git(root,'init','-b','main');git(root,'config','user.name','测试');git(root,'config','user.email','test@invalid')
        (root/'PROJECT_MEMORY.md').write_text('初态');git(root,'add','.');git(root,'commit','-m','初始化测试')
        blobs=[]
        for i in range(4):
            raw=os.urandom(8192);(root/f'data{i}').write_bytes(raw);blobs.append((git(root,'hash-object','-w','--stdin',input=raw).decode().strip(),raw))
        git(root,'add','.');git(root,'commit','-m','保存原数据');head=git(root,'rev-parse','HEAD').decode().strip()
        tree=git(root,'mktree',input=''.join(f'100644 blob {oid}\t{oid}\n' for oid,_ in blobs).encode()).decode().strip()
        aux=git(root,'commit-tree',tree,input='测试辅助对象\n'.encode()).decode().strip();git(root,'update-ref','refs/test/aux',aux)
        try:pack_report(root,head,[aux],limit=4096)
        except RuntimeError:pass
        else:raise AssertionError('Unindexed oversized pack accepted')
        result=prepare(root,budget=12000);assert result['status']=='complete' and len(result['completed'])>=4
        assert pack_report(root,head,[aux],limit=4096)['bytes']<4096
        for oid,raw in blobs:assert git(root,'cat-file','blob',oid)==raw
        assert all((root/f'data{i}').read_bytes()==raw for i,(_,raw) in enumerate(blobs))
        print('PASS bounded multi-pack bitmap excludes remote blobs; all object bytes, HEAD, refs and worktree preserved',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    self_check() if args.self_check else prepare()
