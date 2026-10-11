"""Upload bounded blob batches on an owned temporary ref, preserving main history."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
CHUNK=64*1024**2
REDACT=''


def git(root,*args,input=None,timeout=1800):
    command=['git','-c','gc.auto=0','-c','pack.window=0','-c','pack.threads=2','-c','core.compression=1',*args]
    with subprocess.Popen(command,cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True) as process:
        try:output,_=process.communicate(input,timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.communicate();raise RuntimeError('Git operation timed out; process group stopped') from None
        if process.returncode:
            detail=output.decode(errors='replace')[-2000:]
            if REDACT:detail=detail.replace(REDACT,'<configured remote>')
            raise RuntimeError(f'Git {args[0]} failed: {detail}')
        return output


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def run(root=ROOT,chunk_bytes=CHUNK):
    global REDACT
    root=Path(root);gitdir=Path(git(root,'rev-parse','--absolute-git-dir').decode().strip());statefile=gitdir/'sync-transfer.json'
    if statefile.exists():raise RuntimeError('Transfer state already exists; inspect it before any continuation')
    branch=git(root,'symbolic-ref','--short','HEAD').decode().strip();head=git(root,'rev-parse','HEAD').decode().strip()
    remote=git(root,'config',f'branch.{branch}.remote').decode().strip();target=git(root,'config',f'branch.{branch}.merge').decode().strip()
    assert remote!='.' and target.startswith('refs/heads/')
    old=git(root,'ls-remote',remote,target).decode().split()[0]
    git(root,'merge-base','--is-ancestor',old,head)
    if old==head:return
    lines=git(root,'rev-list','--objects',head,'^'+old).decode().splitlines()
    sizes=git(root,'cat-file','--batch-check=%(objecttype) %(objectsize)',input=('\n'.join(x.split(' ',1)[0] for x in lines)+'\n').encode()).decode().splitlines()
    objects=[]
    for line,meta in zip(lines,sizes):
        kind,size=meta.split();size=int(size)
        if kind!='blob':continue
        oid,_,name=line.partition(' ');path=Path(name)
        if path.suffix.lower() in ('.mp4','.avi','.mov','.mkv','.webm','.m4v','.mpeg','.mpg','.wmv','.flv','.ogv','.mts','.m2ts','.3gp') or path.name=='.env' or (path.name.startswith('.env.') and path.name!='.env.example') or path.suffix in ('.pem','.key'):
            raise RuntimeError('Pending history contains prohibited media/credential path; no upload started')
        if size>=95*1024**2 or size>chunk_bytes:raise RuntimeError('Blob exceeds hook/chunk budget; no upload started')
        objects.append((oid,size))
    batches=[];batch=[];bytes_=0
    for item in objects:
        if batch and bytes_+item[1]>chunk_bytes:batches.append(batch);batch=[];bytes_=0
        batch.append(item);bytes_+=item[1]
    if batch:batches.append(batch)
    name='codex/sync-preseed-'+head[:12];ref='refs/heads/'+name
    if git(root,'ls-remote',remote,ref).strip():raise RuntimeError('Temporary remote ref already exists; refusing to take ownership')
    work=gitdir/('sync-preseed-'+head[:12]);work.mkdir();url=git(root,'remote','get-url',remote).decode().strip();REDACT=url
    state=dict(pid=os.getpid(),status='preparing',snapshot_head=head,original_remote_head=old,remote=remote,target=target,temporary_ref=ref,
        helper_repository=str(work),blob_count=len(objects),raw_bytes=sum(x[1] for x in objects),chunks=len(batches),completed_chunks=0,
        history_rewritten=False,force_push=False,started_epoch=time.time())
    save(statefile,state)
    try:
        git(work,'init','-b',name);(work/'.git/objects/info/alternates').write_text(str(gitdir/'objects')+'\n')
        git(work,'config','user.name','项目同步');git(work,'config','user.email','sync@example.invalid');git(work,'remote','add','origin',url)
        hooks=root/'.githooks';assert (hooks/'commit-msg').is_file()
        for number,items in enumerate(batches,1):
            index=''.join(f'100644 {oid}\t.sync-blobs/{oid}\n' for oid,_ in items).encode()
            git(work,'update-index','--add','--index-info',input=index)
            (work/'PROJECT_MEMORY.md').write_text(f'# 分批同步传输记录\n\n原主分支历史保持不变。快照 {head}，已登记批次 {number}/{len(batches)}，本批 {len(items)} 对象。\n')
            git(work,'add','PROJECT_MEMORY.md')
            git(work,'-c',f'core.hooksPath={hooks}','commit','-m',f'同步：暂存对象批次{number}', '-m','仅辅助传输已有对象，主分支提交不重写；完成主分支快进后删除临时引用。')
            auxiliary=git(work,'rev-parse','HEAD').decode().strip();state.update(status='uploading',current_chunk=number,auxiliary_head=auxiliary);save(statefile,state)
            git(work,'push','origin',f'{auxiliary}:{ref}')
            assert git(root,'ls-remote',remote,ref).decode().split()[0]==auxiliary
            state['completed_chunks']=number;save(statefile,state);print(f'Uploaded {number}/{len(batches)} chunks',flush=True)
        # The alternate store knows local main, so this local fetch transfers only auxiliary metadata.
        tracking='refs/remotes/'+remote+'/'+name
        git(root,'fetch','--no-tags',str(work),f'{ref}:{tracking}')
        current_remote=git(root,'ls-remote',remote,target).decode().split()[0];git(root,'merge-base','--is-ancestor',current_remote,head)
        state['status']='pushing_original_history';save(statefile,state)
        git(root,'push',remote,f'{head}:{target}');assert git(root,'ls-remote',remote,target).decode().split()[0]==head
        git(root,'push',remote,'--delete',name);git(root,'update-ref','-d',tracking)
        state.update(status='complete',completed_epoch=time.time(),temporary_ref_removed=True,remote_snapshot_verified=True);save(statefile,state)
        print('Original history snapshot synchronized; temporary remote ref removed',flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error),failed_epoch=time.time());save(statefile,state);raise


if __name__=='__main__':run()
