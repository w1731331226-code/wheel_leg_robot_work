"""One-off terminal archive;reuse project lock, never rewrite histories."""
import fcntl,json,os,subprocess,time
from pathlib import Path

root=Path('/home/wmt/wheel_leg_robot_work')
out=root/'wheelleg_warp/results/paper_recovery_20261004/execution_input_history_v1'

def git(*args,stdin=None):
    return subprocess.check_output(['git','--literal-pathspecs',*args],cwd=root,input=stdin,stderr=subprocess.STDOUT)

print('Waiting on existing study project-write.lock;no running data staged',flush=True)
with (root/'.git/project-write.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    complete=json.loads((out/'study_completion.json').read_text())
    assert complete['verified'] and complete['policy_samples']==1200000 and complete['evaluations']==1312
    assert not git('diff','--cached','--name-only').strip(),'Respect unrelated staged content'
    untracked={os.fsdecode(x) for x in git('ls-files','--others','--exclude-standard','-z').split(b'\0') if x}
    cohorts=[out/'study_reference'/a for a in ('B0','B1-route')]
    cohorts += [out/'study_runs'/a/str(s) for s in (23301,23302,23303) for a in ('H0','H1')]
    batches=[]
    for directory in cohorts:
        files=sorted(f for f in directory.rglob('*') if f.is_file() and str(f.relative_to(root)) in untracked)
        group=[];size=0
        for path in files:
            n=path.stat().st_size;assert n<95*1024**2
            if group and size+n>1500000000:batches.append((group,size));group=[];size=0
            group.append(path);size+=n
        if group:batches.append((group,size))
    for i,(files,size) in enumerate(batches,1):
        assert not git('diff','--cached','--name-only').strip()
        with (root/'PROJECT_MEMORY.md').open('a') as f:
            f.write(f'\n### 执行输入科学队列终态归档 {i}/{len(batches)}\n\n保存{len(files)}文件/{size}B，成功失败、模型/RMS和原始轨迹均保留。仅终态数据归档，不代表收益门、正式五种子或论文目标完成；fixed收益扩展仍关闭。\n')
        paths=[str(f.relative_to(root)) for f in files]+['PROJECT_MEMORY.md']
        if i==1:paths += [str((out/n).relative_to(root)) for n in ('study_completion.json','study_evaluation_ledger.json','study_progress.json')]
        git('add','--pathspec-from-file=-','--pathspec-file-nul',stdin=b'\0'.join(os.fsencode(p) for p in paths)+b'\0')
        git('commit','-q','-m',f'归档：保存科学队列终态数据第{i}批','-m','按1.5GB以内批次保留全部原始记录与失败模型，更新项目记忆；不修改收益门或科学结论。')
        while True:
            p=subprocess.run(['git','push','origin','HEAD:refs/heads/main'],cwd=root)
            if p.returncode==0:break
            print('Push failed;local commit preserved;retry after60s',flush=True);time.sleep(60)
        print('ARCHIVED',i,len(files),size,flush=True)
    assert git('rev-parse','HEAD').strip()==git('ls-remote','origin','refs/heads/main').split()[0]
    print('TERMINAL DATA ARCHIVE COMPLETE;remoteHEAD matches local;lock released',flush=True)
