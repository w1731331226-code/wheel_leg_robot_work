"""单一1024原生基线：真实状态录像、分轮续训、开发集选模及有界停止。"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib,json,os
from pathlib import Path
import shutil,subprocess,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecCheckNan,VecNormalize
from benchmark_parallel import TimedPPO
from dashboard.live_env import atomic_json as write
from train_yaw import evaluate
from pretrain_yaw import summarize,selection_key
from ppo_env import sample_scenario


def significant(new,anchor):
    return new['complete'] and (new['success_count']>anchor['success_count'] or
        (new['success_count']==anchor['success_count'] and new['mean_yaw_score_deg']<anchor['mean_yaw_score_deg']*.995))


def initialize(output,source):
    comparison=json.loads((source/'comparison.json').read_text());assert comparison['complete']
    groups={n:[r for r in comparison['results'] if r['n_steps']==n] for n in sorted({r['n_steps'] for r in comparison['results']})}
    def rank(rows):
        summaries=[r['records'][-1]['summary'] for r in rows]
        assert len(rows)==3 and all(s['complete'] for s in summaries)
        return (-sum(s['success_count'] for s in summaries),sum(s['mean_yaw_score_deg'] for s in summaries))
    eligible={n:rows for n,rows in groups.items() if len(rows)==3 and all(r['records'][-1]['summary']['complete'] and r['records'][-1]['summary']['success_count']>=30 for r in rows) and sum(r['records'][-1]['summary']['success_count'] for r in rows)>=93}
    assert eligible,'没有配置达到三种子成功率准入门，禁止清理旧基线'
    length=min(eligible,key=lambda n:rank(eligible[n]));rows=eligible[length]
    summaries=[r['records'][-1]['summary'] for r in rows]
    # 旧CPU已有31/32；先守住成功率，再比较偏航。未过门不清理旧模型。
    assert min(s['success_count'] for s in summaries)>=30 and sum(s['success_count'] for s in summaries)>=93,'1024候选尚未达到旧CPU成功率参照'
    chosen=min(rows,key=lambda r:selection_key(r['records'][-1]['summary']))
    origin=Path(chosen.get('run_directory',str(source/f"rollout_{length}_seed_{chosen['seed']}")))/f"step_{chosen['policy_steps']}"
    output.mkdir(parents=True,exist_ok=False);(output/'bootstrap').mkdir()
    for ext in ('.zip','.pkl'):shutil.copy2(str(origin)+ext,output/'bootstrap'/('policy'+ext))
    source_paths=[Path(__file__),ROOT/'wheelleg_warp/benchmark_parallel.py',ROOT/'wheelleg_warp/dashboard/live_env.py']+list((ROOT/'wheelleg_warp/native').glob('*.py'))
    hashes={k:v for k,v in json.loads((ROOT/'wheelleg_warp/CPU_REFERENCE.json').read_text())['source_sha256'].items() if k!='wheelleg_ppo/tools/resume_yaw.py'}
    hashes.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    protocol=dict(created=datetime.now().astimezone().isoformat(),name='native-1024-formal-v1',environments=1024,
        n_steps=length,policy_device='cpu',physics_device='cuda:0',seed=chosen['seed'],max_rounds=10,patience=3,
        meaningful_yaw_improvement=.005,steps_per_round=2048000,inherited_steps=chosen['policy_steps'],
        bootstrap_summary=chosen['records'][-1]['summary'],bootstrap_source=str(origin),
        development_cases=comparison['protocol']['development_cases'],
        final_cases=[dict(seed=s,scenario=asdict(sample_scenario('test_iid',s,3))) for s in range(95000,95064)],
        selection='开发集成功数优先，完整轨迹Jpsi次之；任何严格改善保留最佳，停滞以成功数增加或相对锚点Jpsi降低0.5%计',
        stopping='连续3轮无实质改善或最多10轮；最终保留集只在停止后评估一次，不据此重选模型或继续训练',
        environment_protocol='每轮独立1024个stage3训练场景；回合内/本轮场景固定，下轮重新抽样；恢复策略、优化器和归一化，重置环境及随机流',
        exact_trajectory_resume=False,source_sha256=hashes,
        known_limitations=['开发集反复用于选模，不冒充最终泛化成绩','历史CPU/GPU物理全状态等价门未全通过','不能保证全局最优，只报告固定预算和停止规则下的最佳检查点'])
    write(output/'protocol.json',protocol)
    write(output/'selection.json',dict(best=dict(round=0,path=str(output/'bootstrap/policy'),summary=protocol['bootstrap_summary']),anchor=protocol['bootstrap_summary'],stagnant_rounds=0,rounds=[]))
    write(output/'status.json',dict(status='ready',round=0,new_steps=0,updated=time.time()))


def protocol(output):
    p=json.loads((output/'protocol.json').read_text())
    for name,digest in p['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return p


class Progress(BaseCallback):
    def __init__(self,output,directory,p,round_index,round_start,started):
        super().__init__();self.output=output;self.directory=directory;self.p=p;self.round=round_index;self.start=round_start;self.started=started
    def _on_step(self):return True
    def _on_rollout_start(self):
        row=dict(status='training',round=self.round,policy_steps=self.num_timesteps,new_steps=self.num_timesteps-self.p['inherited_steps'],
            round_steps=self.num_timesteps-self.start,round_budget=self.p['steps_per_round'],updated=time.time(),
            elapsed_round_seconds=time.perf_counter()-self.started,train_seconds=sum(t['seconds'] for t in self.model.timings),source_run=str(self.directory),pid=os.getpid())
        write(self.directory/'progress.json',row);write(self.output/'status.json',row)


def run_round(output,round_index):
    p=protocol(output);directory=output/f'round_{round_index:03d}';directory.mkdir(exist_ok=False)
    previous=output/'bootstrap/policy' if round_index==1 else output/f'round_{round_index-1:03d}/last'
    from native.live import LiveNativeEnv
    torch.set_num_threads(1);started=time.perf_counter()
    model=TimedPPO.load(str(previous)+'.zip',device='cpu');model.timings=[];round_start=model.num_timesteps
    raw=LiveNativeEnv(directory/'live',start_steps=round_start,n=1024,stage=3,seed=800000+round_index*1024)
    env=VecNormalize.load(str(previous)+'.pkl',VecCheckNan(raw,raise_exception=True));env.training=True;env.norm_reward=False
    model.set_env(env);model.set_random_seed(p['seed']+round_index*100003)
    write(directory/'run_config.json',dict(round=round_index,stage=3,start_policy_steps=round_start,
        train_bank_seed=800000+round_index*1024,exploration_seed=p['seed']+round_index*100003,resume_from=str(previous),exact_trajectory_resume=False))
    try:
        start=time.perf_counter();model.learn(total_timesteps=p['steps_per_round'],reset_num_timesteps=False,
            callback=Progress(output,directory,p,round_index,round_start,started));train_seconds=time.perf_counter()-start
        assert model.num_timesteps==round_start+p['steps_per_round']
        assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
        prefix=directory/'last';model.save(prefix);env.save(str(prefix)+'.pkl')
        write(output/'status.json',dict(status='evaluating',round=round_index,new_steps=model.num_timesteps-p['inherited_steps'],
            policy_steps=model.num_timesteps,source_run=str(directory),updated=time.time(),pid=os.getpid(),train_seconds=train_seconds))
        rows=evaluate(model,str(prefix)+'.pkl','diff3',p['development_cases']);summary=summarize(rows,[c['seed'] for c in p['development_cases']])
        protocol(output)
        write(directory/'completed.json',dict(passed=True,round=round_index,policy_steps=model.num_timesteps,train_seconds=train_seconds,
            total_seconds=time.perf_counter()-started,summary=summary,runs=rows,path=str(prefix),updates=len(model.timings)))
    except Exception as exc:write(directory/'failed.json',dict(error=repr(exc),policy_steps=model.num_timesteps));raise
    finally:env.close()


def orchestrate(output):
    p=protocol(output);selection=json.loads((output/'selection.json').read_text())
    assert not selection['rounds'],'不自动覆盖或伪装精确续训'
    for index in range(1,p['max_rounds']+1):
        write(output/'status.json',dict(status='initializing',round=index,new_steps=(index-1)*p['steps_per_round'],updated=time.time()))
        with (output/f'round_{index:03d}.log').open('x') as log:
            child=subprocess.run([sys.executable,'-u',str(Path(__file__)),'round','--output',str(output),'--round',str(index)],stdout=log,stderr=subprocess.STDOUT)
        if child.returncode:
            write(output/'status.json',dict(status='failed',round=index,returncode=child.returncode,updated=time.time()));raise RuntimeError('训练轮次失败')
        row=json.loads((output/f'round_{index:03d}/completed.json').read_text());selection['rounds'].append(row)
        if selection_key(row['summary'])<selection_key(selection['best']['summary']):
            selection['best']=dict(round=index,path=row['path'],summary=row['summary'])
        if significant(row['summary'],selection['anchor']):selection['stagnant_rounds']=0;selection['anchor']=row['summary']
        else:selection['stagnant_rounds']+=1
        write(output/'selection.json',selection)
        if selection['stagnant_rounds']>=p['patience']:break
    write(output/'status.json',dict(status='final_evaluation',round=index,new_steps=index*p['steps_per_round'],updated=time.time()))
    best=selection['best'];model=TimedPPO.load(best['path']+'.zip',device='cpu')
    rows=evaluate(model,best['path']+'.pkl','diff3',p['final_cases'])
    write(output/'final_evaluation.json',dict(checkpoint=best,summary=summarize(rows,[c['seed'] for c in p['final_cases']]),runs=rows,used_for_selection=False))
    write(output/'status.json',dict(status='completed',round=index,new_steps=index*p['steps_per_round'],best=best,
        stop_reason='plateau' if selection['stagnant_rounds']>=p['patience'] else 'round_budget',updated=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['init','round','orchestrate'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source',type=Path);parser.add_argument('--round',type=int)
    a=parser.parse_args()
    if a.command=='init':initialize(a.output.resolve(),a.source.resolve())
    elif a.command=='round':run_round(a.output.resolve(),a.round)
    else:orchestrate(a.output.resolve())
