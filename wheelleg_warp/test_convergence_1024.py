"""1024环境的受控收敛实验；共同CPU开发评估，不启动旧正式训练。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
from benchmark_parallel import TimedPPO
from train_yaw import evaluate
from pretrain_yaw import summarize


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def init(output):
    sweep=ROOT/'wheelleg_warp/results/parallel_sweep_20260921'
    assert json.loads((sweep/'episode_return_after.json').read_text())['passed']
    assert json.loads((sweep/'native_pair.json').read_text())['passed']
    config=json.loads((ROOT/'wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21/training_config.json').read_text())
    cases=json.loads((ROOT/'wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21/protocol_manifest.json').read_text())['sets']['selection']
    sources=[Path(__file__),ROOT/'wheelleg_warp/benchmark_parallel.py',ROOT/'wheelleg_ppo/tools/train_yaw.py',ROOT/'wheelleg_ppo/tools/ppo_env.py']+list((ROOT/'wheelleg_warp/native').glob('*.py'))
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(created=datetime.now().astimezone().isoformat(),environments=1024,seeds=[1609,1610,1611],
        rollout_lengths=[16,250],policy_steps=2048000,checkpoints=[0,256000,512000,1024000,2048000],
        config=config,development_cases=cases,curve_cases=cases[:8],train_bank_seed=730000,stage=3,
        primary='共同CPU开发32例末次检查点成功数；完整轨迹Jpsi为次指标，按配对种子报告',
        selection_rule='不以损失下降代替效果；若成功数与Jpsi存在权衡或种子不一致，不声称唯一收敛最优',
        limitations=['实验固定1024个训练场景，每回合回到对应场景，未实现原正式课程/重采样',
          '2,048,000为两种rollout长度的共同完整更新预算，比原正式2,000,000多2.4%；结果不混入旧正式训练',
          '中间评估在越过阈值的完整更新后执行，报告实际步数；末次预算严格一致',
          '只使用已有开发选择集，未打开研究gate/最终保留集；3种子不等于统计显著性或全局收敛证明',
          'GPU后端与CPU物理全状态等价门仍有历史失败；共同CPU评估用于检查迁移效果'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    write(output/'protocol.json',protocol)


def load(output):
    p=json.loads((output/'protocol.json').read_text())
    for name,digest in p['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return p


def run(output,length,seed):
    p=load(output);assert length in p['rollout_lengths'] and seed in p['seeds']
    directory=output/f'rollout_{length}_seed_{seed}';directory.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    from native.environment import NativeEnv
    started=time.perf_counter()
    raw=NativeEnv(1024,stage=p['stage'],seed=p['train_bank_seed'])
    env=VecNormalize(VecCheckNan(raw,raise_exception=True),**p['config']['normalization'])
    kwargs=dict(p['config']['ppo']);kwargs['n_steps']=length
    model=TimedPPO('MlpPolicy',env,device='cpu',seed=seed,**kwargs);model.timings=[]
    env.reset()
    initial={k:v.clone() for k,v in model.policy.state_dict().items()}
    digest=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in initial.values())).hexdigest()
    setup=time.perf_counter()-started;records=[];training_seconds=0.;evaluation_seconds=0.
    write(directory/'run_config.json',dict(seed=seed,n_steps=length,initial_policy_sha256=digest,setup_seconds=setup,ppo=kwargs))
    try:
        for target in p['checkpoints']:
            if target:
                start=time.perf_counter()
                model.learn(total_timesteps=target-model.num_timesteps,reset_num_timesteps=False)
                training_seconds+=time.perf_counter()-start
                assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
                assert any(not torch.equal(initial[k],v) for k,v in model.policy.state_dict().items())
            steps=model.num_timesteps
            prefix=directory/f'step_{steps}';model.save(prefix);env.save(str(prefix)+'.pkl')
            write(directory/'progress.json',dict(status='evaluating',policy_steps=steps,target=p['policy_steps'],train_seconds=training_seconds))
            cases=p['development_cases'] if target==p['policy_steps'] else p['curve_cases']
            start=time.perf_counter();rows=evaluate(model,str(prefix)+'.pkl','diff3',cases)
            elapsed=time.perf_counter()-start;evaluation_seconds+=elapsed
            summary=summarize(rows,[c['seed'] for c in cases])
            curve=summarize(rows[:8],[c['seed'] for c in p['curve_cases']])
            losses={k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)} if target else {}
            assert all(np.isfinite(v) for v in losses.values())
            record=dict(policy_steps=steps,requested_steps=target,ppo_updates=len(model.timings),
                train_seconds=training_seconds,evaluation_seconds=evaluation_seconds,total_seconds=time.perf_counter()-started,
                summary=summary,curve_summary=curve,losses=losses)
            write(Path(str(prefix)+'.json'),dict(**record,runs=rows));records.append(record)
            write(directory/'curve.json',records)
            write(directory/'progress.json',dict(status='training',**record))
            print(directory.name,steps,summary,flush=True)
        assert model.num_timesteps==p['policy_steps'];load(output)
        write(directory/'completed.json',dict(passed=True,seed=seed,n_steps=length,policy_steps=model.num_timesteps,
            initial_policy_sha256=digest,train_seconds=training_seconds,evaluation_seconds=evaluation_seconds,
            setup_seconds=setup,total_seconds=time.perf_counter()-started,ppo_updates=len(model.timings),records=records))
    except Exception as exc:
        write(directory/'failed.json',dict(error=repr(exc),policy_steps=model.num_timesteps));raise
    finally:env.close()


def orchestrate(output):
    p=load(output)
    for seed in p['seeds']:
        for length in p['rollout_lengths']:
            name=f'rollout_{length}_seed_{seed}'
            write(output/'status.json',dict(status='running',current=name))
            with (output/f'{name}.log').open('x') as log:
                completed=subprocess.run([sys.executable,'-u',str(Path(__file__)),'run','--output',str(output),'--length',str(length),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                write(output/'status.json',dict(status='failed',current=name,returncode=completed.returncode));raise RuntimeError(name)
    results=[]
    for seed in p['seeds']:
        pair=[json.loads((output/f'rollout_{n}_seed_{seed}/completed.json').read_text()) for n in p['rollout_lengths']]
        assert pair[0]['initial_policy_sha256']==pair[1]['initial_policy_sha256']
        results.extend(pair)
    write(output/'comparison.json',dict(protocol=p,results=results,complete=True))
    write(output/'status.json',dict(status='completed'))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('command',choices=['init','run','orchestrate']);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--length',type=int);a.add_argument('--seed',type=int);args=a.parse_args()
    if args.command=='init':init(args.output)
    elif args.command=='run':run(args.output,args.length,args.seed)
    else:orchestrate(args.output)
