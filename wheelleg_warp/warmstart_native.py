"""迁移已有32/32优胜策略到1024环境，三独立随机流保守续训准入。"""
import argparse,hashlib,json,os,pickle,shutil,subprocess,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
from benchmark_parallel import TimedPPO
from train_yaw import evaluate
from pretrain_yaw import summarize,selection_key
from dashboard.live_env import atomic_json as write


def fingerprint(model):return hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()


def init(out):
    old=ROOT/'wheelleg_ppo/tools/results/pilot_v2_M3_seed1609_resume1'
    best=json.loads((old/'selection.json').read_text())['best'];origin=Path(best['path'])
    assert best['success_count']==32 and best['complete']
    out.mkdir(parents=True,exist_ok=False);(out/'bootstrap').mkdir()
    for ext in ('.zip','.pkl'):shutil.copy2(str(origin)+ext,out/'bootstrap'/('policy'+ext))
    cases=json.loads((ROOT/'wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21/protocol_manifest.json').read_text())['sets']['selection']
    sources=[Path(__file__),ROOT/'wheelleg_warp/benchmark_parallel.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/models.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/live.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    hashes.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'bootstrap').iterdir()})
    p=dict(name='native-1024-warmstart-admission',warm_start=True,seeds=[1609,1610,1611],environments=1024,n_steps=50,
        policy_steps=512000,learning_rate=.00003,target_kl=.005,source_checkpoint=str(origin),source_summary=best,
        development_cases=cases,source_sha256=hashes,
        admission='初始策略32/32复核；每个独立续训末次至少30/32，三者合计至少93/96；保留初始/中间/末次最优，不改成功阈值',
        limitations=['共享同一已训练CPU优胜起点，不是三次从头初始化','策略、优化器、归一化迁移；1024训练环境和随机流重新开始','学习率为原来的1/10，target_kl=0.005；固定512000新增步准入，未冒充正式全预算完成','最终保留集不使用'])
    write(out/'protocol.json',p);write(out/'status.json',dict(status='initial_validation'))
    torch.set_num_threads(1);model=PPO.load(str(out/'bootstrap/policy.zip'),device='cpu')
    rows=evaluate(model,str(out/'bootstrap/policy.pkl'),'diff3',cases);summary=summarize(rows,[c['seed'] for c in cases])
    assert summary['success_count']==32 and abs(summary['mean_yaw_score_deg']-best['mean_yaw_score_deg'])<1e-6
    write(out/'initial.json',dict(summary=summary,runs=rows,policy_steps=model.num_timesteps,policy_sha256=fingerprint(model)))


def load(out):
    p=json.loads((out/'protocol.json').read_text())
    for name,digest in p['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return p


def run(out,seed):
    from native.live import LiveNativeEnv
    p=load(out);assert seed in p['seeds'];directory=out/f'round_{seed}';directory.mkdir(exist_ok=False)
    torch.set_num_threads(1);started=time.perf_counter();initial=json.loads((out/'initial.json').read_text())
    raw=LiveNativeEnv(directory/'live',start_steps=initial['policy_steps'],n=1024,stage=3,seed=730000+(seed-1609)*1024,phase='validation_training')
    env=VecNormalize.load(str(out/'bootstrap/policy.pkl'),VecCheckNan(raw,raise_exception=True));env.training=True;env.norm_reward=False
    model=TimedPPO.load(str(out/'bootstrap/policy.zip'),env=env,device='cpu',n_steps=50,learning_rate=p['learning_rate'],target_kl=p['target_kl'])
    model.timings=[];model.set_random_seed(seed);start_step=model.num_timesteps
    assert fingerprint(model)==initial['policy_sha256'] and model.n_envs==1024
    with (out/'bootstrap/policy.pkl').open('rb') as f:old_norm=pickle.load(f)
    np.testing.assert_array_equal(env.obs_rms.mean,old_norm.obs_rms.mean);np.testing.assert_array_equal(env.obs_rms.var,old_norm.obs_rms.var)
    assert env.obs_rms.count==old_norm.obs_rms.count
    records=[];train_seconds=0.;initial_weights={k:v.clone() for k,v in model.policy.state_dict().items()}
    try:
        for extra in (0,256000,512000):
            if extra:
                write(out/'status.json',dict(status='training',current=directory.name))
                start=time.perf_counter();model.learn(total_timesteps=start_step+extra-model.num_timesteps,reset_num_timesteps=False);train_seconds+=time.perf_counter()-start
                assert model.num_timesteps==start_step+extra
                assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
                assert any(not torch.equal(initial_weights[k],v) for k,v in model.policy.state_dict().items())
            prefix=directory/f'step_{model.num_timesteps}';model.save(prefix);env.save(str(prefix)+'.pkl')
            write(directory/'progress.json',dict(status='evaluating',policy_steps=model.num_timesteps,additional_steps=extra,train_seconds=train_seconds))
            write(out/'status.json',dict(status='evaluating',current=directory.name))
            if extra:
                rows=evaluate(model,str(prefix)+'.pkl','diff3',p['development_cases']);summary=summarize(rows,[c['seed'] for c in p['development_cases']])
            else:rows=initial['runs'];summary=initial['summary']
            record=dict(policy_steps=model.num_timesteps,additional_steps=extra,path=str(prefix),summary=summary,curve_summary=summary,train_seconds=train_seconds,runs=rows)
            records.append(record);write(Path(str(prefix)+'.json'),record);write(directory/'curve.json',records)
        load(out);selected=min(records,key=lambda r:selection_key(r['summary'],r['additional_steps']))
        result=dict(passed=True,seed=seed,n_steps=50,policy_steps=model.num_timesteps,initial_policy_sha256=initial['policy_sha256'],
            train_seconds=train_seconds,total_seconds=time.perf_counter()-started,records=records,run_directory=str(directory),
            selected=dict(path=selected['path'],policy_steps=selected['policy_steps'],summary=selected['summary']),initial_reused_verified=True)
        write(directory/'completed.json',result)
    except Exception as exc:write(directory/'failed.json',dict(error=repr(exc)));raise
    finally:env.close()


def orchestrate(out):
    p=load(out);rows=[]
    for seed in p['seeds']:
        name=f'round_{seed}';write(out/'status.json',dict(status='initializing',current=name))
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.run([sys.executable,'-u',str(Path(__file__)),'worker','--output',str(out),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
        if child.returncode:write(out/'status.json',dict(status='failed',current=name,returncode=child.returncode));raise RuntimeError(name)
        rows.append(json.loads((out/name/'completed.json').read_text()))
    summaries=[r['records'][-1]['summary'] for r in rows]
    passed=all(s['complete'] and s['success_count']>=30 for s in summaries) and sum(s['success_count'] for s in summaries)>=93
    write(out/'comparison.json',dict(complete=True,protocol=p,results=rows,admission=dict(passed=passed,last_success_counts=[s['success_count'] for s in summaries])))
    write(out/'status.json',dict(status='completed' if passed else 'admission_failed',admission_passed=passed))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['init','worker','orchestrate']);a.add_argument('--output',type=Path,required=True);a.add_argument('--seed',type=int);args=a.parse_args()
    out=args.output.resolve()
    if args.command=='init':init(out)
    elif args.command=='worker':run(out,args.seed)
    else:orchestrate(out)
