"""Small paired policy interventions; fixed geometry, commands, and success gates."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
from dataclasses import asdict,replace
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from native.terrain import TerrainScenario,sample_terrain_v4
from ppo_env import sample_scenario
from terrain_eval import evaluate_terrain
from dashboard.live_env import atomic_json as write
from stable_baselines3.common.vec_env import VecNormalize
from benchmark_parallel import TimedPPO
from native.live import LiveNativeEnv
from native.terrain import bank

class MaskedPolicy:
    def __init__(self,policy,mask):self.policy=policy;self.mask=np.asarray(mask,dtype=np.float32)
    def predict(self,obs,deterministic=True):
        action,state=self.policy.predict(obs,deterministic=deterministic)
        return action*self.mask,state

def cases(start):
    rows=[];labels=[]
    for hi,h in enumerate((.018,.021,.024,.027)):
        for si,speed in enumerate((.5,.7,.9)):
            for rep in range(4):
                seed=start+hi*100+si*10+rep;s=sample_scenario('train',seed,3)
                rows.append(TerrainScenario(**{**asdict(s),'speed':speed*(-1 if rep<2 else 1),'height_l':0.,'height_r':0.},terrain='step',step_height_m=h,terrain_seed=seed,relative_attitude=True));labels.append('step')
    for i in range(16):
        seed=start+1000+i;rows.append(TerrainScenario(**asdict(sample_scenario('test_iid',seed,3)),terrain_seed=seed));labels.append('legacy')
    for label,kinds in [('surface',('cross_slope','split_level')),('advanced',('single_side_ramp','asymmetric_rough'))]:
        remaining={k:8 for k in kinds};seed=start+(2000 if label=='surface' else 3000)
        while any(remaining.values()):
            s=sample_terrain_v4(seed,'development');seed+=1
            if remaining.get(s.terrain,0):rows.append(s);labels.append(label);remaining[s.terrain]-=1
    assert len({r.terrain_seed for r in rows})==len(rows)==96
    return rows,labels

def summary(rows,labels):
    result={}
    for label in ('step','legacy','surface','advanced'):
        rs=[r for r,k in zip(rows,labels) if k==label]
        result[label]=dict(total=len(rs),success=sum(r['success'] and r['terrain_evidence_passed'] for r in rs),complete=sum(r['reason']=='completed' for r in rs),yaw_mean=float(np.mean([r['peak_deg'][2] for r in rs])))
    return result

def evaluate(policy,norm,rows,labels):
    result=evaluate_terrain(policy,norm,rows)
    return dict(summary=summary(result,labels),runs=result)

def transferred_virtual6(source,env):
    model=TimedPPO('MlpPolicy',env,device='cpu',n_steps=source.n_steps,batch_size=source.batch_size,n_epochs=source.n_epochs,
        learning_rate=source.learning_rate,gamma=source.gamma,gae_lambda=source.gae_lambda,clip_range=source.clip_range,
        ent_coef=source.ent_coef,vf_coef=source.vf_coef,max_grad_norm=source.max_grad_norm,target_kl=source.target_kl,policy_kwargs=source.policy_kwargs)
    state=source.policy.state_dict();order=[0,0,1,1,2,2];sign=torch.tensor([1,-1,1,-1,1,-1],dtype=state['action_net.bias'].dtype)
    state['action_net.weight']=state['action_net.weight'][order]*sign[:,None]
    state['action_net.bias']=state['action_net.bias'][order]*sign
    state['log_std']=state['log_std'][order]
    model.policy.load_state_dict(state)
    obs=np.random.default_rng(714).normal(size=(64,32)).astype(np.float32)
    old=source.predict(obs,deterministic=True)[0];new=model.predict(obs,deterministic=True)[0]
    np.testing.assert_allclose(new,old[:,order]*sign.numpy(),atol=1e-6,rtol=0.)
    return model

def train(output):
    protocol=json.loads((output/'protocol.json').read_text());ck=protocol['checkpoint']
    dev=[TerrainScenario(**s) for s in protocol['development_cases']];labels=protocol['labels']
    # Only change the replay distribution; keep controller, observation, action,
    # reward, geometry, PPO hyperparameters and optimizer from the source policy.
    train_seed=protocol.get('train_seed',620000);train_cases=[];rng=np.random.default_rng(train_seed+1)
    for i in range(1024):
        seed=train_seed+i;s=sample_terrain_v4(seed,'train')
        if i<614:
            s=replace(s,terrain='step',height_l=0.,height_r=0.,grade_deg=0.,roughness_m=0.,step_height_m=float(rng.uniform(.018,.027)),transition_run_m=0.,lateral_margin_m=0.,speed=float(rng.choice((-1,1))*rng.uniform(.5,.65 if i%2==0 else .95)))
        elif i<921:s=TerrainScenario(**asdict(sample_scenario('train',seed,3)),terrain_seed=seed)
        train_cases.append(s)
    directory=output/'round_001';directory.mkdir(exist_ok=False)
    budget_total=protocol['training_budget_steps'];mode=protocol.get('residual_mode','diff3')
    write(directory/'run_config.json',dict(training_cases=[asdict(s) for s in train_cases],checkpoint=ck,budget=budget_total,source_checkpoint_used=True,residual_mode=mode,optimizer_restored=mode=='diff3'))
    raw=LiveNativeEnv(directory/'live',n=1024,scenario=train_cases,bank_factory=bank,phase='terrain_targeted_probe',residual_mode=mode)
    env=VecNormalize.load(ck+'.pkl',raw);env.training=True;env.norm_reward=False
    if mode=='virtual6':env.action_space=raw.action_space
    model=transferred_virtual6(PPO.load(ck+'.zip',device='cpu'),env) if mode=='virtual6' else TimedPPO.load(ck+'.zip',env=env,device='cpu')
    model.timings=[];model.set_random_seed(train_seed+1);start=model.num_timesteps
    try:
        for budget in (budget_total//2,budget_total):
            write(output/'status.json',dict(status='training',round=1,new_steps=model.num_timesteps-start))
            model.learn(total_timesteps=start+budget-model.num_timesteps,reset_num_timesteps=False)
            assert model.num_timesteps==start+budget and all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
            prefix=directory/f'step_{budget}';model.save(prefix);env.save(str(prefix)+'.pkl')
            write(output/'status.json',dict(status='evaluating',round=1,new_steps=budget))
            result_rows=evaluate_terrain(model,str(prefix)+'.pkl',dev,dict(residual_mode=mode))
            result=dict(summary=summary(result_rows,labels),runs=result_rows,checkpoint=str(prefix.resolve()),additional_steps=budget,updates=len(model.timings))
            write(output/f'targeted_{budget}.json',result);print('targeted',budget,result['summary'],flush=True)
        write(output/'status.json',dict(status='completed',round=1,new_steps=budget_total,stop_reason='round_budget',promoted=False))
    except Exception as exc:
        write(output/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--train',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    if a.train:train(a.output);sys.exit(0)
    a.output.mkdir(parents=True,exist_ok=False)
    ck=Path(json.loads((ROOT/'wheelleg_warp/results/terrain_v3_1024_20260922/selection.json').read_text())['best']['path'])
    dev,labels=cases(600000)
    masks={'baseline':[1,1,1],'no_leg_force':[0,1,1],'no_leg_moment':[1,0,1],'no_wheel_yaw':[1,1,0],'zero_residual':[0,0,0]}
    write(a.output/'protocol.json',dict(checkpoint=str(ck),checkpoint_sha256=hashlib.sha256(Path(str(ck)+'.zip').read_bytes()).hexdigest(),development_cases=[asdict(s) for s in dev],labels=labels,masks=masks,holdout_start=610000,holdout_evaluated=False,selection='step success gain with no legacy/surface/advanced success regression; repeat candidate before holdout',training_budget_steps=512000))
    model=PPO.load(str(ck)+'.zip',device='cpu')
    for name,mask in masks.items():
        start=time.perf_counter();r=evaluate(MaskedPolicy(model,mask),str(ck)+'.pkl',dev,labels);r.update(mask=mask,seconds=time.perf_counter()-start)
        write(a.output/(name+'.json'),r);print(name,r['summary'],flush=True)
