"""Prospective matched CUDA learner: engineering admission before main training."""
from pathlib import Path
import json,hashlib,gc,pickle,time
import numpy as np
import torch
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
import train_height_comparison as base_env
from route_pilot_env import RouteState,LegOnly
from route_pilot import RouteLedger
from native.controller import D,control_physical_nominal
from gpu_reference_budget import apply_budget,WIDTH
from fixed_residual_budget import apply_fixed
from reference_residual_budget import mix
from smoke_reward_training import weight_digest,check_agent
from train_height_comparison import equal
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reference_budget_learning_v1'


def proposal():return json.loads((OUT/'proposal.json').read_text())


def make_env(p,arm,seed,n,milestones):
    spec=p['arms'][arm];factory=base_env.raw_env;launch=wp.launch
    def wrapped(cases,mode):
        initial=np.zeros((len(cases),WIDTH));initial[:,2]=1.;stats=wp.array(initial,dtype=D);tracking=wp.array(np.ones(len(cases),np.int32))
        def patched(kernel,dim,inputs=None,**kwargs):
            result=launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
            if kernel is control_physical_nominal:
                if spec['allocation']=='constant':launch(apply_fixed,len(cases),[D(.9),inputs[5],inputs[15],inputs[14],tracking,stats])
                else:launch(apply_budget,len(cases),[int(spec['allocation']=='reference_room'),inputs[0],inputs[7],inputs[12],inputs[5],inputs[15],inputs[14],tracking,stats])
            return result
        wp.launch=patched
        try:raw=factory(cases,mode)
        finally:wp.launch=launch
        raw._learning_budget_buffers=(stats,tracking);gc.collect();assert raw._learning_budget_buffers[0] is stats
        wait=raw.step_wait;reset=raw.reset
        def reset_stats():stats.assign(initial)
        def reset_all():
            result=reset();reset_stats();return result
        def step_wait():
            result=wait();done=np.flatnonzero(result[2])
            if len(done):
                values=stats.numpy()
                for w in done:
                    result[3][w]['allocation_stats']=values[w].tolist();assert np.isfinite(values[w]).all();values[w]=initial[w]
                stats.assign(values)
            return result
        raw.reset=reset_all;raw.step_wait=step_wait;return raw
    base_env.raw_env=wrapped
    try:raw=base_env.CurriculumEnv(p,spec['native'],seed,n,milestones=milestones)
    finally:base_env.raw_env=factory;wp.launch=launch
    route=RouteState(raw,'route');action=LegOnly(route) if spec['dimensions']==2 else route
    env=VecNormalize(VecCheckNan(action,raise_exception=True),**p['normalization']);return raw,route,env


def agent(p,arm,seed,env):
    result=base_env.new_agent(p,p['arms'][arm]['initial'],seed,env)
    with torch.no_grad():
        result.policy.mlp_extractor.policy_net[0].weight[:,38]=0
        result.policy.mlp_extractor.value_net[0].weight[:,38]=0
    assert result.observation_space.shape==(39,) and result.action_space.shape==(p['arms'][arm]['dimensions'],)
    return result


def initial_function_check(model):
    rng=np.random.default_rng(141001);x=torch.as_tensor(rng.normal(size=(128,39)).astype(np.float32),device=model.device);x[:,38]=0
    with torch.no_grad():
        mean=model.policy.get_distribution(x).distribution.mean;value=model.policy.predict_values(x)
        for y in [-3.,3.]:
            other=x.clone();other[:,38]=y
            assert torch.equal(mean,model.policy.get_distribution(other).distribution.mean) and torch.equal(value,model.policy.predict_values(other))
    return True


def fixed_unit():
    rng=np.random.default_rng(141002);n=128;nominal=rng.uniform(-4,4,(n,6));old=rng.uniform(-4,4,(n,6)).astype(np.float32);diag=np.zeros((n,38));diag[:,:6]=nominal;diag[:,12]=.6
    d=wp.array(diag,dtype=D);c=wp.array(old);on=wp.array(np.ones(n,np.int32));s=np.zeros((n,WIDTH));s[:,2]=1.;stats=wp.array(s,dtype=D)
    wp.launch(apply_fixed,n,[D(.9),on,d,c,on,stats]);out=c.numpy()
    for i in range(n):np.testing.assert_array_equal(out[i],mix(nominal[i].astype(np.float32),old[i],.9).astype(np.float32))
    np.testing.assert_array_equal(d.numpy()[:,:6],diag[:,:6]);np.testing.assert_array_equal(d.numpy()[:,10:],diag[:,10:]);np.testing.assert_array_equal(out[:,4:],old[:,4:]);assert np.all(stats.numpy()[:,6]==0)
    return dict(verified=True,rows=n,alpha=.9,physical_steps=0,nominal_wheel_lambda_identity=True)


def freeze_engineering():
    p=proposal();assert not (OUT/'engineering_contract.json').exists();torch.set_num_threads(1);unit=fixed_unit()
    parent=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reference_budget_v1/runtime_contract.json';sources=json.loads(parent.read_text())['source_sha256']
    sources={**sources,**{f'wheelleg_warp/{n}':sha(ROOT/'wheelleg_warp'/n) for n in ['reference_budget_train.py','fixed_residual_budget.py','route_pilot.py','route_pilot_env.py','reward_pilot_train.py','smoke_reward_training.py']}}
    assert all(sha(ROOT/n)==v for n,v in sources.items())
    atomic_json(OUT/'engineering_contract.json',dict(proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,total_policy_steps=48000,unit=unit,
        engineering_milestones=[4000,8000],checkpoint_every=2000,steps_per_arm=12000,environments=10,seed=1761,
        scope='Separate lifecycle engineering, no scientific evaluations/selection/warmstart. Matched initial L2network/world/optimizer/RMS, feature-column function test, GPU updates and exact checkpoint weight/optimizer/RMS reload; physics not restored.',training_steps_before_admission=0))
    print('FROZEN fourarm engineering48k and constantGPU unit',flush=True)


def contract():
    c=json.loads((OUT/'engineering_contract.json').read_text());assert c['proposal_sha256']==sha(OUT/'proposal.json') and all(sha(ROOT/n)==v for n,v in c['source_sha256'].items());return c


def run_one(arm):
    c=contract();p=proposal();torch.set_num_threads(1);seed=c['seed'];directory=OUT/'engineering'/arm;directory.mkdir(parents=True,exist_ok=False)
    raw,route,env=make_env(p,arm,seed,c['environments'],c['engineering_milestones']);model=agent(p,arm,seed,env);function=initial_function_check(model);fingerprint=weight_digest(model)
    world=hashlib.sha256()
    for a in [raw.venv.q0.numpy(),raw.venv.param.numpy(),raw.venv.k['gains'].numpy(),raw.venv.k['reference'].numpy()]:world.update(a.tobytes())
    for peer in ['L2-plain','L2-room','L2-constant']:
        path=OUT/'engineering'/peer/'initialization.json'
        if arm.startswith('L2') and path.exists():
            old=json.loads(path.read_text());assert old['weight_sha256']==fingerprint and old['world_sha256']==world.hexdigest()
    atomic_json(directory/'initialization.json',dict(arm=arm,seed=seed,weight_sha256=fingerprint,world_sha256=world.hexdigest(),feature_function_equal=function,observations=39,dimensions=p['arms'][arm]['dimensions'],engineering_only=True))
    cb=RouteLedger(raw,route,directory,c['checkpoint_every'],p['arms'][arm]['dimensions']==2)
    try:
        start=time.perf_counter();model.learn(total_timesteps=12000,callback=cb);cb.save();wall=time.perf_counter()-start
        assert model.num_timesteps==12000 and model._n_updates==240 and weight_digest(model)!=fingerprint;state=check_agent(model,480)
        assert [r['policy_steps'] for r in cb.saved]==list(range(2000,12001,2000)) and cb.rows and all(x['actual_episode_end'] for x in raw.transition_log) and np.any(raw.stages==3)
        assert np.isfinite(env.obs_rms.mean).all() and np.isfinite(env.obs_rms.var).all() and raw.venv.data.qpos.device.is_cuda
        for row in cb.rows:
            a=np.array(row['allocation_stats']);assert np.isfinite(a).all() and a[0]==row['physical_steps'] and a[6]==a[8]==0
        final=directory/'step_12000';loaded=PPO.load(str(final)+'.zip',device='cuda')
        equal(model.policy.state_dict(),loaded.policy.state_dict());equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict());check_agent(loaded,480)
        with open(str(final)+'.pkl','rb') as f:normal=pickle.load(f)
        equal(env.obs_rms.mean,normal.obs_rms.mean);equal(env.obs_rms.var,normal.obs_rms.var);equal(env.obs_rms.count,normal.obs_rms.count)
        atomic_json(directory/'episodes.json',dict(episodes=cb.rows,curriculum_transitions=raw.transition_log))
        atomic_json(directory/'verification.json',dict(verified=True,policy_steps=12000,ppo_epochs=240,adam_updates=480,adam=state,initial_weight_sha256=fingerprint,
            completed_episodes=len(cb.rows),curriculum_transitions=len(raw.transition_log),final_stage_counts={str(k):int(np.sum(raw.stages==k)) for k in [1,2,3]},checkpoint_steps=[r['policy_steps'] for r in cb.saved],
            weight_optimizer_RMS_reload_exact=True,physics_trajectory_restored=False,actual_cuda_model_value_adam_physics=True,continuous_learn_wall_s=wall,engineering_only=True,scientific_score_or_promotion=False))
        atomic_json(directory/'progress.json',dict(status='complete',sampled_steps=12000,trained_steps=12000));print('PASS ENGINEERING',arm,'12k/240epochs/480Adam',len(cb.rows),'episodes',flush=True)
    except BaseException as e:
        atomic_json(directory/'interruption.json',dict(error=str(e),sampled_steps=model.num_timesteps,trained_steps=cb.trained,silently_resumable=False));raise
    finally:env.close()


def engineering():
    contract();completed=[]
    for arm in proposal()['arms']:
        atomic_json(OUT/'engineering_progress.json',dict(status='running',completed_arms=completed,pending_arm=arm));run_one(arm);completed.append(arm)
    records={arm:json.loads((OUT/'engineering'/arm/'verification.json').read_text()) for arm in completed};assert len(records)==4 and all(r['verified'] for r in records.values())
    atomic_json(OUT/'engineering_verification.json',dict(verified=True,records=records,total_policy_steps=48000,engineering_contract_sha256=sha(OUT/'engineering_contract.json'),main_training_started=False))
    atomic_json(OUT/'engineering_progress.json',dict(status='complete',completed_arms=completed));print('PASS ALL fourarm48k engineering;main notstarted',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze_engineering','engineering']);globals()[p.parse_args().command]()
