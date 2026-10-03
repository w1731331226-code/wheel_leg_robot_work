"""Frozen height115 comparison on CUDA: initialize, smoke, train and one-use gate."""
from pathlib import Path
from dataclasses import asdict,fields,replace
import argparse,json,os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper,VecNormalize,VecCheckNan
from native.environment import NativeEnv,reset_rows
from native.terrain import HeightTerrainScenario,sample_height_terrain_115,V3_TERRAINS
from pretrain_yaw import b1_action,selection_key
from ppo_env import sample_scenario
import wheelleg_sim as sim
from score_yaw_gate import at_least
from terrain_eval import summarize_terrain,validate_terrain_rows
from training_contract import TASK_CONTRACT_VERSION,source_hashes,digest,checkpoint_hashes,verify_checkpoint
from dashboard.live_env import atomic_json

VERSION='height115-yaw-v3-v8-gpu'
METHODS={'M3':'diff3','B2-V':'virtual6','B2':'torque6'}
INIT=ROOT/'wheelleg_warp/results/twentyninth_round_capped_support_20261003/initial_action_config.json'
LEGACY=ROOT/'wheelleg_ppo/tools/results/fixes_2026-09-17/final/baseline.json'
B1_CONFIG=ROOT/'wheelleg_ppo/tools/results/yaw_precision_v1_2026-09-20/training_config.json'


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())
def stage_for(steps,milestones):return 1+sum(steps>=t for t in milestones)
def entry(seed,scenario):return dict(seed=seed,scenario=asdict(scenario))
def initial():
    data=read(INIT)
    for name,value in data['source_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError('Initial action audit source changed: '+name)
    return data


def freeze(out):
    out.mkdir(parents=True,exist_ok=False);init=initial();old=read(B1_CONFIG)
    p=dict(protocol=VERSION,baseline_version=init['baseline'],task_contract_version=TASK_CONTRACT_VERSION,
        methods=METHODS,training_seeds=[1609,1610,1611],formal_seeds=[1609,1610,1611,1612,1613],
        environments=100,policy_steps_per_seed=2000000,evaluation_interval=20000,maximum_selection_evaluations=100,
        physics_dt_s=.0005,actor_dt_s=.02,nominal_design_mass_kg=7.,
        curriculum_milestones=[20000,100000],curriculum_switch='requested at policy milestones; per-world model/target changes only after actual episode termination',
        training_bank_rule='100 fixed sampled worlds per stage and training seed; target fixed within episode; not IID per-episode parameter resampling',
        ppo={**old['ppo'],'n_steps':50,'batch_size':250},normalization=old['normalization'],device='cuda',
        initial_action_config=str(INIT.relative_to(ROOT)),initial_action_config_sha256=digest(INIT),
        residual_scale=1.,initial_log_std={m:init['methods'][mode]['initial_log_std'] for m,mode in METHODS.items()},
        b1_candidates=old['b1_candidates'],b1_formula=old['b1_formula'],
        selection_rule='exclude noncompleted; failures, meanJpsi, earliest policy step; stable first entry for ties',
        relative_improvement=.15,absolute_improvement_deg=.05,paired_direction='M3 lower than B2-V for each of the three same-number training seeds',
        nondegradation=dict(velocity_multiplier=1.05,velocity_add_m_s=.005,arrival_multiplier=1.05,arrival_add_s=.05,legacy_roll_pitch_add_deg=.1,
            preserve_all_B0_successes=True,success_count_at_least_best_reference=True,original28_all_success=True),
        gpu_replay='contact/float-order variation disclosed; deterministic actions do not imply bitwise trajectory reproducibility; no per-method rerun/cherry-picking',
        inference='whole action parameterization; calibrated aggregate RMS does not remove local, covariance, dimension or reachable-set differences',
        checkpoint_resume='weights/Adam/normalization and consumed policy steps; restart physical episodes, explicitly segmented rather than exact trajectory resume')
    p['training_banks']={}
    for seed in p['formal_seeds']:
        banks={}
        for stage in (1,2,3):
            rows=[]
            for i in range(100):
                number=2800000+(seed-1609)*1000+i;s=sample_height_terrain_115(number,stage,'train')
                if stage<3:s=replace(s,mass=7.,delay_ms=0.)
                if stage<2:s=replace(s,mu_l=.8,mu_r=.8,drive_difference=0.)
                rows.append(entry(number,s))
            banks[str(stage)]=rows
        p['training_banks'][str(seed)]=banks
    p['selection']=[entry(i,sample_height_terrain_115(i,3,'development')) for i in range(2900000,2900032)]
    p['gate']=[entry(i,sample_height_terrain_115(i,3,'development')) for i in range(2901000,2901064)]
    baseline=read(LEGACY);assert baseline['total']==baseline['success_count']==28
    p['original_regression']=baseline['runs']
    p['regression']=[dict(seed=r['name'],scenario=asdict(HeightTerrainScenario(**r['scenario'],stand_height_m=.3))) for r in baseline['runs']]
    finals={terrain:[] for terrain in V3_TERRAINS};number=3000000
    while any(len(a)<200 for a in finals.values()):
        s=sample_height_terrain_115(number,3,'development')
        if len(finals[s.terrain])<200:finals[s.terrain].append(entry(number,s))
        number+=1
    finals['reserved_combination']=[]
    for i in range(3200000,3200200):
        base=sample_scenario('test_combination',i,3);height=sample_height_terrain_115(i,3,'development').stand_height_m
        finals['reserved_combination'].append(entry(i,HeightTerrainScenario(**asdict(base),stand_height_m=height,terrain_seed=i)))
    for factor in ('height','mass','friction','drive','delay'):finals['pressure_'+factor]=[]
    for j,factor in enumerate(('height','mass','friction','drive','delay')):
        for i in range(3100000+j*1000,3100200+j*1000):
            s=sample_height_terrain_115(i,3,'development');rng=np.random.default_rng(i)
            if factor=='height':s=replace(s,terrain='legacy',height_l=float(rng.choice([.025,.03])),height_r=0.,grade_deg=0.,roughness_m=0.,step_height_m=0.,relative_attitude=False)
            elif factor=='mass':s=replace(s,mass=float(rng.uniform(7.6,8.)))
            elif factor=='friction':s=replace(s,mu_l=.4,mu_r=1.2)
            elif factor=='drive':s=replace(s,drive_difference=float(rng.choice([-.05,.05])))
            else:s=replace(s,delay_ms=float(rng.choice([15.,20.])))
            finals['pressure_'+factor].append(entry(i,s))
    write(out/'sealed_final_cases.json',dict(role='parameters only; never simulated or scored before configuration/gate lock',cases=finals,old720000_namespace_untouched=True))
    p['sealed_final_sha256']=digest(out/'sealed_final_cases.json')
    p['source_sha256']={**source_hashes(__file__),str(INIT.relative_to(ROOT)):digest(INIT),str(LEGACY.relative_to(ROOT)):digest(LEGACY),
        str(B1_CONFIG.relative_to(ROOT)):digest(B1_CONFIG),'wheelleg_ppo/tools/score_yaw_gate.py':digest(ROOT/'wheelleg_ppo/tools/score_yaw_gate.py')}
    write(out/'protocol.json',p);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    print('FROZEN',VERSION,'selection32/gate64/legacy28/final categories200 each; no simulation',flush=True)


def protocol(out):
    out=Path(out);p=read(out/'protocol.json')
    if p['protocol']!=VERSION or digest(out/'protocol.json')!=(out/'protocol.sha256').read_text().strip():raise ValueError('Protocol changed')
    for name,value in p['source_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError('Frozen source/input changed: '+name)
    if digest(out/'sealed_final_cases.json')!=p['sealed_final_sha256']:raise ValueError('Sealed final parameters changed')
    if p['device']!='cuda' or not torch.cuda.is_available():raise ValueError('Formal PPO requires available CUDA')
    return p


def raw_env(cases,mode):
    env=NativeEnv.height115_candidate(n=len(cases),scenario=[HeightTerrainScenario(**r['scenario']) for r in cases],residual_mode=mode,shared_reference=True)
    if env.baseline_version!=initial()['baseline'] or env.observation_space.shape!=(38,):raise ValueError('Wrong runtime baseline/observation')
    return env


class CurriculumEnv(VecEnvWrapper):
    """Keep captured GPU buffers; change each world only at its real episode reset."""
    def __init__(self,p,mode,seed,n,start=0,milestones=None):
        self.banks={int(s):rows[:n] for s,rows in p['training_banks'][str(seed)].items()}
        self.milestones=milestones or p['curriculum_milestones'];self.policy_steps=start
        stage=stage_for(start,self.milestones);raw=raw_env(self.banks[stage],mode)
        self.stages=np.full(n,stage);self.cache={};self.transition_rows=0;self.transition_log=[]
        super().__init__(raw)
        for s,rows in self.banks.items():
            temp=raw if s==stage else raw_env(rows,mode)
            try:
                names=[f.name for f in fields(mjw.Model) if getattr(f.type,'shape',())[:1]==('*',)]
                np.testing.assert_array_equal(raw.physical_args[7].numpy(),temp.physical_args[7].numpy())
                np.testing.assert_array_equal(raw.physical_args[8].numpy(),temp.physical_args[8].numpy())
                for name in ('gains','feed','angles'):np.testing.assert_array_equal(raw.k[name].numpy(),temp.k[name].numpy())
                self.cache[s]=dict(model={name:getattr(temp.model,name).numpy() for name in names},meaninertia=temp.model.stat.meaninertia.numpy(),
                    q0=temp.q0.numpy(),param=temp.param.numpy(),obs0=temp.obs0.numpy(),reference=temp.k['reference'].numpy(),
                    geom_xpos=temp.data.geom_xpos.numpy(),geom_xmat=temp.data.geom_xmat.numpy(),
                    info={name:list(getattr(temp,name)) for name in ('required_contact_masks','required_terrain_contact_masks','required_terrain_end','relative_attitude','task_goals')})
            finally:
                if temp is not raw:temp.close()
    def reset(self):return self.venv.reset()
    def step_wait(self):
        self.policy_steps+=self.num_envs;desired=stage_for(self.policy_steps,self.milestones)
        obs,reward,done,infos=self.venv.step_wait();raw=self.venv
        for w in np.flatnonzero(done):
            row=self.banks[int(self.stages[w])][w];infos[w].update(curriculum_stage=int(self.stages[w]),seed=row['seed'],scenario=row['scenario'])
        selected=np.flatnonzero(done&(self.stages!=desired))
        if len(selected):
            c=self.cache[desired]
            for name,value in c['model'].items():
                buffer=getattr(raw.model,name);current=buffer.numpy();current[selected]=value[selected];buffer.assign(current)
            for buffer,value in [(raw.model.stat.meaninertia,c['meaninertia']),(raw.q0,c['q0']),(raw.param,c['param']),(raw.obs0,c['obs0']),
                (raw.k['reference'],c['reference']),(raw.data.geom_xpos,c['geom_xpos']),(raw.data.geom_xmat,c['geom_xmat'])]:
                current=buffer.numpy();current[selected]=value[selected];buffer.assign(current)
            for w in selected:
                raw.scenarios[w]=HeightTerrainScenario(**self.banks[desired][w]['scenario']);raw.stand_heights[w]=raw.scenarios[w].stand_height_m
                for name,values in c['info'].items():getattr(raw,name)[w]=values[w]
            mask=np.zeros(self.num_envs,np.int32);mask[selected]=1;raw.mask.assign(mask);wp.launch(reset_rows,self.num_envs,raw.reset_args);mjw.forward(raw.model,raw.data)
            obs[selected]=raw.obs.numpy()[selected];self.stages[selected]=desired;self.transition_rows+=len(selected)
            self.transition_log.append(dict(policy_steps=self.policy_steps,stage=desired,worlds=selected.tolist(),actual_episode_end=True))
        return obs,reward,done,infos


def new_agent(p,method,seed,env):
    kwargs=dict(p['ppo']);policy_kwargs=dict(kwargs.pop('policy_kwargs'));policy_kwargs['net_arch']=[64,64];policy_kwargs['log_std_init']=0.
    agent=PPO('MlpPolicy',env,seed=seed,device=p['device'],policy_kwargs=policy_kwargs,**kwargs)
    value=torch.as_tensor(p['initial_log_std'][method],dtype=agent.policy.log_std.dtype,device=agent.policy.log_std.device)
    if value.shape!=agent.policy.log_std.shape or not torch.isfinite(value).all():raise ValueError('Invalid initial log_std')
    with torch.no_grad():agent.policy.log_std.copy_(value)
    return agent


def evaluate(cases,mode,agent=None,normalization=None,candidate=None):
    rows=[]
    # The original solver50 case is a separate bank; never silently change its option.
    for iterations in sorted({r['scenario']['solver_iterations'] for r in cases}):
        group=[r for r in cases if r['scenario']['solver_iterations']==iterations];raw=raw_env(group,mode);env=raw
        if agent is not None:
            env=VecNormalize.load(str(normalization),VecCheckNan(raw,raise_exception=True));env.training=False;env.norm_reward=False
            stats=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count)
        try:
            obs=env.reset();found=[None]*len(group);deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
            for _ in range(deadline):
                actions=agent.predict(obs,deterministic=True)[0] if agent is not None else np.stack([b1_action(o,candidate) for o in obs])
                obs,_,done,infos=env.step(actions)
                stopped=raw.stopped_q.numpy() if done.any() else None
                for w in np.flatnonzero(done):
                    if found[w] is None:
                        mean_fk=float(np.mean([sim.fk_joints(float(stopped[w,raw.ids.numpy()[2*s]]),float(stopped[w,raw.ids.numpy()[2*s+1]]))['leg_len'] for s in range(2)]))
                        found[w]=dict(**group[w],**{k:v for k,v in infos[w].items() if k!='terminal_observation'},final_mean_fk_leg_m=mean_fk)
                if all(r is not None for r in found):break
            if any(r is None for r in found):raise RuntimeError('Incomplete evaluator collection')
            validate_terrain_rows(found)
            if agent is not None:
                np.testing.assert_array_equal(stats[0],env.obs_rms.mean);np.testing.assert_array_equal(stats[1],env.obs_rms.var);assert stats[2]==env.obs_rms.count
            rows+=found
        finally:env.close()
    by_id={r['seed']:r for r in rows};return [by_id[r['seed']] for r in cases]


def summary(rows):
    for r in rows:
        if type(r['success']) is not bool:raise ValueError('Invalid success flag')
        if r['success']:
            attitude=r['relative_peak_deg'] if r['scenario'].get('relative_attitude') else r['peak_deg']
            if not (r['physical_safety_passed'] and r['design_joint_passed'] and max(attitude)<=5 and r['peak_deg'][2]<=5
                and (not r['scenario'].get('relative_attitude') or max(r['peak_deg'][:2])<=10)
                and r['velocity_rmse']<=.2*abs(r['scenario']['speed']) and r['stop_distance_m']<=.6 and r['tail_speed_m_s']<=.03
                and r['height_rmse_m']<=.02 and abs(r['final_mean_fk_leg_m']-r['target_leg_m'])<=.02):raise ValueError('Success contradicts frozen constraints')
    return summarize_terrain(rows,[r['seed'] for r in rows])
def classical(out):
    p=protocol(out);dest=out/'classical_selection';dest.mkdir();results=[]
    for candidate in p['b1_candidates']:
        runs=evaluate(p['selection'],'diff3',candidate=candidate);record=dict(candidate=candidate,summary=summary(runs),runs=runs)
        write(dest/(candidate['name']+'.json'),record);results.append(record);print('CLASSICAL',candidate['name'],record['summary'],flush=True)
    complete=[r for r in results if r['summary']['complete']]
    best=min(complete,key=lambda r:selection_key(r['summary'])) if complete else None
    b0=results[0];headroom=min(b0['summary']['mean_yaw_score_deg'],best['summary']['mean_yaw_score_deg']) if best and b0['summary']['complete'] else None
    write(out/'classical_selection.json',dict(protocol_sha256=digest(out/'protocol.json'),selected=best,
        classical_headroom_deg=headroom,absolute_headroom_possible=headroom is not None and headroom>=p['absolute_improvement_deg'],gate_opened=False))


class Episodes(BaseCallback):
    def __init__(self,progress=None):super().__init__();self.rows=[];self.progress=progress
    def _on_step(self):
        if self.progress is not None:atomic_json(self.progress,dict(executed_policy_steps=self.num_timesteps))
        for info in self.locals['infos']:
            if 'episode' in info:self.rows.append({k:v for k,v in info.items() if k!='terminal_observation'})
        return True

def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def smoke(out,method):
    p=protocol(out);directory=out/'engineering'/method;directory.mkdir(parents=True);seed=1609;mode=METHODS[method]
    # Ten worlds retain the formal50-step/batch250/ten-epoch settings; no probe selection or promotion.
    raw=CurriculumEnv(p,mode,seed,10,milestones=[4000,8000]);env=VecNormalize(VecCheckNan(raw,raise_exception=True),**p['normalization'])
    agent=new_agent(p,method,seed,env);initial_std=agent.policy.log_std.detach().cpu().numpy().copy();before={k:v.clone() for k,v in agent.policy.state_dict().items()};cb=Episodes();restored=None
    try:
        write(directory/'configuration.json',dict(protocol_sha256=digest(out/'protocol.json'),mode=mode,seed=seed,n=10,
            curriculum_milestones=[4000,8000],policy_steps=12000,ppo=p['ppo'],initial_log_std=initial_std.tolist(),weights_promoted=False))
        agent.learn(total_timesteps=6000,callback=cb)
        assert agent.num_timesteps==6000 and agent._n_updates==120 and cb.rows
        assert any(not torch.equal(before[k],v) for k,v in agent.policy.state_dict().items()) and agent.policy.optimizer.state_dict()['state']
        prefix=directory/'probe';agent.save(prefix);env.save(str(prefix)+'.pkl')
        new_raw=CurriculumEnv(p,mode,seed,10,start=6000,milestones=[4000,8000]);restored=VecNormalize.load(str(prefix)+'.pkl',VecCheckNan(new_raw,raise_exception=True))
        loaded=PPO.load(str(prefix)+'.zip',env=restored,device=p['device'])
        equal(agent.policy.state_dict(),loaded.policy.state_dict());equal(agent.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict())
        equal(env.obs_rms.mean,restored.obs_rms.mean);equal(env.obs_rms.var,restored.obs_rms.var);equal(env.obs_rms.count,restored.obs_rms.count)
        cb2=Episodes();loaded.learn(total_timesteps=6000,reset_num_timesteps=False,callback=cb2)
        assert loaded.num_timesteps==12000 and loaded._n_updates==240 and np.any(new_raw.stages==3)
        assert agent.device.type==loaded.device.type=='cuda'
        assert all(v.device.type=='cuda' for v in loaded.policy.parameters())
        optimizer_devices={v.device.type for state in loaded.policy.optimizer.state.values() for k,v in state.items() if isinstance(v,torch.Tensor) and k!='step'}
        assert optimizer_devices=={'cuda'}
        resumed=directory/'resumed_probe';loaded.save(resumed);restored.save(str(resumed)+'.pkl')
        from probe_height_115_margin import cases
        from native.terrain import sample_height_terrain_115
        scenes=[cases()[4],replace(cases()[5],mass=7.5,mu_l=.6,mu_r=1.,drive_difference=.03,delay_ms=10.),replace(cases()[4],stand_height_m=.38),sample_height_terrain_115(1154043,3,'development')]
        evaluation=evaluate([entry(1155000+i,s) for i,s in enumerate(scenes)],mode,loaded,str(resumed)+'.pkl')
        write(directory/'verification.json',dict(passed=True,policy_steps=12000,updates=240,weights_changed=True,
            weights_optimizer_normalization_restored_exactly=True,initialization_only_on_fresh_policy=True,
            physical_environment_state_restored=False,episodes=cb.rows+cb2.rows,evaluation=evaluation,
            ppo_device=str(loaded.device),policy_parameter_devices=sorted({v.device.type for v in loaded.policy.parameters()}),
            optimizer_moment_devices=sorted(optimizer_devices),physics_device=str(new_raw.venv.data.qpos.device),
            curriculum_transitions=raw.transition_log+new_raw.transition_log,stages_after_resume=new_raw.stages.tolist(),
            before_restore_checkpoint=checkpoint_hashes(prefix),final_checkpoint=checkpoint_hashes(resumed),source_sha256=p['source_sha256'],formal_training=False,checkpoint_promoted=False))
        print('PASS REAL ENTRY SMOKE',method,'12000steps240epochs; exact Adam/RMS restore; stage3 after actual ends; eval',sum(r['success'] for r in evaluation),'/4',flush=True)
    finally:
        env.close()
        if restored is not None:restored.close()


def learn_exact(agent,target,callback=None):
    """Only a resumed partial rollout needs a shorter final buffer to hit the cap."""
    if target<agent.num_timesteps or (target-agent.num_timesteps)%agent.n_envs:raise ValueError('Unreachable exact policy budget')
    while agent.num_timesteps<target:
        steps=min(50,(target-agent.num_timesteps)//agent.n_envs)
        if agent.n_steps!=steps:
            agent.n_steps=steps
            agent.rollout_buffer=agent.rollout_buffer_class(steps,agent.observation_space,agent.action_space,device=agent.device,
                gamma=agent.gamma,gae_lambda=agent.gae_lambda,n_envs=agent.n_envs,**agent.rollout_buffer_kwargs)
        chunk=min(target-agent.num_timesteps,steps*agent.n_envs)
        agent.learn(total_timesteps=chunk,reset_num_timesteps=False,callback=callback)
    assert agent.num_timesteps==target


def train(out,method,seed,resume=False):
    p=protocol(out);admission=read(out/'readiness.json')
    if not admission['passed'] or admission['protocol_sha256']!=digest(out/'protocol.json'):raise ValueError('Current protocol not admitted')
    if seed not in p['formal_seeds']:raise ValueError('Unknown training seed')
    directory=out/'runs'/method/str(seed);mode=METHODS[method];records=[];start=0;last=None
    if resume:
        last=read(directory/'last_checkpoint.json')
        if last['protocol_sha256']!=digest(out/'protocol.json') or not last['resumable']:raise ValueError('Checkpoint not resumable under current protocol')
        verify_checkpoint(last['path'],last)
        start=last['consumed_policy_steps']
        if read(directory/'progress.json')['executed_policy_steps']!=start:raise ValueError('Unsaved executed steps; do not replay them inside frozen budget')
        records=[read(path) for path in sorted(directory.glob('step_*.json'),key=lambda x:int(x.stem.split('_')[-1]))]
    else:directory.mkdir(parents=True)
    raw=CurriculumEnv(p,mode,seed,p['environments'],start=start)
    if resume:
        env=VecNormalize.load(last['path']+'.pkl',VecCheckNan(raw,raise_exception=True));env.training=True;env.norm_reward=False
        agent=PPO.load(last['path']+'.zip',env=env,device=p['device'])
        if agent.num_timesteps!=start:raise ValueError('Checkpoint consumed budget mismatch')
    else:env=VecNormalize(VecCheckNan(raw,raise_exception=True),**p['normalization']);agent=new_agent(p,method,seed,env)
    if agent.device.type!='cuda' or any(v.device.type!='cuda' for v in agent.policy.parameters()):raise ValueError('PPO policy/value must remain on CUDA')
    progress=directory/'progress.json';atomic_json(progress,dict(executed_policy_steps=start));callback=Episodes(progress)
    def save(prefix,resumable=True):
        agent.save(prefix);env.save(str(prefix)+'.pkl')
        atomic_json(directory/'last_checkpoint.json',dict(protocol_sha256=digest(out/'protocol.json'),path=str(prefix),
            consumed_policy_steps=agent.num_timesteps,resumable=resumable,physical_trajectory_restart_on_resume=True,**checkpoint_hashes(prefix)))
    try:
        for steps in range(p['evaluation_interval'],p['policy_steps_per_seed']+1,p['evaluation_interval']):
            if steps<start or any(r['policy_steps']==steps for r in records):continue
            protocol(out);learn_exact(agent,steps,callback)
            assert agent.num_timesteps==steps
            prefix=directory/f'step_{steps}';save(prefix)
            runs=evaluate(p['selection'],mode,agent,str(prefix)+'.pkl');record=dict(policy_steps=steps,path=str(prefix),summary=summary(runs),runs=runs,**checkpoint_hashes(prefix))
            write(directory/(prefix.name+'.json'),record);records.append(record)
            eligible=[r for r in records if r['summary']['complete']]
            write(directory/'selection.json',dict(protocol_sha256=digest(out/'protocol.json'),consumed_policy_steps=steps,
                best=min(eligible,key=lambda r:selection_key(r['summary'],r['policy_steps'])) if eligible else None,
                evaluation_count=len(records),physical_trajectory_restart_on_resume=resume))
            print('TRAIN',method,seed,steps,record['summary'],flush=True)
    except BaseException:
        # Charge every attempted physical sampling step, including an environment
        # error before PPO receives its transition; never replay a partial buffer.
        consumed=max(agent.num_timesteps,raw.policy_steps)
        agent.num_timesteps=consumed;atomic_json(progress,dict(executed_policy_steps=consumed))
        finite=all(torch.isfinite(v).all() for v in agent.policy.state_dict().values()) and np.isfinite(env.obs_rms.mean).all() and np.isfinite(env.obs_rms.var).all()
        save(directory/f'interrupted_{consumed}',resumable=bool(finite and consumed<=p['policy_steps_per_seed']))
        raise
    finally:env.close()


def lock_gate(out):
    p=protocol(out);classical=read(out/'classical_selection.json')
    if classical['protocol_sha256']!=digest(out/'protocol.json') or not classical['selected']:raise ValueError('No current complete B1')
    locked={}
    for method in METHODS:
        locked[method]={}
        for seed in p['training_seeds']:
            r=read(out/'runs'/method/str(seed)/'selection.json')
            if r['protocol_sha256']!=digest(out/'protocol.json') or r['consumed_policy_steps']!=p['policy_steps_per_seed'] or not r['best']:raise ValueError('Incomplete pilot/selection')
            verify_checkpoint(r['best']['path'],r['best']);locked[method][str(seed)]=r['best']
    with (out/'gate_lock.json').open('x') as f:json.dump(dict(protocol_sha256=digest(out/'protocol.json'),b1=classical['selected']['candidate'],checkpoints=locked),f,indent=2)


def assess(p,gate,regression):
    failures=[];scores={};seeds=list(map(str,p['training_seeds']))
    if set(gate)!=set(METHODS)|{'B0','B1'} or set(regression)!=set(seeds):raise ValueError('Missing/extra methods or regression seeds')
    for method,runs in gate.items():
        if set(runs)!=({'fixed'} if method in ('B0','B1') else set(seeds)):raise ValueError('Missing/extra gate seeds')
        scores[method]={}
        for seed,rows in runs.items():
            if [r['seed'] for r in rows]!=[r['seed'] for r in p['gate']] or [r['scenario'] for r in rows]!=[r['scenario'] for r in p['gate']]:raise ValueError('Gate cases changed')
            s=summary(rows);scores[method][seed]=s['mean_yaw_score_deg']
            if not s['complete']:failures.append('incomplete: '+method+'/'+seed)
    for seed,rows in regression.items():
        if [r['seed'] for r in rows]!=[r['seed'] for r in p['regression']]:raise ValueError('Regression identity changed')
        for row,old in zip(rows,p['original_regression']):
            if row['reason']!='completed':failures.append('incomplete legacy: '+seed+'/'+str(row['seed']));continue
            if not row['success']:failures.append('legacy failure: '+seed+'/'+str(row['seed']))
            if not at_least(old['velocity_rmse']*1.05+.005,row['velocity_rmse']):failures.append('legacy velocity: '+seed+'/'+str(row['seed']))
            if any(not at_least(old['peak_deg'][j]+.1,row['peak_deg'][j]) for j in (0,1)):failures.append('legacy attitude: '+seed+'/'+str(row['seed']))
        if [r['scenario'] for r in rows]!=[r['scenario'] for r in p['regression']]:raise ValueError('Regression scenarios changed')
    if any(value is None for runs in scores.values() for value in runs.values()):return dict(passed=False,failures=failures,scores=scores)
    q={m:float(np.mean(list(r.values()))) for m,r in scores.items()};qref=min(q[m] for m in ('B0','B1','B2','B2-V'));delta=qref-q['M3']
    if qref==0 or not at_least(delta/qref,.15) or not at_least(delta,.05):failures.append('effect below15% or0.05deg')
    for seed in seeds:
        candidate=gate['M3'][seed];base=gate['B0']['fixed']
        if not scores['M3'][seed]<scores['B2-V'][seed]:failures.append('paired direction: '+seed)
        if sum(r['success'] for r in candidate)<max(sum(r['success'] for r in gate[m]['fixed' if m in ('B0','B1') else seed]) for m in ('B0','B1','B2','B2-V')):failures.append('success count: '+seed)
        for row,old in zip(candidate,base):
            if old['success'] and not row['success']:failures.append('lost B0: '+seed+'/'+str(row['seed']))
            if not at_least(old['velocity_rmse']*1.05+.005,row['velocity_rmse']):failures.append('velocity: '+seed+'/'+str(row['seed']))
            if not at_least(old['arrival_s']*1.05+.05,row['arrival_s']):failures.append('arrival: '+seed+'/'+str(row['seed']))
    return dict(passed=not failures,failures=failures,scores=scores,method_scores=q,absolute_improvement_deg=delta,relative_improvement=delta/qref if qref else None)


def gate(out):
    p=protocol(out);locked=read(out/'gate_lock.json')
    if locked['protocol_sha256']!=digest(out/'protocol.json'):raise ValueError('Gate lock/protocol mismatch')
    for runs in locked['checkpoints'].values():
        for record in runs.values():verify_checkpoint(record['path'],record)
    with (out/'gate_opened.json').open('x') as f:json.dump(dict(protocol_sha256=digest(out/'protocol.json'),single_use=True),f)
    results={};regression={}
    for method,candidate in [('B0',p['b1_candidates'][0]),('B1',locked['b1'])]:results[method]={'fixed':evaluate(p['gate'],'diff3',candidate=candidate)};write(out/('gate_'+method+'.json'),results[method])
    for method,mode in METHODS.items():
        results[method]={}
        for seed,record in locked['checkpoints'][method].items():
            agent=PPO.load(record['path']+'.zip',device=p['device']);results[method][seed]=evaluate(p['gate'],mode,agent,record['path']+'.pkl')
            write(out/f'gate_{method}_{seed}.json',results[method][seed])
            if method=='M3':regression[seed]=evaluate(p['regression'],mode,agent,record['path']+'.pkl');write(out/f'gate_legacy_{seed}.json',regression[seed])
    write(out/'gate_result.json',assess(p,results,regression))


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','classical','smoke','train','lock-gate','gate'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--method',choices=METHODS);parser.add_argument('--seed',type=int,default=1609);parser.add_argument('--resume',action='store_true')
    a=parser.parse_args();out=a.output.resolve()
    if a.command=='freeze':freeze(out)
    elif a.command=='classical':classical(out)
    elif a.command=='smoke':smoke(out,a.method)
    elif a.command=='train':train(out,a.method,a.seed,a.resume)
    elif a.command=='lock-gate':lock_gate(out)
    else:gate(out)
