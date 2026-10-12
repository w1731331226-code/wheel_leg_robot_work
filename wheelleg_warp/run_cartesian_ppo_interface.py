"""Single preregistered matched PPO collection queue, with optimizer updates forbidden."""
import copy
import fcntl
import inspect
import json
import os
import random
import shutil
import subprocess
import time
from unittest.mock import patch
import numpy as np
import torch
import mujoco
import warp as wp
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper,VecNormalize,VecCheckNan
from stable_baselines3.common import base_class,on_policy_algorithm,buffers,policies
from stable_baselines3.common.vec_env import vec_normalize
import run_cartesian_interface_v2 as parent
import cartesian_ppo_recorder as recorder
from fixed_reference_force import ForceActions,embed
from reference_learning_engineering import agent
from train_height_comparison import equal
from dashboard.live_env import atomic_json

ROOT,BASE,sha=parent.ROOT,parent.BASE,parent.sha
OUT=BASE/'cartesian_ppo_interface_v1';QUEUE=OUT/'runtime'


def runtime():
    value=parent.runtime()
    value['PPO_source_sha256']={inspect.getfile(m):sha(inspect.getfile(m)) for m in (base_class,on_policy_algorithm,buffers,policies,vec_normalize)}
    return value


def inputs():
    old=parent.inputs();p=json.loads((OUT/'proposal.json').read_text())
    assert p['arms']==old['arms'] and p['worlds']==10 and p['training_banks']==old['training_banks']
    assert p['bank_seed']==p['engineering_seed']==33531 and p['curriculum_milestones']==[4000,8000]
    assert p['rollout_packets_each']==1200 and p['ppo_rollout_chunks_each']==24 and p['ppo']['n_steps']==50
    assert p['actual_graph_world_step_budget']==960800 and p['constructor_FD_budget']==80
    assert p['policy_forward_row_budget']==24000 and p['value_only_forward_row_budget']==24480 and p['value_only_forward_call_budget']==24048
    assert p['optimizer_updates']==0 and p['latent_standard_deviation']==.25 and p['reference_log_std']==np.log(.25)
    assert not p['runtime_admitted'] and not p['scientific_training_admitted'] and not p['formal5_admitted']
    for mapping in (p['parent_sha256'],p['old_failures_retained']):assert all(sha(ROOT/n)==h for n,h in mapping.items())
    return p


def verify(committed=False):
    p=inputs();a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json') and a['runtime']==runtime()
    assert all(sha(ROOT/n)==h for n,h in a['source_sha256'].items())
    if committed:
        expected={str((OUT/'source_admission.json').relative_to(ROOT)):sha(OUT/'source_admission.json'),**a['source_sha256']}
        names=list(expected);r=subprocess.run(['git','cat-file','--batch'],input=''.join('HEAD:'+n+'\n' for n in names).encode(),cwd=ROOT,capture_output=True,check=True)
        assert parent.blob_hashes(r.stdout,names)==expected
    assert shutil.disk_usage(ROOT).free>=2*1024**3
    return p,a


def freeze():
    inputs();assert not QUEUE.exists() and not (OUT/'source_admission.json').exists()
    from test_cartesian_ppo_interface import run as check
    check()
    sources=json.loads((parent.OUT/'source_admission.json').read_text())['source_sha256'].copy()
    for n in ('run_cartesian_ppo_interface.py','test_cartesian_ppo_interface.py','cartesian_ppo_recorder.py','test_cartesian_ppo_recorder.py',
              'review_cartesian_light_history.py','review_cartesian_light_map.py','reference_learning_engineering.py','fixed_reference_force.py'):
        sources['wheelleg_warp/'+n]=sha(ROOT/'wheelleg_warp'/n)
    atomic_json(OUT/'source_admission.json',dict(round=376,verified=True,source_sha256=sources,runtime=runtime(),proposal_sha256=sha(OUT/'proposal.json'),
        immutable_original_graph_worlds_1201_and_network_pre_call_caps=True,committed_manifest_and_all_source_blobs_required=True,
        actual_terminal_post_private_bank_and_nominal_slot_recording=True,optimizer_updates_forbidden=True,new_physics_steps=0,
        full_PPO_interface_not_yet_verified=True))


class NetworkBudget:
    def __init__(self,p):self.p=p;self.policy_rows=0;self.value_rows=0;self.policy_calls=0;self.value_calls=0;self.last={}
    def call(self,kind,fn,*args,**kwargs):
        obs=args[0] if args else kwargs['obs'];assert obs.ndim==2 and obs.shape[1]==481 and 1<=len(obs)<=10
        n=len(obs)
        if kind=='policy':
            assert n==10
            if self.policy_rows+n>self.p['policy_forward_row_budget']:raise RuntimeError('Policy forward budget exhausted')
            self.policy_rows+=n;self.policy_calls+=1
        else:
            if self.value_rows+n>self.p['value_only_forward_row_budget'] or self.value_calls>=self.p['value_only_forward_call_budget']:raise RuntimeError('Value forward budget exhausted')
            self.value_rows+=n;self.value_calls+=1
        self.last[kind]=dict(normalized_input=obs.detach().cpu().numpy().copy() if isinstance(obs,torch.Tensor) else np.asarray(obs).copy())
        result=fn(*args,**kwargs)
        if kind=='policy':
            for name,value in zip(('gaussian_action3','values','log_probs'),result):self.last[kind][name]=value.detach().cpu().numpy().copy()
        else:self.last[kind]['values']=result.detach().cpu().numpy().copy()
        return result
    def report(self):return dict(policy_rows=self.policy_rows,policy_calls=self.policy_calls,value_only_rows=self.value_rows,value_only_calls=self.value_calls)


def forbidden(*args,**kwargs):raise RuntimeError('Optimizer/train/learn not admitted')


def rms(norm):
    values={'returns':norm.returns.copy()}
    for label in ('obs_rms','ret_rms'):
        for name in ('mean','var','count'):values[label+'_'+name]=np.asarray(getattr(getattr(norm,label),name)).copy()
    return values


def policy_snapshot(model,norm):
    return dict(policy={k:v.detach().cpu().clone() for k,v in model.policy.state_dict().items()},optimizer=copy.deepcopy(model.policy.optimizer.state_dict()),
        python_RNG=random.getstate(),numpy_RNG=np.random.get_state(),torch_RNG=torch.get_rng_state().clone(),cuda_RNG=[v.clone() for v in torch.cuda.get_rng_state_all()],
        action_space_RNG=copy.deepcopy(model.action_space.np_random.bit_generator.state),
        RMS=rms(norm),method_observation_version='cartesian_latent_memory_v1/causal_H1_481',action_version='Cart_latent3_paired6_v1')


class RawReceiptTap(VecEnvWrapper):
    def __init__(self,history,raw):
        super().__init__(history);self.raw=raw;self.norm=None;self.observation=None;self.packet=None
    def reset(self):
        self.observation=self.venv.reset().copy();return self.observation.copy()
    def step_async(self,actions):
        self.actor=self.observation.copy();self.action=actions.copy();self.before=rms(self.norm)
        self.packet=dict(actor=self.actor,action=self.action,rms_before=self.before,post={},bank={})
        self.venv.step_async(actions)
    def step_wait(self):
        value=self.venv.step_wait();obs,reward,done,infos=value
        self.packet.update(returned=obs.copy(),reward=reward.copy(),done=done.copy(),infos=copy.deepcopy(infos))
        self.packet['nominal_inputs']=self.raw._cartesian_ppo_buffers['nominal'].numpy().copy()
        self.packet['post']=recorder.post_private(self.raw);self.packet['bank']=recorder.bank_actual(self.raw)
        self.observation=obs.copy();return value


class ReceiptCallback(BaseCallback):
    def __init__(self,p,env,raw,history,norm,tap,directory,graph,errors):
        super().__init__();self.p=p;self.env=env;self.raw=raw;self.history=history;self.norm=norm;self.tap=tap;self.directory=directory
        self.graph=graph;self.errors=errors;self.records=[];self.blocks=[];self.episodes=[];self.packet=0;self.chunks=0;self.last_actor=None
        self.replay=parent.Replay(raw.obs.numpy());self.stages=env.stages.copy()
        self.events=dict(partial_resets=0,stage2_rows=0,stage3_rows=0,positive_delay_rows=0)
    def _on_step(self):
        raw=self.raw;tap=self.tap.packet;saved=raw._cartesian_last;done=tap['done'];after={n:raw._cartesian_buffers[n].numpy() for n in ('latent','shadow','reference')}
        stages=self.stages;intervals=raw._execution_intervals;post=tap['post'];bank=tap['bank'];local=self.locals
        terminal_actual=np.stack([info.get('terminal_observation',np.zeros(481,np.float32)) for info in tap['infos']])
        terminal_normalized=np.stack([info.get('terminal_observation',np.zeros(481,np.float32)) for info in local['infos']])
        actions=local['actions'].copy();clipped=local['clipped_actions'].copy()
        distribution=self.model.policy.action_dist.distribution
        self.last_actor=dict(packet=tap['actor'],normalized=local['obs_tensor'].detach().cpu().numpy().copy(),gaussian_action3=actions,clipped_action3=clipped,actions=tap['action'])
        record=dict(actor=tap['actor'],action=tap['action'],returned=tap['returned'],reward=tap['reward'],stage_before=stages.copy(),stage_after=self.env.stages.copy(),**saved,
            **{n+'_after':v for n,v in after.items()},**{'post_'+n:v for n,v in post.items()},**{'bank_'+n:v for n,v in bank.items()},
            **{'rms_before_'+n:v for n,v in tap['rms_before'].items()},**{'rms_after_'+n:v for n,v in rms(self.norm).items()},
            nominal_inputs=tap['nominal_inputs'],command_integral=intervals['command_integral'],sensor_elapsed_s=intervals['sensor_elapsed_s'],public_after=raw.obs.numpy(),
            terminal_info_history=terminal_actual,terminal_expected_history=np.full((10,481),np.nan,np.float32),terminal_expected_verified=np.array(False),
            normalized_terminal_history=terminal_normalized,normalized_actor=self.last_actor['normalized'],normalized_returned=local['new_obs'].copy(),
            gaussian_action3=actions,clipped_action3=clipped,values=local['values'].detach().cpu().numpy().copy(),log_probs=local['log_probs'].detach().cpu().numpy().copy(),
            gaussian_mean3=distribution.mean.detach().cpu().numpy().copy(),gaussian_std3=distribution.stddev.detach().cpu().numpy().copy(),
            env_reward_before_bootstrap=local['rewards'].copy(),timeout=np.array([info.get('TimeLimit.truncated',False) for info in tap['infos']]))
        self.records.append(record)
        np.testing.assert_array_equal(clipped,np.clip(actions,-1,1));np.testing.assert_array_equal(tap['action'],embed(clipped,'D3',10))
        parent.physical(saved['state']);record['terminal_expected_history']=self.replay.check(tap['actor'],tap['returned'],done,tap['infos'],saved,intervals,record['public_after'])
        record['terminal_expected_verified']=np.array(True)
        np.testing.assert_array_equal(record['terminal_info_history'][done],record['terminal_expected_history'][done])
        np.testing.assert_array_equal(record['normalized_returned'],self.norm.normalize_obs(tap['returned']))
        if done.any():np.testing.assert_array_equal(terminal_normalized[done],self.norm.normalize_obs(terminal_actual[done]))
        np.testing.assert_array_equal(local['rewards'],tap['reward'].astype(np.float32))
        np.testing.assert_array_equal(tap['nominal_inputs'][:,:,0]==1,saved['trace'][:,:,0]==1)
        assert not tap['nominal_inputs'][:,:,4][saved['trace'][:,:,0]==1].any()
        for n in after:np.testing.assert_array_equal(after[n][~done],saved[n][~done])
        for n in post:
            if n=='previous_sensor_steps':continue
            if n in ('phase_role','phase_monitor'):np.testing.assert_array_equal(post[n][:,~done],saved[n][:,~done])
            else:np.testing.assert_array_equal(post[n][~done],saved[n][~done])
        np.testing.assert_array_equal(post['previous_sensor_steps'],self.replay.previous)
        if done.any():
            np.testing.assert_array_equal(after['latent'][done],0);np.testing.assert_array_equal(after['shadow'][done,16:22],0)
            np.testing.assert_allclose(after['reference'][done],parent.public_reference(raw.obs.numpy()[done]),rtol=0,atol=1e-7)
            for n in ('command_total','command_prefix'):np.testing.assert_array_equal(post[n][done],0)
            stamp=np.full((done.sum(),parent.collector.RING),-1);stamp[:,0]=0;np.testing.assert_array_equal(post['command_stamp'][done],stamp)
        for w in np.flatnonzero(done):
            info=tap['infos'][w];row=self.env.banks[int(stages[w])][w]
            assert info['physical_safety_passed'] and info['design_joint_passed']
            assert info['curriculum_stage']==stages[w] and info['seed']==row['seed'] and info['scenario']==row['scenario']
            self.episodes.append(dict(packet=self.packet,world=int(w),**{n:v for n,v in info.items() if n!='terminal_observation'}))
        changed=stages!=self.env.stages;assert not np.any(changed&~done)
        if changed.any():parent.check_bank(raw,self.env,self.directory,str(self.packet))
        for w in np.flatnonzero(changed):
            cache=self.env.cache[int(self.env.stages[w])]
            for n,key in (('param','param'),('q0','q0'),('obs0','obs0'),('native_reference','reference')):np.testing.assert_array_equal(bank[n][w],cache[key][w])
        self.events['partial_resets']+=int(0<done.sum()<10)
        for s in (2,3):self.events[f'stage{s}_rows']+=int((changed&(self.env.stages==s)).sum())
        self.events['positive_delay_rows']+=int(((saved['delay_steps']>0)&(intervals['sensor_elapsed_s']>0)&(saved['guard'][:,:,0].sum(axis=0)>0)).sum())
        self.stages=self.env.stages.copy();self.packet+=1
        if len(self.records)==100:
            parent.flush(self.directory,self.records,self.blocks)
            atomic_json(self.directory/'progress.json',dict(packets=self.packet,events=self.events,graph_steps=self.graph.graph_steps))
            print('PPO376',self.directory.name,self.packet,self.events,flush=True)
        return True
    def _on_rollout_end(self):
        buffer=self.model.rollout_buffer;assert buffer.full and buffer.pos==50
        fields=('observations','actions','rewards','episode_starts','values','log_probs','advantages','returns')
        np.savez_compressed(self.directory/f'rollout_{self.chunks:02d}.npz',**{n:getattr(buffer,n).copy() for n in fields},
            last_values=self.locals['values'].detach().cpu().numpy().copy(),last_dones=self.locals['dones'].copy())
        self.chunks+=1
    def finish(self,primary):parent.finish(self.errors,primary,self.directory,self.records,self.blocks,self.episodes,self.env,self.events,self.last_actor)


def build(p,arm):
    old=parent.old;original=old.base.raw_env;constructed=[];topologies=[]
    def factory(rows,mode):
        standard=lambda:old.collector.instrument(lambda a,b:old.parking.instrument(original,a,b),rows,mode)
        guarded=lambda:old.guard.instrument(lambda:(standard(),),rows,None,dense=False)
        raw=recorder.instrument(guarded,rows,0 if arm=='CartLive3' else 1)[0]
        constructed.append(parent.model_values(raw));topologies.append(copy.deepcopy(raw._cartesian_ppo_topology));return raw
    with patch.object(old.base,'raw_env',factory):env=old.CartesianCurriculum(p,'virtual6',33531,10,milestones=[4000,8000])
    assert len(constructed)==3
    raw=env.venv;history=old.ExecutionHistory(old.RouteState(env),raw,'H1')
    return env,raw,history,constructed,topologies


def save_initial(directory,env,raw,history,models,topologies):
    for stage,(arrays,metadata) in enumerate(models,1):
        np.savez_compressed(directory/f'constructor_stage{stage}.npz',**arrays);atomic_json(directory/f'constructor_stage{stage}.json',dict(scalars=metadata,topology=topologies[stage-1]))
    for stage,cache in env.cache.items():
        np.savez_compressed(directory/f'cache_stage{stage}.npz',**cache['model'],meaninertia=cache['meaninertia'],q0=cache['q0'],param=cache['param'],obs0=cache['obs0'],reference=cache['reference'],geom_xpos=cache['geom_xpos'],geom_xmat=cache['geom_xmat'])
    parent.check_bank(raw,env,directory,'initial');np.savez_compressed(directory/'initial.npz',**parent.snapshot(raw,history))


def run():
    with (ROOT/'.git/project-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,a=verify(True);QUEUE.mkdir(exist_ok=False);torch.set_num_threads(1)
        atomic_json(QUEUE/'started.json',dict(round=376,pid=os.getpid(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),admission_sha256=sha(OUT/'source_admission.json'),implicit_retry=False))
        graph=parent.Budget(p);network=NetworkBudget(p);errors=[];holders=[];reports={};first_models=first_caches=first_initial=first_policy=None;start=time.perf_counter()
        launch,fd=wp.capture_launch,mujoco.mjd_transitionFD
        try:
            with patch.object(wp,'capture_launch',lambda *x,**k:graph.capture(launch,*x,**k)),patch.object(mujoco,'mjd_transitionFD',lambda *x,**k:graph.fd(fd,*x,**k)):
                for arm in p['arms']:
                    verify(True);directory=QUEUE/arm;directory.mkdir();env,raw,history,models,topologies=build(p,arm)
                    holder=dict(raw=raw,history=history,directory=directory,norm=None,model=None,callback=None,tap=None);holders.append(holder);graph.bind(raw)
                    obs=history.reset();save_initial(directory,env,raw,history,models,topologies)
                    initial=parent.snapshot(raw,history);caches={s:{**c['model'],**{n:c[n] for n in ('meaninertia','q0','param','obs0','reference','geom_xpos','geom_xmat')}} for s,c in env.cache.items()}
                    if first_models is None:first_models=models;first_caches=caches;first_initial=initial
                    else:equal(models,first_models);equal(caches,first_caches);equal(initial,first_initial)
                    replay=parent.Replay(raw.obs.numpy());zero=history.step(np.zeros((10,6),np.float32));saved=raw._cartesian_last
                    np.savez_compressed(directory/'zero_before_autoreset.npz',**saved,nominal_inputs=raw._cartesian_ppo_buffers['nominal'].numpy())
                    np.savez_compressed(directory/'zero_prefix.npz',**parent.snapshot(raw,history),actor=obs,returned=zero[0],reward=zero[1],done=zero[2],**raw._execution_intervals)
                    assert not zero[2].any();parent.physical(saved['state']);replay.check(obs,zero[0],zero[2],zero[3],saved,raw._execution_intervals,raw.obs.numpy())
                    np.testing.assert_array_equal(saved['latent'],0);np.testing.assert_array_equal(saved['shadow'][:,16:22],0);assert env.policy_steps==10
                    tap=RawReceiptTap(history,raw);holder['tap']=tap
                    norm=VecNormalize(VecCheckNan(ForceActions(tap,'D3'),raise_exception=True),**p['normalization']);tap.norm=norm;holder['norm']=norm
                    model,_=agent(p,norm);holder['model']=model;callback=ReceiptCallback(p,env,raw,history,norm,tap,directory,graph,errors);holder['callback']=callback
                    model._setup_learn(total_timesteps=12000,callback=callback,reset_num_timesteps=True);assert norm.action_space.shape==(3,) and model.rollout_buffer.buffer_size==50
                    callback.replay=parent.Replay(raw.obs.numpy())
                    model.policy.set_training_mode(False);callback.on_training_start({}, {})
                    before=policy_snapshot(model,norm)
                    before.update(initial_raw_observation=tap.observation.copy(),initial_normalized_observation=model._last_obs.copy())
                    assert not torch.count_nonzero(model.policy.action_net.weight) and not torch.count_nonzero(model.policy.action_net.bias)
                    torch.testing.assert_close(model.policy.log_std.exp(),torch.full_like(model.policy.log_std,.25),rtol=0,atol=3e-8)
                    torch.save(before,directory/'initial_policy.pt')
                    np.savez_compressed(directory/'initial_policy_observation.npz',raw=tap.observation,normalized=model._last_obs,**rms(norm))
                    if first_policy is None:first_policy=before
                    else:equal(before,first_policy)
                    np.testing.assert_allclose(norm.obs_rms.count,10.0001,rtol=0,atol=1e-12)
                    original_forward,original_values=model.policy.forward,model.policy.predict_values
                    primary=None
                    try:
                        with patch.object(model.policy,'forward',lambda *x,**k:network.call('policy',original_forward,*x,**k)),patch.object(model.policy,'predict_values',lambda *x,**k:network.call('value',original_values,*x,**k)),\
                             patch.object(model,'train',forbidden),patch.object(model,'learn',forbidden),patch.object(model.policy.optimizer,'step',forbidden):
                            for _ in range(24):assert model.collect_rollouts(norm,callback,model.rollout_buffer,n_rollout_steps=50)
                    except BaseException as error:primary=error;raise
                    finally:callback.finish(primary)
                    assert callback.packet==1200 and callback.chunks==24 and model.num_timesteps==12000 and model._n_updates==0 and env.policy_steps==12010
                    callback.on_training_end()
                    equal(model.policy.state_dict(),before['policy']);equal(model.policy.optimizer.state_dict(),before['optimizer'])
                    np.testing.assert_allclose(norm.obs_rms.count,12010.0001,rtol=0,atol=1e-7)
                    np.testing.assert_allclose(norm.ret_rms.count,12000.0001,rtol=0,atol=1e-7)
                    torch.save(policy_snapshot(model,norm),directory/'final_policy.pt');model.save(directory/'final_model.zip');norm.save(directory/'final_RMS.pkl')
                    parent.check_bank(raw,env,directory,'final');parent.preserve(raw,history,directory,errors)
                    assert graph.active['calls']==1201 and not errors;norm.close();verify(True)
                    reports[arm]=dict(status='producer_passed_pending_independent_receipt' if all(callback.events.values()) else 'inconclusive_missing_registered_events',packets=callback.packet,chunks=callback.chunks,
                        events=callback.events,episodes=len(callback.episodes),task_successes=sum(bool(e['success']) for e in callback.episodes),optimizer_updates=0)
            assert graph.graph_steps==960800 and network.policy_rows==24000
            atomic_json(QUEUE/'completion.json',dict(round=376,status='producer_finished_pending_independent_receipt',reports=reports,graph_world_steps=graph.graph_steps,graph_calls_each=[b['calls'] for b in graph.bindings],FD_calls=graph.FD,
                network=network.report(),collected_policy_samples=24000,optimizer_updates=0,wall_seconds=time.perf_counter()-start,secondary_errors=errors,formal5_admitted=False))
        except BaseException as error:
            for h in holders:
                parent.preserve(h['raw'],h['history'],h['directory'],errors)
                parent.secondary(errors,h['directory'].name+'/nominal',lambda h=h:np.savez_compressed(h['directory']/'interrupted_nominal.npz',nominal=h['raw']._cartesian_ppo_buffers['nominal'].numpy()))
                if h['model'] is not None:
                    parent.secondary(errors,h['directory'].name+'/policy',lambda h=h:torch.save(policy_snapshot(h['model'],h['norm']),h['directory']/'interrupted_policy.pt'))
                    buffer=h['model'].rollout_buffer
                    parent.secondary(errors,h['directory'].name+'/partial_rollout',lambda h=h,buffer=buffer:np.savez_compressed(h['directory']/'interrupted_rollout.npz',
                        **{n:getattr(buffer,n).copy() for n in ('observations','actions','rewards','episode_starts','values','log_probs','advantages','returns')},pos=buffer.pos,full=buffer.full))
                if h['tap'] is not None:
                    tap=h['tap']
                    if hasattr(tap,'actor'):
                        parent.secondary(errors,h['directory'].name+'/latest_actor',lambda h=h,tap=tap:np.savez_compressed(h['directory']/'interrupted_actor.npz',actor=tap.actor,paired6=tap.action))
                    if tap.packet is not None:
                        current=tap.packet
                        parent.secondary(errors,h['directory'].name+'/tap',lambda h=h,current=current:np.savez_compressed(h['directory']/'interrupted_tap.npz',
                            **{n:current[n] for n in ('actor','action','returned','reward','done','nominal_inputs') if n in current},
                            **{'terminal_info_world_'+str(w):i['terminal_observation'] for w,i in enumerate(current.get('infos',[])) if 'terminal_observation' in i},
                            **{'post_'+n:v for n,v in current['post'].items()},**{'bank_'+n:v for n,v in current['bank'].items()},**{'rms_before_'+n:v for n,v in current['rms_before'].items()}))
                        parent.secondary(errors,h['directory'].name+'/tap_infos',lambda h=h,current=current:atomic_json(h['directory']/'interrupted_tap_infos.json',dict(captured_fields=list(current),infos=[{n:v for n,v in i.items() if n!='terminal_observation'} for i in current.get('infos',[])])))
                parent.secondary(errors,h['directory'].name+'/close',(h['norm'] or h['history']).close)
            for kind,values in network.last.items():
                parent.secondary(errors,'network/'+kind,lambda kind=kind,values=values:np.savez_compressed(QUEUE/('interrupted_'+kind+'_forward.npz'),**values))
            failure=dict(round=376,status='failed',primary_error=repr(error),secondary_errors=errors,reports=reports,graph_world_steps=graph.graph_steps,graph_calls_each=[b['calls'] for b in graph.bindings],FD_calls=graph.FD,network=network.report(),optimizer_updates=0,wall_seconds=time.perf_counter()-start,implicit_retry=False)
            parent.secondary(errors,'failure_manifest',lambda:atomic_json(QUEUE/'failure.json',failure));print('FAILED376',json.dumps(failure,ensure_ascii=False),flush=True);raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args();{'freeze':freeze,'run':run}[args.command]()
