"""One frozen19-job original/half/reflection diagnostic; no learning or retries."""
import fcntl
import json
import os
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import mujoco
import torch
import warp as wp
from stable_baselines3 import PPO
import execution_history_evaluation as evaluation
import joint_state_guard as guard
from fixed_reference_force import PolicyActor
from train_fixed_force_study import OUT as STUDY
from review_fixed_force_evaluation import chain_minima,physical_metrics
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
import run_wheel_interference as prior
from audit_policy_reflection import reflect_raw,ACTION,self_check as reflection_check

OUT=STUDY.parent/'frozen_reflection_diagnostic_v1'


class DiagnosticActor(PolicyActor):
    def __init__(self,model,arm,operator,charge):
        super().__init__(model,arm)
        if operator not in ('original','half','reflection'):raise ValueError('Registered operator required')
        self.operator=operator;self.charge=charge;self.history=self.norm=None
        self.requests=[];self.secondary=[];self.secondary_obs=[];self.submitted=[]
    def predict(self,obs,deterministic=True):
        if not deterministic:raise ValueError('Only frozen deterministic predictions admitted')
        if self.history is None or self.norm is None or self.norm.training or self.norm.norm_reward:raise RuntimeError('Bind public history and frozenNormalizer first')
        raw=self.history.encode();np.testing.assert_array_equal(obs,self.norm.normalize_obs(raw.copy()))
        self.charge(len(obs)*(2 if self.operator=='reflection' else 1))
        original,state=super().predict(obs,deterministic=True);self.requests.append(original.copy());submitted=original
        if self.operator=='half':submitted=.5*original
        elif self.operator=='reflection':
            reflected=reflect_raw(raw);normalized=self.norm.normalize_obs(reflected)
            other,_=super().predict(normalized,deterministic=True)
            submitted=.5*(original+other[:,ACTION]);self.secondary.append(other.copy());self.secondary_obs.append(normalized.copy())
            if len(self.requests)==1 and self.arm=='D3' and np.array_equal(reflected,raw):np.testing.assert_array_equal(submitted,0)
        assert np.isfinite(submitted).all() and abs(submitted).max()<=1
        self.submitted.append(submitted.copy());return submitted,state


runtime=prior.runtime


def inputs():
    original=prior.inputs()
    p=json.loads((OUT/'proposal.json').read_text())
    assert p['evaluations']==len(p['conditions'])==19 and p['models']==original['models'] and p['case']==original['case'] and p['new_learning_samples']==0
    assert p['old_candidate_benefit_branch_closed'] and not p['formal5_admitted']
    direction=json.loads((OUT.parent/'round345_direction_review.json').read_text());assert direction['diagnostic_proposal_sha256']==sha(OUT/'proposal.json')
    for mapping in (p['source_sha256'],p['model_sha256'],p['evidence_sha256']):assert all(sha(ROOT/f)==h for f,h in mapping.items())
    a=json.loads((STUDY/'model_evaluation_admission.json').read_text());t=json.loads((STUDY/'training_review.json').read_text());c=json.loads((STUDY/'training/completion.json').read_text())
    assert a['verified'] and t['verified'] and c['verified'] and c['policy_samples']==1200000 and len(c['reports'])==6
    assert a['proposal_sha256']==sha(STUDY/'proposal.json') and a['training_review_sha256']==sha(STUDY/'training_review.json')
    assert a['training_completion_sha256']==t['completion_sha256']==sha(STUDY/'training/completion.json')
    assert p['model_sha256']==a['model_sha256']
    assert p['conditions'][0]==dict(label='guard_B0',model=None,operator='zero')
    expected=[dict(label=m['label']+'_'+name,model=m['label'],operator=name) for m in p['models'] for name in ('original','half','reflection')]
    assert p['conditions'][1:]==expected and len({m['label'] for m in p['models']})==6
    return p


def self_check():
    reflection_check()
    class Fake:
        policy=None
        def __init__(self,dim):self.dim=dim
        def predict(self,obs,deterministic=True):return np.tanh(obs[:,:self.dim]*.2+np.arange(self.dim)*.1).astype(np.float32),None
    class History:
        def __init__(self,x):self.x=x
        def encode(self):return self.x.copy()
    class Norm:
        training=norm_reward=False
        def normalize_obs(self,x):return (x+np.arange(481,dtype=np.float32)/1000).astype(np.float32)
    for arm,dim in [('D3',3),('V6',6)]:
        for op in ('original','half','reflection'):
            calls=[];actor=DiagnosticActor(Fake(dim),arm,op,calls.append);raw=np.zeros((2,481),np.float32);actor.history=History(raw);actor.norm=Norm();obs=actor.norm.normalize_obs(raw)
            value=actor.predict(obs)[0];u=actor.requests[0]
            expected=u if op=='original' else (.5*u if op=='half' else .5*(u+actor.secondary[0][:,ACTION]))
            np.testing.assert_array_equal(value,expected);assert sum(calls)==(4 if op=='reflection' else 2)
            if op=='reflection':np.testing.assert_array_equal(actor.secondary_obs[0],actor.norm.normalize_obs(reflect_raw(raw)))
            try:actor.predict(obs+1)
            except AssertionError:pass
            else:raise AssertionError('Stale/wrong normalized input accepted')
        try:DiagnosticActor(Fake(dim),arm,'other',lambda n:None)
        except ValueError:pass
        else:raise AssertionError('Unregistered operator admitted')
    print('PASS346 raw-beforeRMS/actualbinding/original-half-reflection/D3resetzero/querycounts/stale rejection;0physics',flush=True)


def freeze():
    p=inputs();assert not (OUT/'source_admission.json').exists();self_check();models=[]
    for m in p['models']:
        model=PPO.load(ROOT/m['model'],device='cpu');assert model.num_timesteps==200000 and model._n_updates==400 and model.observation_space.shape==(481,)
        assert model.action_space.shape==(3 if m['arm']=='D3' else 6,)
        with (ROOT/m['normalization']).open('rb') as f:norm=pickle.load(f)
        assert norm.obs_rms.mean.shape==(481,) and np.isfinite(norm.obs_rms.mean).all() and np.isfinite(norm.obs_rms.var).all() and (norm.obs_rms.var>=0).all()
        np.testing.assert_allclose(norm.obs_rms.count,200100.0001,rtol=0,atol=1e-7)
        raw=np.zeros((2,481),np.float32);norm.training=False;norm.norm_reward=False
        class History:
            def encode(self):return raw.copy()
        actor=DiagnosticActor(model,m['arm'],'reflection',lambda n:None);actor.history=History();actor.norm=norm
        value=actor.predict(norm.normalize_obs(raw.copy()))[0];assert value.shape==(2,6) and np.isfinite(value).all()
        if m['arm']=='D3':np.testing.assert_array_equal(value,0)
        models.append(dict(label=m['label'],timesteps=model.num_timesteps,epochs=model._n_updates,model_sha256=sha(ROOT/m['model']),RMS_sha256=sha(ROOT/m['normalization'])))
    sources={**p['source_sha256'],str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__),
        'wheelleg_warp/review_fixed_force_evaluation.py':sha(ROOT/'wheelleg_warp/review_fixed_force_evaluation.py'),
        'wheelleg_warp/run_wheel_interference.py':sha(ROOT/'wheelleg_warp/run_wheel_interference.py')}
    atomic_json(OUT/'source_admission.json',dict(round=346,verified=True,proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,runtime=runtime(),model_receipts=models,
        complete_original_training_admission_bound=True,constructor_sources_prechecked=True,operator_unit_passed=True,new_physics_steps=0,new_learning_samples=0,formal5_admitted=False))
    print('FROZEN346 exact19 diagnostic/runtime/model/input links;0physics/learning',flush=True)


def verify():
    p=inputs();a=json.loads((OUT/'source_admission.json').read_text());assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items()) and a['runtime']==runtime()
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Need2GiB free for diagnostic archives')
    return p,a


def run():
    with (ROOT/'.git/project-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,a=verify()
        assert not any((OUT/f).exists() for f in ('started.json','completion.json','failure.json'))
        torch.set_num_threads(1);records=[];steps=0;query_rows=0;fd_calls=[];holder={};current=None;initial=None;start=time.perf_counter()
        old=evaluation.make_env;graph=wp.capture_launch;fd=mujoco.mjd_transitionFD
        def capture(*args,**kwargs):
            nonlocal steps
            assert steps+40<=p['actual_graph_world_step_budget'];steps+=40;return graph(*args,**kwargs)
        def counted(*args,**kwargs):
            assert len(fd_calls)<p['constructor_FD_budget'];fd_calls.append(float(args[2]));return fd(*args,**kwargs)
        def charge(n):
            nonlocal query_rows
            assert query_rows+n<=p['model_forward_row_budget'];query_rows+=n
        def factory(cases,arm,normalization,directory):
            def build():
                value=old(cases,arm,normalization,directory);holder.update(raw=value[0],history=value[2]);return value
            result=guard.instrument(build,cases,directory);raw,_,history,norm,env=result;holder['guard_chunks']=[]
            if actor is not None:actor.history=history;actor.norm=norm
            reset,async_,wait=env.reset,raw.step_async,raw.step_wait
            def reset_all():
                nonlocal initial
                value=reset();snap=dict(q=raw.data.qpos.numpy(),v=raw.data.qvel.numpy(),param=raw.param.numpy(),reference=raw.k['reference'].numpy(),
                    controller_memory=raw.k['state'].numpy(),guard_memory=raw._joint_guard_buffers[0].numpy(),frames=history.frames.copy(),inputs=history.inputs.copy(),elapsed=history.elapsed.copy(),valid=history.valid.copy(),geom_xpos=raw.data.geom_xpos.numpy(),geom_xmat=raw.data.geom_xmat.numpy())
                if initial is None:initial=snap
                else:
                    for k in snap:np.testing.assert_array_equal(snap[k],initial[k])
                np.savez_compressed(directory/'initial.npz',**snap);return value
            def step_async(actions):
                assert raw.state.numpy()[0,0]+40<=p['per_job_first_episode_steps_cap'];return async_(actions)
            def step_wait():
                value=wait();g=raw._joint_guard_buffers[1].numpy()[:,0];holder['guard_chunks'].append(g[g[:,0]==1].copy())
                t=raw._complete_buffers[0].numpy()[:,0];keep=t[:,0]==1
                pre=raw._complete_buffers[4].numpy()[:,0][keep];post=raw._complete_buffers[5].numpy()[:,0][keep]
                if len(post):
                    lengths,_=chain_minima(post[:,:17],raw.cpu);margin,actual,issued=physical_metrics(pre,post,raw.cpu)
                    assert min(lengths)>=HEIGHT_115_GEOMETRIC_MIN and margin>=0 and max(actual,issued)<=1e-6 and t[keep,24].min()>=0,'Recorded physical/design violation'
                return value
            env.reset,raw.step_async,raw.step_wait=reset_all,step_async,step_wait;return result
        def preserve(directory,actor):
            if actor is not None and actor.requests:
                np.savez_compressed(directory/'unmasked_requests.npz',trace=np.concatenate(actor.requests))
                np.savez_compressed(directory/'operator_queries.npz',original=np.concatenate(actor.requests),submitted=np.concatenate(actor.submitted) if actor.submitted else np.empty((0,6),np.float32),
                    secondary=np.concatenate(actor.secondary) if actor.secondary else np.empty((0,6),np.float32),
                    secondary_normalized=np.concatenate(actor.secondary_obs) if actor.secondary_obs else np.empty((0,481),np.float32))
            if 'raw' in holder:
                raw=holder['raw'];h=holder['history'];extra={}
                if hasattr(raw,'_joint_guard_buffers'):extra=dict(guard_memory=raw._joint_guard_buffers[0].numpy(),guard_last=raw._joint_guard_buffers[1].numpy())
                np.savez_compressed(directory/'terminal.npz',current_q=raw.data.qpos.numpy(),current_v=raw.data.qvel.numpy(),terminal_q=raw.stopped_q.numpy(),terminal_v=raw.stopped_v.numpy(),
                    state=raw.state.numpy(),controller_memory=raw.k['state'].numpy(),frames=h.frames,inputs=h.inputs,**extra)
                if holder.get('guard_chunks'):np.savez_compressed(directory/'guard_prefix.npz',trace=np.concatenate(holder['guard_chunks']))
        evaluation.make_env=factory;wp.capture_launch=capture;mujoco.mjd_transitionFD=counted
        atomic_json(OUT/'started.json',dict(pid=os.getpid(),proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'source_admission.json'),exclusive_project_lock=True))
        directory=OUT;actor=None
        try:
            for condition in p['conditions']:
                verify();current=condition['label'];holder.clear();actor=None;directory=OUT/current;directory.mkdir(exist_ok=False);normalization=None
                if condition['model'] is not None:
                    m=next(m for m in p['models'] if m['label']==condition['model']);loaded=PPO.load(ROOT/m['model'],device='cuda');assert loaded.num_timesteps==200000 and loaded._n_updates==400
                    actor=DiagnosticActor(loaded,m['arm'],condition['operator'],charge);normalization=ROOT/m['normalization']
                before=steps;queries_before=query_rows;result,checked=evaluation.evaluate([p['case']],'H1' if actor else 'B0',actor,normalization,directory)
                preserve(directory,actor);row=result['runs'][0]
                with np.load(directory/row['actor_trace']['path']) as z:submitted=z['trace'][:,962:]
                original=np.concatenate(actor.requests) if actor else np.zeros_like(submitted)
                assert original.shape==submitted.shape
                expected=np.zeros_like(submitted)
                if actor:
                    expected=original if actor.operator=='original' else (.5*original if actor.operator=='half' else .5*(original+np.concatenate(actor.secondary)[:,ACTION]))
                    np.testing.assert_array_equal(submitted,np.concatenate(actor.submitted))
                np.testing.assert_array_equal(submitted,expected)
                assert query_rows-queries_before==(0 if actor is None else len(submitted)*(2 if actor.operator=='reflection' else 1))
                assert row['physical_steps']<=p['per_job_first_episode_steps_cap'] and result['physical']==result['design']==1
                if actor is None:assert row['success'],'FreshB0 task qualification failed'
                records.append(dict(condition=current,path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),success=row['success'],physical=result['physical'],design=result['design'],
                    actual_graph_world_steps=steps-before,model_forward_rows=query_rows-queries_before,first_episode_world_steps=row['physical_steps'],checked=checked,initial_sha256=sha(directory/'initial.npz'),terminal_sha256=sha(directory/'terminal.npz'),unmasked_sha256=None if actor is None else sha(directory/'unmasked_requests.npz'),operator_queries_sha256=None if actor is None else sha(directory/'operator_queries.npz')))
                atomic_json(OUT/'progress.json',dict(records=records,evaluations=len(records),actual_graph_world_steps=steps,model_forward_rows=query_rows,FD_calls=fd_calls))
                print('DIAGNOSTIC346',current,'task',row['success'],'yaw',row['peak_deg'][2],flush=True)
            verify();assert len(records)==19
            atomic_json(OUT/'completion.json',dict(round=346,verified=True,records=records,evaluations=19,model_forward_rows=query_rows,proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'source_admission.json'),
                actual_graph_world_steps=steps,first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),FD_calls=fd_calls,wall_seconds=time.perf_counter()-start,new_learning_samples=0,formal5_admitted=False,original_candidate_benefit_branch_closed=True))
        except BaseException as error:
            preservation_error=None
            try:preserve(directory,actor)
            except BaseException as saving_error:preservation_error=repr(saving_error)
            atomic_json(OUT/'failure.json',dict(error=repr(error),preservation_error=preservation_error,condition=current,records=records,actual_graph_world_steps=steps,model_forward_rows_reserved=query_rows,FD_calls=fd_calls,partial_raw_available='raw' in holder,implicit_retry=False));raise
        finally:evaluation.make_env=old;wp.capture_launch=graph;mujoco.mjd_transitionFD=fd


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['self-check','freeze','run']);args=parser.parse_args()
    {'self-check':self_check,'freeze':freeze,'run':run}[args.command]()
