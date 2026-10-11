"""One frozen13-job wheel-request diagnostic; no learning, retries or promotion."""
import fcntl
import importlib.metadata as metadata
import inspect
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import mujoco
import torch
import warp as wp
from stable_baselines3 import PPO
from mujoco_warp._src import forward,support
import execution_history_evaluation as evaluation
import joint_state_guard as guard
from fixed_reference_force import PolicyActor
from train_fixed_force_study import OUT as STUDY
from review_fixed_force_evaluation import chain_minima,physical_metrics
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
from native.terrain import HEIGHT_115_GEOMETRIC_MIN

OUT=STUDY.parent/'frozen_wheel_interference_diagnostic_v1'


class DiagnosticActor(PolicyActor):
    def __init__(self,model,arm,mask):
        super().__init__(model,arm);self.mask=np.asarray(mask,np.float32)
        if self.mask.shape!=(6,) or tuple(self.mask) not in ((1,1,1,1,1,1),(1,1,1,1,0,0)):raise ValueError('Registered original/wheel_off mask required')
        self.requests=[]
    def predict(self,obs,deterministic=True):
        if not deterministic:raise ValueError('Only frozen deterministic predictions admitted')
        original,state=super().predict(obs,deterministic=True);submitted=original*self.mask
        np.testing.assert_array_equal(submitted[:,:4],original[:,:4])
        self.requests.append(original.copy());return submitted,state


def runtime():
    return dict(versions={n:metadata.version(n) for n in ('mujoco','mujoco-warp','warp-lang','torch','stable-baselines3','gymnasium','numpy')},
        python=sys.version,
        cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        installed_sha256={inspect.getfile(m):sha(inspect.getfile(m)) for m in (forward,support)})


def inputs():
    p=json.loads((OUT/'proposal.json').read_text())
    assert p['evaluations']==len(p['conditions'])==13 and len(p['models'])==6 and p['new_learning_samples']==0
    assert p['old_candidate_benefit_branch_closed'] and not p['formal5_admitted']
    direction=json.loads((OUT.parent/'round340_direction_review.json').read_text());assert direction['diagnostic_proposal_sha256']==sha(OUT/'proposal.json')
    for mapping in (p['source_sha256'],p['model_sha256'],p['evidence_sha256']):assert all(sha(ROOT/f)==h for f,h in mapping.items())
    a=json.loads((STUDY/'model_evaluation_admission.json').read_text());t=json.loads((STUDY/'training_review.json').read_text());c=json.loads((STUDY/'training/completion.json').read_text())
    assert a['verified'] and t['verified'] and c['verified'] and c['policy_samples']==1200000 and len(c['reports'])==6
    assert a['proposal_sha256']==sha(STUDY/'proposal.json') and a['training_review_sha256']==sha(STUDY/'training_review.json')
    assert a['training_completion_sha256']==t['completion_sha256']==sha(STUDY/'training/completion.json')
    assert p['model_sha256']==a['model_sha256']
    assert p['conditions'][0]==dict(label='guard_B0',model=None,mask=[0]*6)
    expected=[dict(label=m['label']+'_'+name,model=m['label'],mask=mask) for m in p['models'] for name,mask in [('original',[1]*6),('wheel_off',[1,1,1,1,0,0])]]
    assert p['conditions'][1:]==expected and len({m['label'] for m in p['models']})==6
    return p


def self_check():
    class Fake:
        policy=None
        def __init__(self,dim):self.dim=dim
        def predict(self,obs,deterministic=True):return np.tile(np.linspace(-.8,.8,self.dim,dtype=np.float32),(len(obs),1)),None
    for arm,dim in [('D3',3),('V6',6)]:
        actor=DiagnosticActor(Fake(dim),arm,[1,1,1,1,0,0]);x=np.zeros((2,481),np.float32);masked=actor.predict(x)[0]
        np.testing.assert_array_equal(masked[:,:4],actor.requests[0][:,:4]);assert not masked[:,4:].any()
        original=DiagnosticActor(Fake(dim),arm,[1]*6);np.testing.assert_array_equal(original.predict(x)[0],original.requests[0])
        for mask in ([1]*3,[1,1,1,1,.5,0]):
            try:DiagnosticActor(Fake(dim),arm,mask)
            except ValueError:pass
            else:raise AssertionError('Nonregistered mask admitted')
        try:actor.predict(x,False)
        except ValueError:pass
        else:raise AssertionError('Random prediction admitted')
    print('PASS341 D3/V6 embed/original/wheeloff/ownlegs/badmask/random rejection;0physics',flush=True)


def freeze():
    p=inputs();assert not (OUT/'source_admission.json').exists();self_check();models=[]
    for m in p['models']:
        model=PPO.load(ROOT/m['model'],device='cpu');assert model.num_timesteps==200000 and model._n_updates==400 and model.observation_space.shape==(481,)
        assert model.action_space.shape==(3 if m['arm']=='D3' else 6,)
        with (ROOT/m['normalization']).open('rb') as f:norm=pickle.load(f)
        assert norm.obs_rms.mean.shape==(481,) and np.isfinite(norm.obs_rms.mean).all() and np.isfinite(norm.obs_rms.var).all() and (norm.obs_rms.var>=0).all()
        np.testing.assert_allclose(norm.obs_rms.count,200100.0001,rtol=0,atol=1e-7)
        obs=norm.normalize_obs(np.zeros((2,481),np.float32));actor=DiagnosticActor(model,m['arm'],[1,1,1,1,0,0]);value=actor.predict(obs)[0]
        assert value.shape==(2,6) and np.isfinite(value).all() and not value[:,4:].any()
        models.append(dict(label=m['label'],timesteps=model.num_timesteps,epochs=model._n_updates,model_sha256=sha(ROOT/m['model']),RMS_sha256=sha(ROOT/m['normalization'])))
    sources={**p['source_sha256'],str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__),
        'wheelleg_warp/review_fixed_force_evaluation.py':sha(ROOT/'wheelleg_warp/review_fixed_force_evaluation.py')}
    atomic_json(OUT/'source_admission.json',dict(round=341,verified=True,proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,runtime=runtime(),model_receipts=models,
        complete_original_training_admission_bound=True,constructor_sources_prechecked=True,mask_unit_passed=True,new_physics_steps=0,new_learning_samples=0,formal5_admitted=False))
    print('FROZEN341 exact13 diagnostic/runtime/model/input links;0physics/learning',flush=True)


def verify():
    p=inputs();a=json.loads((OUT/'source_admission.json').read_text());assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items()) and a['runtime']==runtime()
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Need2GiB free for diagnostic archives')
    return p,a


def run():
    with (ROOT/'.git/project-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,a=verify()
        assert not any((OUT/f).exists() for f in ('started.json','completion.json','failure.json'))
        torch.set_num_threads(1);records=[];steps=0;fd_calls=[];holder={};current=None;initial=None;start=time.perf_counter()
        old=evaluation.make_env;graph=wp.capture_launch;fd=mujoco.mjd_transitionFD
        def capture(*args,**kwargs):
            nonlocal steps
            assert steps+40<=p['actual_graph_world_step_budget'];steps+=40;return graph(*args,**kwargs)
        def counted(*args,**kwargs):
            assert len(fd_calls)<p['constructor_FD_budget'];fd_calls.append(float(args[2]));return fd(*args,**kwargs)
        def factory(cases,arm,normalization,directory):
            def build():
                value=old(cases,arm,normalization,directory);holder.update(raw=value[0],history=value[2]);return value
            result=guard.instrument(build,cases,directory);raw,_,history,_,env=result;holder['guard_chunks']=[]
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
            if actor is not None and actor.requests:np.savez_compressed(directory/'unmasked_requests.npz',trace=np.concatenate(actor.requests))
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
                    actor=DiagnosticActor(loaded,m['arm'],condition['mask']);normalization=ROOT/m['normalization']
                before=steps;result,checked=evaluation.evaluate([p['case']],'H1' if actor else 'B0',actor,normalization,directory)
                preserve(directory,actor);row=result['runs'][0]
                with np.load(directory/row['actor_trace']['path']) as z:submitted=z['trace'][:,962:]
                original=np.concatenate(actor.requests) if actor else np.zeros_like(submitted)
                assert original.shape==submitted.shape;np.testing.assert_array_equal(submitted,original*np.asarray(condition['mask'],np.float32))
                if actor:np.testing.assert_array_equal(submitted[:,:4],original[:,:4])
                assert row['physical_steps']<=p['per_job_first_episode_steps_cap'] and result['physical']==result['design']==1
                if actor is None:assert row['success'],'FreshB0 task qualification failed'
                records.append(dict(condition=current,path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),success=row['success'],physical=result['physical'],design=result['design'],
                    actual_graph_world_steps=steps-before,first_episode_world_steps=row['physical_steps'],checked=checked,initial_sha256=sha(directory/'initial.npz'),terminal_sha256=sha(directory/'terminal.npz'),unmasked_sha256=None if actor is None else sha(directory/'unmasked_requests.npz')))
                atomic_json(OUT/'progress.json',dict(records=records,evaluations=len(records),actual_graph_world_steps=steps,FD_calls=fd_calls))
                print('DIAGNOSTIC341',current,'task',row['success'],'yaw',row['peak_deg'][2],flush=True)
            verify();assert len(records)==13
            atomic_json(OUT/'completion.json',dict(round=341,verified=True,records=records,evaluations=13,proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'source_admission.json'),
                actual_graph_world_steps=steps,first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),FD_calls=fd_calls,wall_seconds=time.perf_counter()-start,new_learning_samples=0,formal5_admitted=False,original_candidate_benefit_branch_closed=True))
        except BaseException as error:
            preserve(directory,actor);atomic_json(OUT/'failure.json',dict(error=repr(error),condition=current,records=records,actual_graph_world_steps=steps,FD_calls=fd_calls,partial_raw_available='raw' in holder,implicit_retry=False));raise
        finally:evaluation.make_env=old;wp.capture_launch=graph;mujoco.mjd_transitionFD=fd


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['self-check','freeze','run']);args=parser.parse_args()
    {'self-check':self_check,'freeze':freeze,'run':run}[args.command]()
