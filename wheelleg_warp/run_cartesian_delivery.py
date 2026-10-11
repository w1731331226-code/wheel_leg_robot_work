"""One registered 20-episode Cartesian delivery queue; no PPO or implicit retries."""
import fcntl
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import mujoco
import torch
import warp as wp
import execution_history_evaluation as evaluation
import joint_state_guard as guard
import cartesian_pair_runtime as cart
from cartesian_pair_action import OUT
from run_reflection_retention import receive_initial
from run_wheel_interference import runtime
from review_fixed_force_evaluation import chain_minima,physical_metrics
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
from analyze_reward_failures import flags

QUEUE=OUT/'runtime'


class Scripted:
    num_timesteps=0
    _n_updates=0
    def __init__(self,profile):
        assert profile in ('zero','sine');self.profile=profile;self.k=0;self.policy=torch.nn.Module()
    def predict(self,obs,deterministic=True):
        assert obs.shape==(1,481) and np.isfinite(obs).all() and deterministic
        t=self.k*.02;self.k+=1
        a=np.zeros(3) if self.profile=='zero' else .15*np.array([np.sin(np.pi*t),np.cos(np.pi*t),np.sin(2*np.pi*t)])
        return np.array([[a[0],-a[0],a[1],-a[1],a[2],-a[2]]],np.float32),None


def inputs():
    p=json.loads((OUT/'runtime_proposal.json').read_text());k=json.loads((OUT/'kernel_review.json').read_text())
    assert p['evaluations']==len(p['conditions'])==20 and p['new_learning_samples']==0 and not p['formal5_admitted']
    assert k['verified'] and k['proposal_sha256']==sha(OUT/'runtime_proposal.json')
    assert k['kernel_sha256']==sha(ROOT/'wheelleg_warp/cartesian_pair_kernel.py')
    for mapping in (p['source_sha256'],p['evidence_sha256'],k['input_sha256'],k['outputs_sha256']):
        assert all(sha(ROOT/f)==h for f,h in mapping.items())
    expected=[dict(label=f'{c["seed"]}_{arm}_{profile}',case=c,arm=arm,profile=profile) for c in p['cases'] for arm,profile in
        [('B0','zero'),('CartLive3','zero'),('CartReset3','zero'),('CartLive3','sine'),('CartReset3','sine')]]
    assert p['conditions']==expected and len(p['cases'])==4
    return p


def freeze():
    from test_cartesian_pair_runtime import run as routing_check
    p=inputs();assert not (OUT/'runtime_admission.json').exists();routing_check()
    obs=np.zeros((1,481),np.float32)
    np.testing.assert_array_equal(Scripted('zero').predict(obs)[0],0)
    actor=Scripted('sine')
    first=actor.predict(obs)[0];np.testing.assert_array_equal(first,np.array([[0,0,.15,-.15,0,0]],np.float32))
    source=json.loads((OUT.parent/'classical_route_reference_diagnostic_v1/source_admission.json').read_text())['source_sha256'].copy()
    for name in ['cartesian_pair_action.py','cartesian_pair_kernel.py','check_cartesian_pair_kernel.py','cartesian_pair_runtime.py','test_cartesian_pair_runtime.py','run_cartesian_delivery.py']:
        file=ROOT/'wheelleg_warp'/name;source[str(file.relative_to(ROOT))]=sha(file)
    assert all(sha(ROOT/f)==h for f,h in source.items())
    atomic_json(OUT/'runtime_admission.json',dict(round=364,verified=True,proposal_sha256=sha(OUT/'runtime_proposal.json'),
        kernel_review_sha256=sha(OUT/'kernel_review.json'),integration_contract_sha256=sha(OUT.parent/'round363_integration_contract.json'),
        source_sha256=source,runtime=runtime(),routing_spy_passed=True,new_physics_steps=0,new_learning_samples=0,
        actual_constructor_topology_required_before_first_launch=True,full_delivery_not_yet_verified=True))


def verify():
    p=inputs();a=json.loads((OUT/'runtime_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'runtime_proposal.json') and a['kernel_review_sha256']==sha(OUT/'kernel_review.json')
    assert a['runtime']==runtime() and all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Need2GiB diagnostic archive space')
    return p,a


def zero_compare(directory,row,baseline):
    old_dir,old_row=baseline
    assert row['physical_steps']==old_row['physical_steps'] and flags(row)==flags(old_row)
    for field in ('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace','joint_guard_trace','actor_trace'):
        with np.load(directory/row[field]['path']) as x,np.load(old_dir/old_row[field]['path']) as y:
            assert set(x.files)==set(y.files)
            for name in x.files:np.testing.assert_array_equal(x[name],y[name],err_msg=f'Zero mismatch {field}/{name}')


def run():
    with (ROOT/'.git/project-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,a=verify();QUEUE.mkdir(exist_ok=False)
        torch.set_num_threads(1);records=[];steps=0;actor_rows=0;fd_calls=[];holder={};initial={};baselines={};current=None
        started=time.perf_counter();old=evaluation.make_env;graph=wp.capture_launch;fd=mujoco.mjd_transitionFD
        def capture(*args,**kwargs):
            nonlocal steps
            assert steps+40<=p['actual_graph_world_step_budget'];steps+=40;return graph(*args,**kwargs)
        def counted(*args,**kwargs):
            assert len(fd_calls)<p['constructor_FD_budget'];fd_calls.append(float(args[2]));return fd(*args,**kwargs)
        def factory(cases,arm,normalization,directory):
            assert len(cases)==1 and normalization is None
            def build():
                result=old(cases,arm,normalization,directory);holder.update(raw=result[0],history=result[2]);return result
            guarded=lambda:guard.instrument(build,cases,directory)
            result=guarded() if current['arm']=='B0' else cart.instrument(guarded,cases,directory,0 if current['arm']=='CartLive3' else 1)
            raw,_,history,_,env=result;holder['guard_chunks']=[]
            reset,async_,wait=env.reset,raw.step_async,raw.step_wait
            def reset_all():
                value=reset();snap=dict(q=raw.data.qpos.numpy(),v=raw.data.qvel.numpy(),ctrl=raw.data.ctrl.numpy(),time=raw.data.time.numpy(),
                    warm=raw.data.qacc_warmstart.numpy(),sensors=raw.data.sensordata.numpy(),param=raw.param.numpy(),reference=raw.k['reference'].numpy(),
                    controller_memory=raw.k['state'].numpy(),guard_memory=raw._joint_guard_buffers[0].numpy(),frames=history.frames.copy(),inputs=history.inputs.copy(),
                    elapsed=history.elapsed.copy(),valid=history.valid.copy(),geom_xpos=raw.data.geom_xpos.numpy(),geom_xmat=raw.data.geom_xmat.numpy())
                receive_initial(initial,cases[0]['seed'],snap);np.savez_compressed(directory/'initial.npz',**snap)
                if hasattr(raw,'_cartesian_buffers'):
                    c=raw._cartesian_buffers
                    assert not c['latent'].numpy().any() and not c['shadow'].numpy().any()
                    np.savez_compressed(directory/'cartesian_initial.npz',latent=c['latent'].numpy(),shadow=c['shadow'].numpy(),reference=c['reference'].numpy())
                return value
            def step_async(action):
                nonlocal actor_rows
                assert raw.state.numpy()[0,0]+40<=p['per_job_first_episode_steps_cap'] and actor_rows+1<=p['actor_row_budget']
                actor_rows+=1;return async_(action)
            def step_wait():
                value=wait();g=raw._joint_guard_buffers[1].numpy()[:,0];holder['guard_chunks'].append(g[g[:,0]==1].copy())
                t=raw._complete_buffers[0].numpy()[:,0];keep=t[:,0]==1
                pre=raw._complete_buffers[4].numpy()[:,0][keep];post=raw._complete_buffers[5].numpy()[:,0][keep]
                if len(post):
                    lengths,_=chain_minima(post[:,:17],raw.cpu);margin,actual,issued=physical_metrics(pre,post,raw.cpu)
                    assert min(lengths)>=HEIGHT_115_GEOMETRIC_MIN and margin>=0 and max(actual,issued)<=1e-6 and t[keep,24].min()>=0,'Recorded physical/design violation'
                return value
            env.reset,raw.step_async,raw.step_wait=reset_all,step_async,step_wait
            return result
        def preserve(directory):
            if 'raw' not in holder:return
            raw=holder['raw'];h=holder['history'];cart.preserve(raw,directory)
            extra={}
            if hasattr(raw,'_joint_guard_buffers'):extra=dict(guard_memory=raw._joint_guard_buffers[0].numpy(),guard_last=raw._joint_guard_buffers[1].numpy())
            np.savez_compressed(directory/'terminal.npz',current_q=raw.data.qpos.numpy(),current_v=raw.data.qvel.numpy(),terminal_q=raw.stopped_q.numpy(),terminal_v=raw.stopped_v.numpy(),
                state=raw.state.numpy(),controller_memory=raw.k['state'].numpy(),frames=h.frames,inputs=h.inputs,**extra)
            if holder.get('guard_chunks'):np.savez_compressed(directory/'guard_prefix.npz',trace=np.concatenate(holder['guard_chunks']))
        evaluation.make_env=factory;wp.capture_launch=capture;mujoco.mjd_transitionFD=counted;directory=QUEUE
        atomic_json(QUEUE/'started.json',dict(pid=os.getpid(),proposal_sha256=sha(OUT/'runtime_proposal.json'),admission_sha256=sha(OUT/'runtime_admission.json')))
        try:
            for current in p['conditions']:
                verify();holder.clear();directory=QUEUE/current['label'];directory.mkdir(exist_ok=False)
                actor=None if current['arm']=='B0' else Scripted(current['profile']);before=steps;before_rows=actor_rows
                result,checked=evaluation.evaluate([current['case']],'B0' if actor is None else 'H1',actor,None,directory)
                preserve(directory);row=result['runs'][0];key=current['case']['seed'];equal_zero=None
                assert result['physical']==result['design']==1
                with np.load(directory/row['actor_trace']['path']) as z:trace=z['trace']
                assert len(trace)==actor_rows-before_rows==int(np.ceil(row['physical_steps']/40))
                if actor is not None:
                    expected=Scripted(current['profile'])
                    for record in trace:np.testing.assert_array_equal(record[962:],expected.predict(record[None,:481])[0][0])
                if current['arm']=='B0':baselines[key]=(directory,row)
                elif current['profile']=='zero':zero_compare(directory,row,baselines[key]);equal_zero=True
                records.append(dict(condition=current['label'],arm=current['arm'],profile=current['profile'],case=key,path=str((directory/'result.json').relative_to(QUEUE)),
                    sha256=sha(directory/'result.json'),success=row['success'],physical=result['physical'],design=result['design'],zero_exact=equal_zero,
                    actual_graph_world_steps=steps-before,first_episode_world_steps=row['physical_steps'],actor_rows=actor_rows-before_rows,
                    checked=checked,initial_sha256=sha(directory/'initial.npz'),terminal_sha256=sha(directory/'terminal.npz')))
                atomic_json(QUEUE/'progress.json',dict(records=records,evaluations=len(records),actual_graph_world_steps=steps,actor_rows=actor_rows,FD_calls=fd_calls))
                print('DELIVERY364',current['label'],'task',row['success'],'zero_exact',equal_zero,flush=True)
            verify();assert len(records)==20 and sum(r['zero_exact'] is True for r in records)==8
            atomic_json(QUEUE/'completion.json',dict(round=364,verified=True,evaluations=20,records=records,actual_graph_world_steps=steps,
                first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),actor_rows=actor_rows,FD_calls=fd_calls,wall_seconds=time.perf_counter()-started,
                proposal_sha256=sha(OUT/'runtime_proposal.json'),admission_sha256=sha(OUT/'runtime_admission.json'),new_learning_samples=0,new_model_forward_rows=0,formal5_admitted=False))
        except BaseException as error:
            preservation_error=None
            try:preserve(directory)
            except BaseException as saving:preservation_error=repr(saving)
            atomic_json(QUEUE/'failure.json',dict(error=repr(error),preservation_error=preservation_error,condition=current,records=records,actual_graph_world_steps=steps,
                actor_rows=actor_rows,FD_calls=fd_calls,partial_raw_available='raw' in holder,implicit_retry=False));raise
        finally:evaluation.make_env=old;wp.capture_launch=graph;mujoco.mjd_transitionFD=fd


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.command]()
