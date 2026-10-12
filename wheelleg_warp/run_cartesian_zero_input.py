"""One approved matched-construction three-packet input diagnostic; no retry."""
from dataclasses import fields,is_dataclass
import fcntl
import hashlib
import io
import json
import os
import subprocess
import time
from unittest.mock import patch
import numpy as np
import mujoco
import warp as wp
import train_height_comparison as base
import execution_input_collector as collector
import light_parking_request as parking
import joint_state_guard as guard
import cartesian_light_runtime as cart
import cartesian_zero_input_trace as observer
from execution_history_env import ExecutionHistory
from route_state import RouteState
from run_cartesian_light_interface import BASE,snapshot,physical
from run_wheel_interference import runtime
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=BASE/'cartesian_zero_input_diagnostic_v1'
QUEUE=OUT/'runtime'


def inputs():
    p=json.loads((OUT/'proposal.json').read_text());d=json.loads((BASE/'round370_direction_review.json').read_text())
    assert d['supplement']['direction_admitted'] and d['supplement']['proposal_sha256']==sha(OUT/'proposal.json')
    assert p['conditions']==[dict(label='B0_repeat1',arm='B0'),dict(label='B0_repeat2',arm='B0'),dict(label='CartLive3',arm='CartLive3')]
    assert p['worlds']==10 and p['actor_packets_per_condition']==1 and p['actual_graph_world_step_budget']==1200 and p['constructor_FD_budget']==80
    assert p['new_learning_samples']==p['new_model_forward_rows']==0 and not p['runtime_admitted']
    for mapping in (p['source_sha256'],json.loads((BASE/'cartesian_training_interface_v1/source_admission.json').read_text())['source_sha256']):
        assert all(sha(ROOT/n)==h for n,h in mapping.items())
    assert sha(ROOT/p['bank_source'])==p['bank_source_sha256']
    assert sha(BASE/'cartesian_training_interface_v1/runtime/failure.json')==p['parent_failure_sha256']
    assert sha(BASE/'cartesian_training_interface_v1/numerical_contract_review.json')==p['numerical_contract_review_sha256']
    return p


def blob_hashes(output,names):
    stream=io.BytesIO(output);result={}
    for name in names:
        header=stream.readline().split()
        if len(header)!=3 or header[1]!=b'blob':raise ValueError('Missing committed source: '+name)
        count=int(header[2]);assert 0<=count<=len(output)
        value=stream.read(count);assert len(value)==count and stream.read(1)==b'\n'
        result[name]=hashlib.sha256(value).hexdigest()
    assert stream.read()==b''
    return result


def verify(committed=False):
    p=inputs();a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json') and a['runtime']==runtime()
    assert a['direction_review_sha256']==sha(BASE/'round370_direction_review.json')
    assert all(sha(ROOT/n)==h for n,h in a['source_sha256'].items())
    if committed:
        expected={str((OUT/'source_admission.json').relative_to(ROOT)):sha(OUT/'source_admission.json'),**a['source_sha256']}
        names=list(expected);query=''.join('HEAD:'+n+'\n' for n in names).encode()
        result=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=query,capture_output=True,check=True)
        assert blob_hashes(result.stdout,names)==expected,'Committed source differs from admitted working source'
    return p,a


def freeze():
    inputs();assert not QUEUE.exists() and not (OUT/'source_admission.json').exists()
    from test_cartesian_zero_input import run as check
    check()
    sources=json.loads((BASE/'cartesian_training_interface_v1/source_admission.json').read_text())['source_sha256'].copy()
    for name in ('review_cartesian_repair.py','cartesian_zero_input_trace.py','run_cartesian_zero_input.py','test_cartesian_zero_input.py'):
        path=ROOT/'wheelleg_warp'/name;sources[str(path.relative_to(ROOT))]=sha(path)
    atomic_json(OUT/'source_admission.json',dict(round=371,verified=True,source_sha256=sources,runtime=runtime(),
        proposal_sha256=sha(OUT/'proposal.json'),direction_review_sha256=sha(BASE/'round370_direction_review.json'),
        graph_identity_worlds_and_once_bound=True,all_source_blobs_committed_before_launch_required=True,
        primary_secondary_failures_separate=True,new_physics_steps=0,new_learning_samples=0,runtime_not_yet_verified=True))


class Budget:
    def __init__(self,p):self.p=p;self.steps=0;self.FD=[];self.raw=None;self.graph=None;self.worlds=None;self.used=[]
    def bind(self,raw):
        assert raw.num_envs==raw.data.qpos.shape[0]==self.p['worlds']
        assert not any(raw.graph is g for g in self.used)
        self.raw=raw;self.graph=raw.graph;self.worlds=raw.num_envs
    def capture(self,original,*args,**kwargs):
        graph=args[0] if args else kwargs['graph'];raw=self.raw
        if raw is None or graph is not self.graph or raw.graph is not self.graph or any(graph is g for g in self.used):raise RuntimeError('Unregistered or repeated graph')
        assert raw.num_envs==raw.data.qpos.shape[0]==self.worlds==self.p['worlds']
        amount=40*self.worlds
        if self.steps+amount>self.p['actual_graph_world_step_budget']:raise RuntimeError('Graph budget exhausted')
        self.steps+=amount;self.used.append(graph);return original(*args,**kwargs)
    def fd(self,original,*args,**kwargs):
        if len(self.FD)>=self.p['constructor_FD_budget']:raise RuntimeError('FD budget exhausted')
        self.FD.append(float(args[2]));return original(*args,**kwargs)


def model_values(raw):
    arrays={};scalars={}
    def collect(value,name):
        if hasattr(value,'numpy'):arrays[name]=value.numpy()
        elif is_dataclass(value):
            for f in fields(value):collect(getattr(value,f.name),name+'.'+f.name)
        elif isinstance(value,(tuple,list)):
            for i,v in enumerate(value):collect(v,name+'.'+str(i))
        elif isinstance(value,(bool,int,float,str)):scalars[name]=value
    collect(raw.model,'model')
    for name,value in dict(q0=raw.q0,obs0=raw.obs0,param=raw.param,reference=raw.k['reference'],
        gains=raw.k['gains'],feed=raw.k['feed'],angles=raw.k['angles'],geom_xpos=raw.data.geom_xpos,geom_xmat=raw.data.geom_xmat).items():
        arrays[name]=value.numpy()
    return arrays,scalars


def build(p,arm):
    bank=json.loads((ROOT/p['bank_source']).read_text());config=dict(training_banks=bank['training_banks'],curriculum_milestones=[4000,8000])
    original=base.raw_env;constructed=[]
    def factory(rows,mode):
        standard=lambda:collector.instrument(lambda a,b:parking.instrument(original,a,b),rows,mode)
        guarded=lambda:guard.instrument(lambda:(standard(),),rows,None,dense=False)
        plain=lambda:guarded()[0]
        candidate=lambda:cart.instrument(guarded,rows,0)[0]
        raw=observer.instrument(candidate if arm=='CartLive3' else plain,rows,arm=='CartLive3')
        values,scalars=model_values(raw);constructed.append((values,scalars))
        return raw
    with patch.object(base,'raw_env',factory):env=base.CurriculumEnv(config,'virtual6',33531,10,milestones=[4000,8000])
    assert len(constructed)==3 and env.policy_steps==0 and np.all(env.stages==1)
    raw=env.venv;return env,raw,ExecutionHistory(RouteState(env),raw,'H1'),constructed


def differences(a,b):
    assert set(a)==set(b)
    result={}
    for name in a:
        assert a[name].shape==b[name].shape and a[name].dtype==b[name].dtype
        mask=a[name]!=b[name]
        if mask.any():
            indices=np.argwhere(mask);result[name]=dict(count=len(indices),first=indices[0].tolist(),max_abs=float(abs(a[name].astype(float)-b[name].astype(float)).max()))
    return result


def classify(baseline,current):
    a,b=baseline['pre'],current['pre'];edges=current['boundaries'];old=baseline['boundaries']
    for slot in range(40):
        if not np.array_equal(a[slot],b[slot]):return dict(status='input_chain_divergence',slot=slot,phase_differences=differences({'pre':a[slot]},{'pre':b[slot]}))
        if not np.array_equal(edges[slot,:,0,2],edges[slot,:,1,2]):return dict(status='Cart_boundary_memory5_error',slot=slot,boundary='clear_native')
        if not np.array_equal(old[slot,:,2,:4],edges[slot,:,2,:4]):return dict(status='input_chain_divergence',slot=slot,boundary='Nom_before')
        if not np.array_equal(old[slot,:,3,2],edges[slot,:,3,2]):return dict(status='Nom_execution_context_error',slot=slot)
        if not np.array_equal(edges[slot,:,4,2],edges[slot,:,5,2]):return dict(status='Cart_boundary_memory5_error',slot=slot,boundary='deliver')
    return dict(status='inconclusive_not_reproduced_under_instrumentation')


class Stop(RuntimeError):
    def __init__(self,result):self.result=result;super().__init__(result['status'])


def secondary(errors,label,action):
    try:action()
    except BaseException as error:errors.append(dict(stage=label,error=repr(error)))


def save_buffers(raw,history,directory,errors):
    secondary(errors,directory.name+'/common',lambda:np.savez_compressed(directory/'preserved_common.npz',**snapshot(raw,history)))
    secondary(errors,directory.name+'/observer',lambda:np.savez_compressed(directory/'preserved_observer.npz',**{k:b.numpy() for k,b in raw._zero_input_buffers.items()}))
    if hasattr(raw,'_cartesian_buffers'):
        secondary(errors,directory.name+'/cartesian',lambda:np.savez_compressed(directory/'preserved_cartesian.npz',**{k:b.numpy() for k,b in raw._cartesian_buffers.items()}))
    if raw._zero_input_terminal is not None:
        secondary(errors,directory.name+'/terminal',lambda:np.savez_compressed(directory/'preserved_terminal.npz',**raw._zero_input_terminal))


def run():
    with (ROOT/'.git/project-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,admission=verify(committed=True);QUEUE.mkdir(exist_ok=False)
        budget=Budget(p);reports=[];holders=[];initial=None;constructors=None;baseline=None;base_trace=None;secondary_errors=[];start=time.perf_counter()
        atomic_json(QUEUE/'started.json',dict(round=371,pid=os.getpid(),admission_sha256=sha(OUT/'source_admission.json'),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),implicit_retry=False))
        launch=wp.capture_launch;fd=mujoco.mjd_transitionFD
        try:
            with patch.object(wp,'capture_launch',lambda *a,**k:budget.capture(launch,*a,**k)),\
                 patch.object(mujoco,'mjd_transitionFD',lambda *a,**k:budget.fd(fd,*a,**k)):
                for condition in p['conditions']:
                    verify(committed=True);directory=QUEUE/condition['label'];directory.mkdir()
                    env,raw,history,models=build(p,condition['arm']);holders.append((raw,history,directory));budget.bind(raw)
                    for i,(values,scalars) in enumerate(models):
                        np.savez_compressed(directory/f'constructor_stage{i+1}.npz',**values)
                        atomic_json(directory/f'constructor_stage{i+1}.json',dict(scalars=scalars))
                    if constructors is None:constructors=models
                    else:
                        for (values,scalars),(old_values,old_scalars) in zip(models,constructors):
                            assert scalars==old_scalars and not differences(old_values,values),'Constructor/model mismatch'
                    obs=history.reset();before=snapshot(raw,history);np.savez_compressed(directory/'initial.npz',**before)
                    if initial is None:initial=before
                    else:assert not differences(initial,before),'Initial common mismatch'
                    result=history.step(np.zeros((10,6),np.float32));returned,reward,done,infos=result
                    after={**snapshot(raw,history), 'returned':returned,'reward':reward,'done':done}
                    trace={k:b.numpy() for k,b in raw._zero_input_buffers.items()}
                    np.savez_compressed(directory/'zero_prefix.npz',**after);np.savez_compressed(directory/'input_trace.npz',**trace)
                    np.savez_compressed(directory/'terminal_before_wait.npz',**raw._zero_input_terminal)
                    if done.any():
                        np.savez_compressed(directory/'terminal_history.npz',terminal=np.stack([info.get('terminal_observation',np.zeros(481)) for info in infos]))
                        atomic_json(directory/'terminal_infos.json',dict(infos=[{k:v for k,v in info.items() if k!='terminal_observation'} for info in infos]))
                    atomic_json(directory/'topology.json',raw._zero_input_topology)
                    physical(after['state']);assert not done.any() and env.policy_steps==10
                    assert all(np.isfinite(v).all() for v in trace.values())
                    np.testing.assert_array_equal(trace['pre'][:,:,0:2],1)
                    np.testing.assert_array_equal(trace['pre'][:,:,2],np.tile(np.arange(1,41)[:,None],(1,10)))
                    np.testing.assert_array_equal(trace['boundaries'][:,:,:,0:2],1)
                    reports.append(dict(condition=condition['label'],snapshot_sha256=sha(directory/'zero_prefix.npz'),
                        trace_sha256=sha(directory/'input_trace.npz'),worlds=10,physical_substeps=int(after['state'][:,37].sum())))
                    if baseline is None:baseline=after;base_trace=trace
                    elif condition['arm']=='B0':
                        diff=differences(baseline,after);input_diff=differences(base_trace,trace)
                        if diff or input_diff:raise Stop(dict(status='baseline_repeat_failure',common_differences=diff,input_trace_differences=input_diff))
                    else:
                        result=classify(base_trace,trace);diff=differences(baseline,after)
                        if result['status']=='inconclusive_not_reproduced_under_instrumentation' and diff:result=dict(status='other_common_zero_mismatch',common_differences=diff)
                        result['common_differences']=diff;reports[-1]['diagnosis']=result
                        if result['status']!='inconclusive_not_reproduced_under_instrumentation':raise Stop(result)
                    history.close()
                    print('ZERO371',condition['label'],'graph',budget.steps,'FD',len(budget.FD),flush=True)
            assert budget.steps==1200
            verify(committed=True)
            atomic_json(QUEUE/'completion.json',dict(round=371,status='inconclusive_not_reproduced_under_instrumentation',reports=reports,
                graph_world_steps=budget.steps,FD_calls=budget.FD,wall_seconds=time.perf_counter()-start,
                new_learning_samples=0,new_model_forward_rows=0,light_interface_admitted=False,implicit_retry=False))
        except BaseException as error:
            for raw,history,directory in holders:
                save_buffers(raw,history,directory,secondary_errors)
                secondary(secondary_errors,directory.name+'/close',history.close)
            failure=dict(round=371,status=error.result['status'] if isinstance(error,Stop) else 'source_physical_or_recording_failure',
                primary_error=repr(error),diagnosis=error.result if isinstance(error,Stop) else None,secondary_errors=secondary_errors,
                reports=reports,graph_world_steps=budget.steps,FD_calls=budget.FD,wall_seconds=time.perf_counter()-start,
                new_learning_samples=0,new_model_forward_rows=0,implicit_retry=False)
            secondary(secondary_errors,'failure_manifest',lambda:atomic_json(QUEUE/'failure.json',failure))
            print('FAILED371',json.dumps(failure,ensure_ascii=False),flush=True)
            raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.command]()
