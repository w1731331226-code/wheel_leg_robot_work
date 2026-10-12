"""Prospective584 first-episode classical queue; refuses missing resource admission."""
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import mujoco
import warp as wp
import cartesian_classical
import cartesian_pair_runtime as cart
import execution_history_evaluation as evaluation
import joint_state_guard as guard
from cartesian_pair_kernel_v2 import deliver_accepted_nominal
from fixed_reference_force import embed
from run_cartesian_repair import repaired_instrument,require_accepted_nominal
from run_cartesian_zero_input import blob_hashes,model_values,secondary
from run_wheel_interference import runtime
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
from review_fixed_force_evaluation import chain_minima,physical_metrics
from native.terrain import HEIGHT_115_GEOMETRIC_MIN

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_classical_candidate_v1'


class ResourceNotAdmitted(RuntimeError):pass


def resource_gate(resource,proposal_hash,budget_hash,initial_space=True):
    if resource.get('verified') is not True or resource.get('resource_admitted') is not True or resource.get('archive_strategy_frozen') is not True:
        raise ResourceNotAdmitted('Complete dense evidence resource admission is not verified')
    assert resource['proposal_sha256']==proposal_hash and resource['job_budget_review_sha256']==budget_hash
    path=Path(resource['durable_root'])
    if not path.is_absolute() or not path.is_dir():raise ResourceNotAdmitted('Admitted absolute durable_root is unavailable')
    assert resource['workspace_peak_B']<=4*1024**3 and resource['free_reserve_B']>=2*1024**3
    required=(resource['required_durable_free_B'] if initial_space else 0)+resource['free_reserve_B']
    if shutil.disk_usage(path).free<required:
        raise ResourceNotAdmitted('Durable storage no longer meets the admitted worst-case reserve')
    return path/OUT.name/'runtime'


def verify(committed=True,initial_space=True):
    # Resource rejection precedes runtime queries and all model constructors/FD.
    resource_path=OUT/'resource_admission.json'
    if not resource_path.exists():raise ResourceNotAdmitted('Missing complete dense resource_admission.json')
    resource=json.loads(resource_path.read_text())
    if resource.get('verified') is not True:raise ResourceNotAdmitted('Complete dense resource admission verified=false')
    p=json.loads((OUT/'proposal.json').read_text());budget=json.loads((OUT/'job_budget_review.json').read_text())
    queue=resource_gate(resource,sha(OUT/'proposal.json'),sha(OUT/'job_budget_review.json'),initial_space)
    assert budget['proposal_sha256']==sha(OUT/'proposal.json')
    assert budget['totals']==dict(constructors=68,episodes=584,graph_worldstep_cap=12839760,world_packet_cap=320994)
    assert p['prospective_budget']['constructor_FD_plan']['total_max_calls']==345
    for mapping in (p['source_sha256'],p['input_sha256'],budget['input_sha256'],budget['current_formula_source_sha256']):
        assert all(sha(ROOT/n)==h for n,h in mapping.items())
    a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json')
    assert a['job_budget_review_sha256']==sha(OUT/'job_budget_review.json')
    assert a['resource_admission_sha256']==sha(resource_path) and a['runtime']==runtime()
    required=('run_cartesian_classical_baseline.py','test_cartesian_classical_baseline.py','review_cartesian_classical_baseline.py')
    assert all('wheelleg_warp/'+n in a['source_sha256'] for n in required)
    assert all(sha(ROOT/n)==h for n,h in a['source_sha256'].items())
    if committed:
        expected={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ('proposal.json','job_budget_review.json','resource_admission.json','source_admission.json')}
        expected.update(a['source_sha256'])
        names=list(expected);query=''.join('HEAD:'+n+'\n' for n in names).encode()
        reply=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=query,capture_output=True,check=True)
        assert blob_hashes(reply.stdout,names)==expected
    return p,budget,a,resource,queue


class Budget:
    def __init__(self):self.steps=0;self.FD=[];self.constructors=0;self.job=None
    def open(self,descriptor):
        assert self.job is None and self.constructors<68
        self.job=dict(descriptor=descriptor,raw=None,graph=None,calls=0,fd_start=len(self.FD),fd_limit=10 if self.constructors==0 else 5)
    def fd(self,original,*args,**kwargs):
        j=self.job
        if j is None or len(self.FD)-j['fd_start']>=j['fd_limit'] or len(self.FD)>=345:
            raise RuntimeError('Unplanned per-constructor FD call')
        self.FD.append(float(args[2]));return original(*args,**kwargs)
    def bind(self,raw):
        j=self.job;d=j['descriptor'];assert j['raw'] is None and raw.num_envs==d['num_envs']<=10
        assert len(self.FD)-j['fd_start']==j['fd_limit'],'Actual FD construction path differs from frozen fresh-process plan'
        check_actual(raw,d)
        j.update(raw=raw,graph=raw.graph);self.constructors+=1
    def capture(self,original,*args,**kwargs):
        graph=args[0] if args else kwargs['graph'];j=self.job
        if j is None or j['raw'] is None or graph is not j['graph'] or j['raw'].graph is not graph:
            raise RuntimeError('Foreign, unbound or replaced graph')
        d=j['descriptor'];amount=40*j['raw'].num_envs
        if j['calls']>=d['deadline_packets'] or (j['calls']+1)*amount>d['graph_worldstep_cap_per_constructor'] or self.steps+amount>12839760:
            raise RuntimeError('Frozen per-job/global graph deadline exhausted')
        j['calls']+=1;self.steps+=amount;return original(*args,**kwargs)
    def close(self):
        j=self.job;self.job=None
        return dict(graph_calls=j['calls'],actual_graph_world_steps=40*j['descriptor']['num_envs']*j['calls'],FD_calls=self.FD[j['fd_start']:])


def check_actual(raw,descriptor):
    param=raw.param.numpy();rows=descriptor['cases']
    assert raw.num_envs==len(rows) and int(raw.cpu.opt.iterations)==int(raw.model.opt.iterations)==descriptor['solver_iterations']
    assert raw.data.naconmax==64*raw.num_envs,'Actual contact capacity differs from the admitted resource bound'
    np.testing.assert_array_equal(param[:,0],[r['speed'] for r in rows])
    np.testing.assert_allclose(param[:,2],[r['task_goal_progress_m_archived'] for r in rows],rtol=0,atol=1e-12)
    np.testing.assert_allclose(param[:,3],[r['param3_s_arithmetic'] for r in rows],rtol=0,atol=1e-12)
    assert int(np.ceil((param[:,3].max()+2)/.02))+2==descriptor['deadline_packets']
    np.testing.assert_array_equal(param[:,4],[round(s.delay_ms*2) for s in raw.scenarios])
    assert param[:,4].max()*.5==descriptor['max_delay_ms']<=10
    assert raw._cartesian_topology['counts']['control']==80 and raw._joint_guard_topology['captured_controller_calls']==80
    assert raw._complete_topology['physics_calls']==40 and raw._execution_topology['control_steps']==40
    assert raw._parking_contract['condition']=='original','Classical protocol requires original B1 parking(enabled0)'
    require_accepted_nominal(raw)


def state_snapshot(raw,history):
    result=dict(q=raw.data.qpos.numpy(),v=raw.data.qvel.numpy(),ctrl=raw.data.ctrl.numpy(),public=raw.obs.numpy(),
        controller_memory=raw.k['state'].numpy(),guard_memory=raw._joint_guard_buffers[0].numpy(),
        frames=history.frames.copy(),inputs=history.inputs.copy(),elapsed=history.elapsed.copy(),valid=history.valid.copy(),
        previous_sensor_steps=raw._execution_previous_sensor_steps.copy())
    result.update({n:b.numpy() for n,b in zip(('total','prefix','stamp'),raw._execution_buffers)})
    result.update({'cart_'+n:raw._cartesian_buffers[n].numpy() for n in ('latent','shadow','reference')})
    return result


class Actor:
    num_timesteps=0;_n_updates=0
    def __init__(self,cases,condition,yaw):
        self.cases=cases;self.condition=condition;self.yaw=yaw;self.pending=np.ones(len(cases),bool)
        self.parts=[[] for _ in cases];self.terminals={};self.resets={};self.policy=SimpleNamespace(state_dict=lambda:{})
    def predict(self,obs,deterministic=True):
        assert deterministic and obs.shape==(len(self.cases),481)
        latent,diagnostic=cartesian_classical.actions(obs,self.condition['gamma'],self.yaw)
        for w in np.flatnonzero(self.pending):self.parts[w].append({n:v[w].copy() for n,v in diagnostic.items()})
        return embed(latent,'D3',len(self.cases)),None
    def received(self,value,raw=None,history=None):
        worlds=np.flatnonzero(value[2]&self.pending)
        saved=state_snapshot(raw,history) if len(worlds) and raw is not None else None
        for w in worlds:
            self.terminals[int(w)]=value[3][w]['terminal_observation'].copy();self.pending[w]=False
            if saved is not None:self.resets[int(w)]={n:a[w].copy() for n,a in saved.items()}
    def save(self,directory):
        for w,part in enumerate(self.parts):
            if part:np.savez_compressed(directory/f'classical_request_{w}.npz',**{n:np.stack([r[n] for r in part]) for n in part[0]})
        for w,value in self.terminals.items():np.savez_compressed(directory/f'actual_terminal_history_{w}.npz',actual_raw481=value)
        for w,value in self.resets.items():np.savez_compressed(directory/f'actual_after_reset_{w}.npz',**value)


def preserve(holder,actor,directory,cases,errors):
    if 'raw' not in holder:return
    raw=holder['raw'];history=holder['history']
    for label,action in (
        ('original_dense',lambda:evaluation.recorder.preserve(raw,directory,cases)),
        ('parking',lambda:evaluation.parking.preserve(raw,directory)),
        ('command',lambda:evaluation.collector.preserve(raw,directory/'last_command_buffers.npz')),
        ('cartesian',lambda:cart.preserve(raw,directory)),
        ('history',lambda:np.savez_compressed(directory/'last_history.npz',frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)),
        ('classical_actual_terminal',lambda:actor.save(directory))):secondary(errors,directory.name+'/'+label,action)


def measure(cases,condition,descriptor,p,global_budget,directory,errors):
    holder={};actor=Actor(cases,condition,p['parameters']['yaw_config']);original=evaluation.make_env
    def factory(cases_,arm,normalization,directory_):
        assert cases_==cases and arm=='B1-route' and normalization is None and directory_==directory
        def base():
            value=original(cases,'B1-route',None,directory);holder.update(raw=value[0],history=value[2]);return value
        guarded=lambda:guard.instrument(base,cases,directory,dense=True)
        result=repaired_instrument(guarded,cases,directory,0 if condition['chart']=='CartLive3' else 1)
        raw,route,history,norm,env=result;global_budget.bind(raw)
        assert all(all(getattr(s,n)==v for n,v in case['scenario'].items()) for s,case in zip(raw.scenarios,cases))
        arrays,scalars=model_values(raw);np.savez_compressed(directory/'actual_model_before_launch.npz',**arrays)
        atomic_json(directory/'actual_model_before_launch.json',dict(scalars=scalars,ids=raw.ids.numpy().tolist(),descriptor=descriptor))
        reset=env.reset;wait=history.step_wait;raw_wait=raw.step_wait
        def reset_all():
            obs=reset();np.savez_compressed(directory/'initial.npz',raw481=obs,**state_snapshot(raw,history))
            np.savez_compressed(directory/'cartesian_initial.npz',**{n:b.numpy() for n,b in raw._cartesian_buffers.items()});return obs
        def waited():
            value=wait();actor.received(value,raw,history);return value
        def physical_wait():
            value=raw_wait();keep=raw._complete_buffers[0].numpy()[:,:,0]==1
            before,after=raw._complete_buffers[4].numpy()[keep],raw._complete_buffers[5].numpy()[keep]
            if len(after):
                length,_=chain_minima(after[:,:17],raw.cpu);margin,actual,issued=physical_metrics(before,after,raw.cpu)
                assert min(length)>=HEIGHT_115_GEOMETRIC_MIN and margin>=0 and max(actual,issued)<=1e-6
                assert raw._complete_buffers[0].numpy()[:,:,24][keep].min()>=0
            return value
        env.reset=reset_all;history.step_wait=waited;raw.step_wait=physical_wait
        return result
    try:
        with patch.object(evaluation,'make_env',factory),patch.object(cart,'deliver',deliver_accepted_nominal):
            result,checked=evaluation.evaluate(cases,'B1-route',actor,None,directory)
        actor.save(directory)
        assert len(result['runs'])==result['physical']==result['design']==len(cases)
        assert all(not x for x in actor.pending) and checked['evaluations']==len(cases)
        for w,row in enumerate(result['runs']):
            assert len(actor.parts[w])==int(np.ceil(row['physical_steps']/40))
        return result,checked
    finally:preserve(holder,actor,directory,cases,errors)


def aggregate(parts):
    from train_height_comparison import summary
    runs=[r for part in parts for r in part['runs']]
    return dict(runs=runs,summary=summary(runs),physical=sum(r['physical_safety_passed'] for r in runs),design=sum(r['design_joint_passed'] for r in runs))


def job_record(queue,directory,condition,item,phase,result,checked,used,selection_hash=None):
    entry=dict(phase=phase,condition=condition['label'],chart=condition['chart'],gamma=condition['gamma'],panel=item['panel'],batch=item['batch'],indices=item['indices'],cases=item['cases'],
        result_path=str((directory/'result.json').relative_to(queue)),result_sha256=sha(directory/'result.json'),provider_checked=checked,
        first_episode_world_steps=sum(r['physical_steps'] for r in result['runs']),
        raw_artifacts={str(f.relative_to(queue)):sha(f) for f in directory.iterdir() if f.is_file()},
        original_B1_parking='original_enabled0',independent_raw_qualified=False,**used)
    if selection_hash is not None:entry['selection_review_sha256']=selection_hash
    return entry


def run():
    with (ROOT/'.git/project-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,plan,admission,resource,queue=verify(True)
        assert not queue.exists() and not (OUT/'execution_locator.json').exists()
        from rm_controller import nominal_design_115
        assert nominal_design_115.cache_info().currsize==0,'A fresh construction process is required'
        queue.mkdir(parents=True);atomic_json(OUT/'execution_locator.json',dict(runtime_path=str(queue),proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'source_admission.json')))
        atomic_json(queue/'started.json',dict(runtime_path=str(queue),proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'source_admission.json'),implicit_retry=False))
        budget=Budget();records=[];errors=[];selection_parts={c['label']:[] for c in p['conditions']};start=time.perf_counter();current=None
        original_graph,original_fd=wp.capture_launch,mujoco.mjd_transitionFD
        def job(condition,item,phase,selection_hash=None):
            nonlocal current
            assert verify(True,initial_space=False)==(p,plan,admission,resource,queue),'Frozen queue inputs changed before constructor'
            key=(phase,item['panel'],item['batch']);matches=[d for d in plan['jobs'] if (d['phase'],d['panel'],d['batch'])==key]
            assert len(matches)==1;descriptor=matches[0];assert [r['seed'] for r in item['cases']]==[r['seed'] for r in descriptor['cases']]
            current=phase+'_'+condition['label']+'_'+item['panel']+'_'+str(item['batch']);directory=queue/current;directory.mkdir()
            budget.open(descriptor);result,checked=measure(item['cases'],condition,descriptor,p,budget,directory,errors)
            if errors:raise RuntimeError('Raw evidence preservation failed')
            entry=job_record(queue,directory,condition,item,phase,result,checked,budget.close(),selection_hash)
            records.append(entry);atomic_json(queue/'progress.json',dict(records=records,actual_graph_world_steps=budget.steps,
                first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),FD_calls=budget.FD))
            if shutil.disk_usage(queue).free<resource['free_reserve_B']:raise ResourceNotAdmitted('Durable free reserve exhausted')
            return result
        try:
            with patch.object(wp,'capture_launch',lambda *a,**k:budget.capture(original_graph,*a,**k)),patch.object(mujoco,'mjd_transitionFD',lambda *a,**k:budget.fd(original_fd,*a,**k)):
                for condition in p['conditions']:
                    for item in p['selection_jobs']:selection_parts[condition['label']].append(job(condition,item,'selection'))
                from review_cartesian_classical_baseline import selection_decision
                choice=selection_decision(p,{name:aggregate(parts) for name,parts in selection_parts.items()})
                assert choice['verified'] and len(choice['selected'])==2
                choice.update(proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'source_admission.json'),reviewer_sha256=sha(ROOT/'wheelleg_warp/review_cartesian_classical_baseline.py'))
                atomic_json(queue/'selection_review.json',choice);selection_hash=sha(queue/'selection_review.json')
                for selected in choice['selected']:
                    condition=next(c for c in p['conditions'] if c['label']==selected['label'])
                    for item in p['eval_jobs']:job(condition,item,'evaluation',selection_hash)
            assert budget.constructors==len(records)==68 and sum(len(p['selection_jobs'][r['batch']]['cases']) if r['phase']=='selection' else len(next(j['cases'] for j in p['eval_jobs'] if (j['panel'],j['batch'])==(r['panel'],r['batch']))) for r in records)==584
            assert verify(True,initial_space=False)==(p,plan,admission,resource,queue)
            atomic_json(queue/'completion.json',dict(verified=True,status='producer_completed_pending_independent_raw_receipt',runtime_path=str(queue),episodes=584,records=records,
                actual_graph_world_steps=budget.steps,first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),FD_calls=budget.FD,proposal_sha256=sha(OUT/'proposal.json'),source_admission_path=str(OUT/'source_admission.json'),
                source_admission_sha256=sha(OUT/'source_admission.json'),selection_review_sha256=selection_hash,secondary_errors=errors,wall_seconds=time.perf_counter()-start,
                independent_raw_qualified=False,new_learning_samples=0,new_model_forward_rows=0,new_optimizer_updates=0,formal5_admitted=False))
        except BaseException as error:
            failure=dict(status='failed',current_job=current,primary_error=repr(error),secondary_errors=errors,records=records,actual_graph_world_steps=budget.steps,FD_calls=budget.FD,implicit_retry=False)
            secondary(errors,'failure_manifest',lambda:atomic_json(queue/'failure.json',failure));raise


if __name__=='__main__':run()
