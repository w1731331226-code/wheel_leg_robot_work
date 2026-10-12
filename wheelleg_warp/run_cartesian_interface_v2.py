"""Registered within-trajectory light10world qualification; sealed semantic prerequisite."""
import fcntl
import json
import os
import shutil
import subprocess
import time
from unittest.mock import patch
import numpy as np
import mujoco
import warp as wp
import run_cartesian_light_interface as old
from run_cartesian_light_interface import BASE,ROOT,sha,Replay,script,physical,public_reference,collector,flush,snapshot
from run_cartesian_zero_input import blob_hashes,model_values,secondary
from run_wheel_interference import runtime
from dashboard.live_env import atomic_json

OUT=BASE/'cartesian_training_interface_v2'
QUEUE=OUT/'runtime'


def inputs():
    p=json.loads((OUT/'proposal.json').read_text());semantic=json.loads((OUT/'semantic_acceptance.json').read_text())
    assert p['arms']==['CartLive3','CartReset3'] and p['worlds']==10 and p['rollout_packets_each']==1200
    assert p['zero_prefix_packets_total']==2 and p['actual_graph_world_step_budget']==960800 and p['constructor_FD_budget']==80
    assert p['new_learning_samples']==p['new_model_forward_rows']==0 and not p['runtime_admitted']
    assert p['bank_seed']==33531 and p['curriculum_milestones']==[4000,8000]
    assert not p['formal5_admitted'] and not p['scientific_training_admitted']
    assert p['parent_closed_route_sha256']==sha(BASE/'round372_route_decision.json')
    bank=json.loads((BASE/'fixed_force_prequalification_v1/proposal.json').read_text())['training_banks']['33531']
    assert p['training_banks']['33531']=={k:v[:10] for k,v in bank.items()}
    assert semantic['verified'] and semantic['proposal_sha256']==sha(OUT/'proposal.json')
    assert semantic['semantic_review_sha256']==sha(OUT/'semantic_review.json')
    for n,h in semantic['outputs_sha256'].items():assert sha(OUT/n)==h
    for mapping in (p['source_sha256'],p['old_failures_retained'],semantic['core_source_sha256']):
        assert all(sha(ROOT/n)==h for n,h in mapping.items())
    assert semantic['checker_sha256']==sha(ROOT/'wheelleg_warp/check_cartesian_control_semantics.py')
    return p


def verify(committed=False):
    p=inputs();a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json') and a['semantic_acceptance_sha256']==sha(OUT/'semantic_acceptance.json')
    assert a['runtime']==runtime() and all(sha(ROOT/n)==h for n,h in a['source_sha256'].items())
    if committed:
        expected={str((OUT/'source_admission.json').relative_to(ROOT)):sha(OUT/'source_admission.json'),**a['source_sha256']}
        names=list(expected);query=''.join('HEAD:'+n+'\n' for n in names).encode()
        r=subprocess.run(['git','cat-file','--batch'],input=query,cwd=ROOT,capture_output=True,check=True)
        assert blob_hashes(r.stdout,names)==expected
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Need2GiB archive space')
    return p,a


def freeze():
    inputs();assert not QUEUE.exists() and not (OUT/'source_admission.json').exists()
    from test_cartesian_interface_v2 import run as check
    check()
    s=json.loads((BASE/'cartesian_zero_input_diagnostic_v1/source_admission.json').read_text())['source_sha256'].copy()
    for n in ('run_cartesian_interface_v2.py','test_cartesian_interface_v2.py','check_cartesian_control_semantics.py'):
        path=ROOT/'wheelleg_warp'/n;s[str(path.relative_to(ROOT))]=sha(path)
    atomic_json(OUT/'source_admission.json',dict(round=374,verified=True,source_sha256=s,runtime=runtime(),
        proposal_sha256=sha(OUT/'proposal.json'),semantic_acceptance_sha256=sha(OUT/'semantic_acceptance.json'),
        graph_identity_actual_worlds_and_1201_packets_bound=True,committed_source_blobs_required=True,
        primary_secondary_and_each_buffer_saved_separately=True,new_physics_steps=0,new_learning_samples=0,
        actual_interface_not_yet_verified=True))


class Budget:
    def __init__(self,p):self.p=p;self.graph_steps=0;self.FD=[];self.bindings=[];self.active=None
    def bind(self,raw):
        assert raw.num_envs==raw.data.qpos.shape[0]==self.p['worlds']
        assert not any(raw.graph is b['graph'] for b in self.bindings)
        assert len(self.bindings)<2
        entry=dict(raw=raw,graph=raw.graph,worlds=raw.num_envs,calls=0);self.bindings.append(entry);self.active=entry
    def capture(self,original,*args,**kwargs):
        graph=args[0] if args else kwargs['graph'];b=self.active
        if b is None or graph is not b['graph'] or b['raw'].graph is not b['graph']:raise RuntimeError('Unregistered or replaced graph')
        assert b['raw'].num_envs==b['raw'].data.qpos.shape[0]==b['worlds']==self.p['worlds']
        if b['calls']>=1201 or self.graph_steps+40*b['worlds']>self.p['actual_graph_world_step_budget']:raise RuntimeError('Graph packet/worldstep budget exhausted')
        b['calls']+=1;self.graph_steps+=40*b['worlds'];return original(*args,**kwargs)
    def fd(self,original,*args,**kwargs):
        if len(self.FD)>=self.p['constructor_FD_budget']:raise RuntimeError('FD budget exhausted')
        self.FD.append(float(args[2]));return original(*args,**kwargs)


def finish(errors,primary,directory,records,blocks,episodes,env,events,last_actor):
    pending=[]
    if last_actor is not None:pending.append(('last_actor',lambda:np.savez_compressed(directory/'last_submitted_actor.npz',**last_actor)))
    pending.extend([('pending_blocks',lambda:flush(directory,records,blocks)),
        ('episodes',lambda:atomic_json(directory/'episodes.json',dict(episodes=episodes,transitions=env.transition_log,events=events,blocks=blocks)))])
    first=None
    for label,action in pending:
        try:action()
        except BaseException as error:
            errors.append(dict(stage=directory.name+'/'+label,error=repr(error)))
            if first is None:first=error
    if primary is None and first is not None:raise first


def check_bank(raw,env,directory,label):
    values={'model.'+name:getattr(raw.model,name).numpy() for name in env.cache[1]['model']}
    values['meaninertia']=raw.model.stat.meaninertia.numpy()
    np.savez_compressed(directory/('bank_'+label+'.npz'),stages=env.stages,**values)
    for stage in np.unique(env.stages):
        mask=env.stages==stage;cache=env.cache[int(stage)]
        for name,value in values.items():
            expected=cache['meaninertia'] if name=='meaninertia' else cache['model'][name[6:]]
            np.testing.assert_array_equal(value[mask],expected[mask])


def rollout(p,env,raw,history,obs,directory,budget,errors):
    records=[];blocks=[];episodes=[];replay=Replay(raw.obs.numpy());ticks=np.zeros(10,np.int64)
    events=dict(partial_resets=0,stage2_rows=0,stage3_rows=0,positive_delay_rows=0)
    primary=None;last_actor=None
    try:
        for k in range(1200):
            actions=script(ticks);stages=env.stages.copy();reference=raw._cartesian_buffers['reference'].numpy()
            last_actor=dict(packet=obs,actions=actions,ticks=ticks.copy(),stages=stages)
            new,reward,done,infos=history.step(actions);saved=raw._cartesian_last
            after={name:raw._cartesian_buffers[name].numpy() for name in ('latent','reference','shadow')}
            intervals=raw._execution_intervals
            record=dict(actor=obs,action=actions,ticks=ticks.copy(),returned=new,reward=reward,
                stage_before=stages,stage_after=env.stages.copy(),reference_before=reference,**saved,
                latent_after=after['latent'],reference_after=after['reference'],shadow_after=after['shadow'],
                command_integral=intervals['command_integral'],sensor_elapsed_s=intervals['sensor_elapsed_s'],
                public_after=raw.obs.numpy(),terminal_history=np.stack([info.get('terminal_observation',np.zeros(481,np.float32)) for info in infos]))
            records.append(record)
            physical(saved['state']);terminal=replay.check(obs,new,done,infos,saved,intervals,record['public_after']);record['terminal_history']=terminal
            for name in after:
                np.testing.assert_array_equal(after[name][~done],saved[name][~done])
            for name,buffer in [('guard_memory',raw._joint_guard_buffers[0]),('parking_seen',raw._light_parking_buffers[0]),
                    ('parking_effective',raw._light_parking_buffers[1]),('phase_reference',raw._light_phase_buffers[0]),
                    ('command_total',raw._execution_buffers[0]),('command_prefix',raw._execution_buffers[1]),('command_stamp',raw._execution_buffers[2])]:
                np.testing.assert_array_equal(buffer.numpy()[~done],saved[name][~done])
            for name,buffer in [('phase_role',raw._light_phase_buffers[1]),('phase_monitor',raw._light_phase_buffers[2])]:
                np.testing.assert_array_equal(buffer.numpy()[:,~done],saved[name][:,~done])
            np.testing.assert_array_equal(raw._execution_previous_sensor_steps,replay.previous)
            if done.any():
                np.testing.assert_array_equal(after['latent'][done],0);np.testing.assert_array_equal(after['shadow'][done,16:22],0)
                np.testing.assert_allclose(after['reference'][done],public_reference(raw.obs.numpy()[done]),rtol=0,atol=1e-7)
                np.testing.assert_array_equal(raw._execution_previous_sensor_steps[done],0)
                np.testing.assert_array_equal(raw._execution_buffers[0].numpy()[done],0)
                np.testing.assert_array_equal(raw._execution_buffers[1].numpy()[done],0)
                expected=np.full((done.sum(),collector.RING),-1);expected[:,0]=0
                np.testing.assert_array_equal(raw._execution_buffers[2].numpy()[done],expected)
            for w in np.flatnonzero(done):
                info=infos[w];row=env.banks[int(stages[w])][w]
                assert info['physical_safety_passed'] and info['design_joint_passed']
                assert info['curriculum_stage']==stages[w] and info['seed']==row['seed'] and info['scenario']==row['scenario']
                episodes.append(dict(packet=k,world=int(w),**{key:value for key,value in info.items() if key!='terminal_observation'}))
            changed=stages!=env.stages
            assert not np.any(changed&~done)
            if changed.any():check_bank(raw,env,directory,str(k))
            for w in np.flatnonzero(changed):
                cache=env.cache[int(env.stages[w])]
                for buffer,name in ((raw.param,'param'),(raw.q0,'q0'),(raw.obs0,'obs0'),(raw.k['reference'],'reference')):
                    np.testing.assert_array_equal(buffer.numpy()[w],cache[name][w])
            events['partial_resets']+=int(0<done.sum()<10)
            for stage in (2,3):events[f'stage{stage}_rows']+=int((changed&(env.stages==stage)).sum())
            events['positive_delay_rows']+=int(((saved['delay_steps']>0)&(intervals['sensor_elapsed_s']>0)&(saved['guard'][:,:,0].sum(axis=0)>0)).sum())
            ticks+=1;ticks[done]=0;obs=new
            if len(records)==100:
                flush(directory,records,blocks)
                atomic_json(directory/'progress.json',dict(packets=k+1,events=events,actual_graph_steps=budget.graph_steps,FD_calls=len(budget.FD)))
                print('LIGHT374',directory.name,k+1,events,'graph',budget.graph_steps,flush=True)
        assert env.policy_steps==12010
        status='producer_passed_pending_independent_receipt' if all(events.values()) else 'inconclusive_missing_registered_events'
        return dict(status=status,packets=1200,events=events,blocks=blocks,episodes=len(episodes),task_successes=sum(bool(r['success']) for r in episodes))
    except BaseException as error:
        primary=error;raise
    finally:
        finish(errors,primary,directory,records,blocks,episodes,env,events,last_actor)


def preserve(raw,history,directory,errors):
    secondary(errors,directory.name+'/common',lambda:np.savez_compressed(directory/'final_common.npz',**snapshot(raw,history)))
    secondary(errors,directory.name+'/cartesian',lambda:np.savez_compressed(directory/'final_cartesian.npz',**{k:v.numpy() for k,v in raw._cartesian_buffers.items()}))
    if raw._cartesian_last is not None:
        secondary(errors,directory.name+'/last_terminal',lambda:np.savez_compressed(directory/'last_before_autoreset.npz',**raw._cartesian_last))


def build(p,arm):
    original=old.base.raw_env;constructed=[]
    def factory(rows,mode):
        standard=lambda:old.collector.instrument(lambda a,b:old.parking.instrument(original,a,b),rows,mode)
        guarded=lambda:old.guard.instrument(lambda:(standard(),),rows,None,dense=False)
        raw=old.cart.instrument(guarded,rows,0 if arm=='CartLive3' else 1)[0]
        constructed.append(model_values(raw))
        return raw
    with patch.object(old.base,'raw_env',factory):
        env=old.CartesianCurriculum(p,'virtual6',33531,10,milestones=[4000,8000])
    assert len(constructed)==3
    raw=env.venv;history=old.ExecutionHistory(old.RouteState(env),raw,'H1')
    return env,raw,history,constructed


def run():
    with (ROOT/'.git/project-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,a=verify(committed=True);QUEUE.mkdir(exist_ok=False)
        budget=Budget(p);errors=[];holders=[];reports={};initial_models=None;initial=None;private=None;constructors=None;caches=None;start=time.perf_counter()
        atomic_json(QUEUE/'started.json',dict(round=374,pid=os.getpid(),admission_sha256=sha(OUT/'source_admission.json'),
            semantic_acceptance_sha256=sha(OUT/'semantic_acceptance.json'),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),implicit_retry=False))
        launch=wp.capture_launch;fd=mujoco.mjd_transitionFD
        try:
            with patch.object(wp,'capture_launch',lambda *x,**k:budget.capture(launch,*x,**k)),patch.object(mujoco,'mjd_transitionFD',lambda *x,**k:budget.fd(fd,*x,**k)):
                for arm in p['arms']:
                    verify(committed=True);directory=QUEUE/arm;directory.mkdir()
                    env,raw,history,constructed=build(p,arm);holders.append((raw,history,directory));budget.bind(raw)
                    assert raw._cartesian_topology['counts']['control']==40 and raw._joint_guard_topology['captured_controller_calls']==40
                    obs=history.reset();values,scalars=model_values(raw)
                    np.savez_compressed(directory/'initial_model.npz',**values);atomic_json(directory/'initial_model.json',dict(scalars=scalars))
                    for stage,(arrays,metadata) in enumerate(constructed,1):
                        np.savez_compressed(directory/f'constructor_stage{stage}.npz',**arrays)
                        atomic_json(directory/f'constructor_stage{stage}.json',dict(scalars=metadata))
                    for stage,cache in env.cache.items():
                        np.savez_compressed(directory/f'cache_stage{stage}.npz',**cache['model'],meaninertia=cache['meaninertia'],q0=cache['q0'],param=cache['param'],obs0=cache['obs0'],reference=cache['reference'],geom_xpos=cache['geom_xpos'],geom_xmat=cache['geom_xmat'])
                    cache_values={stage:{**cache['model'],**{n:cache[n] for n in ('meaninertia','q0','param','obs0','reference','geom_xpos','geom_xmat')}} for stage,cache in env.cache.items()}
                    if constructors is None:constructors=constructed;caches=cache_values
                    else:
                        for (arrays,metadata),(first_arrays,first_metadata) in zip(constructed,constructors):
                            assert metadata==first_metadata and set(arrays)==set(first_arrays)
                            for name in arrays:np.testing.assert_array_equal(arrays[name],first_arrays[name])
                        assert set(cache_values)==set(caches)
                        for stage,values_cache in cache_values.items():
                            assert set(values_cache)==set(caches[stage])
                            for name in values_cache:np.testing.assert_array_equal(values_cache[name],caches[stage][name])
                    check_bank(raw,env,directory,'initial')
                    before=snapshot(raw,history);np.savez_compressed(directory/'initial.npz',**before)
                    own={k:raw._cartesian_buffers[k].numpy() for k in ('latent','shadow','reference')}
                    np.testing.assert_array_equal(own['latent'],0);np.testing.assert_array_equal(own['shadow'][:,16:22],0)
                    np.testing.assert_array_equal(before['controller_memory'][:,16:22],0)
                    if initial_models is None:initial_models=(values,scalars);initial=before;private=own
                    else:
                        assert scalars==initial_models[1] and set(values)==set(initial_models[0])
                        for name in values:np.testing.assert_array_equal(values[name],initial_models[0][name])
                        for name in initial:np.testing.assert_array_equal(before[name],initial[name])
                        for name in private:np.testing.assert_array_equal(own[name],private[name])
                    replay=Replay(raw.obs.numpy());zero=history.step(np.zeros((10,6),np.float32))
                    np.savez_compressed(directory/'zero_before_autoreset.npz',**raw._cartesian_last)
                    np.savez_compressed(directory/'zero_prefix.npz',**snapshot(raw,history),actor=obs,returned=zero[0],reward=zero[1],done=zero[2],**raw._execution_intervals)
                    assert not zero[2].any();physical(raw._cartesian_last['state'])
                    np.testing.assert_array_equal(raw._cartesian_last['latent'],0)
                    np.testing.assert_array_equal(raw._cartesian_last['shadow'][:,16:22],0)
                    np.testing.assert_array_equal(raw.k['state'].numpy()[:,16:22],0)
                    replay.check(obs,zero[0],zero[2],zero[3],raw._cartesian_last,raw._execution_intervals,raw.obs.numpy())
                    assert env.policy_steps==10
                    obs=history.reset();reports[arm]=rollout(p,env,raw,history,obs,directory,budget,errors)
                    check_bank(raw,env,directory,'final')
                    preserve(raw,history,directory,errors)
                    if errors:raise RuntimeError('Recording preservation error')
                    assert budget.active['calls']==1201;history.close();verify(committed=True)
            assert budget.graph_steps==960800 and len(budget.bindings)==2
            atomic_json(QUEUE/'completion.json',dict(round=374,status='producer_finished_pending_independent_receipt',reports=reports,
                graph_world_steps=budget.graph_steps,graph_calls_each=[b['calls'] for b in budget.bindings],FD_calls=budget.FD,
                wall_seconds=time.perf_counter()-start,secondary_errors=errors,admission_sha256=sha(OUT/'source_admission.json'),
                new_learning_samples=0,new_model_forward_rows=0,formal5_admitted=False))
        except BaseException as error:
            for raw,history,directory in holders:
                preserve(raw,history,directory,errors);secondary(errors,directory.name+'/close',history.close)
            failure=dict(round=374,status='failed',primary_error=repr(error),secondary_errors=errors,reports=reports,
                graph_world_steps=budget.graph_steps,graph_calls_each=[b['calls'] for b in budget.bindings],FD_calls=budget.FD,
                wall_seconds=time.perf_counter()-start,new_learning_samples=0,new_model_forward_rows=0,implicit_retry=False)
            secondary(errors,'failure_manifest',lambda:atomic_json(QUEUE/'failure.json',failure))
            print('FAILED374',json.dumps(failure,ensure_ascii=False),flush=True);raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.command]()
