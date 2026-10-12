"""Registered light10world scripted qualification; no policy, optimizer or retry."""
from collections import deque
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
import train_height_comparison as base
import execution_input_collector as collector
import light_parking_request as parking
import joint_state_guard as guard
import cartesian_light_runtime as cart
from cartesian_curriculum import CartesianCurriculum
from execution_history_env import ExecutionHistory
from execution_input_features import SCALES
from route_state import RouteState
from fixed_reference_force import embed
from probe_packet_odometry import lateral_velocity
from state_estimation import leg_kinematics
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
from review_cartesian_repair import projection
from review_yaw_sector import ROOT,sha
from run_wheel_interference import runtime
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'
OUT=BASE/'cartesian_training_interface_v1'
QUEUE=OUT/'runtime'


def inputs():
    p=json.loads((OUT/'proposal.json').read_text())
    assert p['arms']==['CartLive3','CartReset3'] and p['worlds']==10 and p['bank_seed']==33531
    assert p['rollout_packets_each']==1200 and p['zero_prefix_packets_total']==3
    assert p['actual_graph_world_step_budget']==961200 and p['constructor_FD_budget']==80
    assert p['curriculum_milestones']==[4000,8000] and p['new_learning_samples']==p['new_model_forward_rows']==0
    assert not p['scientific_training_admitted'] and not p['formal5_admitted']
    bank=BASE/'fixed_force_prequalification_v1/proposal.json'
    assert sha(bank)==p['training_bank_sha256']
    old=json.loads(bank.read_text())['training_banks']['33531']
    assert p['training_banks']['33531']=={k:v[:10] for k,v in old.items()}
    for mapping in (p['source_sha256'],p['evidence_sha256']):
        assert all(sha(ROOT/f)==h for f,h in mapping.items())
    return p


def freeze():
    inputs();assert not (OUT/'source_admission.json').exists() and not QUEUE.exists()
    self_check()
    sources=json.loads((BASE/'cartesian_pair_delivery_repair_v2/runtime_admission.json').read_text())['source_sha256'].copy()
    assert all(sha(ROOT/f)==h for f,h in sources.items())
    for name in ('cartesian_light_runtime.py','cartesian_curriculum.py','test_cartesian_light_runtime.py',
                 'test_cartesian_curriculum.py','run_cartesian_light_interface.py'):
        path=ROOT/'wheelleg_warp'/name;sources[str(path.relative_to(ROOT))]=sha(path)
    atomic_json(OUT/'source_admission.json',dict(round=368,verified=True,proposal_sha256=sha(OUT/'proposal.json'),
        source_sha256=sources,runtime=runtime(),new_physics_steps=0,new_learning_samples=0,
        real_interface_not_yet_verified=True,commit_required_before_first_launch=True,
        records_block_packets=100,implicit_retry=False))


def verify(committed=False):
    p=inputs();a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json')
    assert a['runtime']==runtime() and all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    if committed:
        path=str((OUT/'source_admission.json').relative_to(ROOT))
        import hashlib
        assert hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)).hexdigest()==sha(OUT/'source_admission.json')
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Need2GiB archive space')
    return p,a


class Budget:
    def __init__(self,p):self.p=p;self.graph_steps=0;self.FD=[]
    def capture(self,original,*args,**kwargs):
        amount=40*self.p['worlds']
        if self.graph_steps+amount>self.p['actual_graph_world_step_budget']:raise RuntimeError('Graph budget exhausted')
        self.graph_steps+=amount;return original(*args,**kwargs)
    def fd(self,original,*args,**kwargs):
        if len(self.FD)>=self.p['constructor_FD_budget']:raise RuntimeError('FD budget exhausted')
        self.FD.append(float(args[2]));return original(*args,**kwargs)


def build(p,arm):
    original=base.raw_env
    def factory(rows,mode):
        standard=lambda:collector.instrument(lambda a,b:parking.instrument(original,a,b),rows,mode)
        guarded=lambda:guard.instrument(lambda:(standard(),),rows,None,dense=False)
        return guarded()[0] if arm=='B0' else cart.instrument(guarded,rows,0 if arm=='CartLive3' else 1)[0]
    if arm=='B0':env=raw=factory(p['training_banks']['33531']['1'],'virtual6')
    else:
        with patch.object(base,'raw_env',factory):
            env=CartesianCurriculum(p,'virtual6',33531,10,milestones=p['curriculum_milestones'])
        raw=env.venv
    assert raw._joint_guard_topology['captured_controller_calls']==40 and not raw._joint_guard_topology['dense_recording']
    assert raw._execution_topology['final_ctrl_before_physics'] and raw._light_phase_topology['control_calls']==40
    return env,raw,ExecutionHistory(RouteState(env),raw,'H1')


def snapshot(raw,history):
    values={name:b.numpy() for name,b in dict(q=raw.data.qpos,v=raw.data.qvel,ctrl=raw.data.ctrl,
        warm=raw.data.qacc_warmstart,time=raw.data.time,sensors=raw.data.sensordata,state=raw.state,
        controller_memory=raw.k['state'],diag=raw.diag,param=raw.param,reference=raw.k['reference'],
        public=raw.obs,guard_memory=raw._joint_guard_buffers[0],guard=raw._joint_guard_buffers[1]).items()}
    for name,buffers in [('command',raw._execution_buffers),('parking',raw._light_parking_buffers),('phase',raw._light_phase_buffers)]:
        values.update({f'{name}_{i}':b.numpy() for i,b in enumerate(buffers)})
    values.update({name:getattr(history,name).copy() for name in ('frames','inputs','elapsed','valid')})
    values['previous_sensor_steps']=raw._execution_previous_sensor_steps.copy()
    return values


def physical(state):
    assert state.ndim==2 and state.shape[1]==39 and np.isfinite(state).all()
    assert np.all(state[:,37]==state[:,0]) and np.all(state[:,0]>0),'Missing physical evidence'
    assert np.min(state[:,31:33])>=HEIGHT_115_GEOMETRIC_MIN,'Physical leg margin'
    assert np.min(state[:,33])>=0 and np.max(state[:,35:37])<=1e-6,'Physical joint/torque margin'
    assert np.min(state[:,38])>=0,'Original active1.4 design gate'


def public_reference(packet):
    return np.array([np.concatenate([leg_kinematics(row[12+2*s:14+2*s].astype(float),np.zeros(2))[0] for s in range(2)]) for row in packet])


def script(ticks):
    t=np.asarray(ticks)*.02
    return embed(.15*np.column_stack((np.sin(np.pi*t),np.cos(np.pi*t),np.sin(2*np.pi*t))),'D3',len(t))


class Replay:
    """Check old sensor windows using a separate final-command ring and episode clocks."""
    def __init__(self,packet):
        self.commands=[deque(maxlen=82) for _ in packet]
        self.previous=np.zeros(len(packet),np.int64);self.counts=self.previous.copy()
        self.y=np.zeros(len(packet));self.clock=self.y.copy();self.velocity=lateral_velocity(packet)
    def check(self,old,returned,done,infos,saved,intervals,public_after):
        n=len(done);g=saved['guard'];c=saved['trace'];valid=g[:,:,0]==1
        np.testing.assert_array_equal(valid,c[:,:,0]==1)
        np.testing.assert_array_equal(valid,saved['parking'][:,:,0]==1)
        count=saved['state'][:,0].astype(np.int64);delay=saved['delay_steps'].astype(np.int64)
        np.testing.assert_array_equal(saved['previous_sensor_steps'],self.previous)
        end=np.maximum(count-delay,0);dt=(end-self.previous)*.0005;impulse=np.zeros((n,6))
        for w in range(n):
            values=g[:,w][valid[:,w]]
            np.testing.assert_array_equal(values[:,1],np.arange(self.counts[w]+1,count[w]+1))
            self.commands[w].extend((int(row[1]),row[20:26].copy()) for row in values)
            keep=[u for step,u in self.commands[w] if self.previous[w]<step<=end[w]]
            assert len(keep)==end[w]-self.previous[w] and 0<=len(keep)<=40
            if keep:impulse[w]=np.array(keep).sum(axis=0)*.0005
            for step in (self.previous[w],end[w]):assert saved['command_stamp'][w,step%collector.RING]==step
        np.testing.assert_allclose(intervals['command_integral'],impulse,rtol=0,atol=1e-10)
        prefix=saved['command_prefix'];world=np.arange(n)
        np.testing.assert_allclose(prefix[world,end%collector.RING]-prefix[world,self.previous%collector.RING],impulse,rtol=0,atol=1e-10)
        np.testing.assert_array_equal(intervals['sensor_elapsed_s'],dt)
        assert np.all((dt>=0)&(dt<=.02))
        cv=c[valid];gv=g[valid];park=saved['parking'][valid]
        np.testing.assert_allclose(cv[:,14],projection(gv[:,2:8],cv[:,15:21],cv[:,27:33]),rtol=0,atol=1e-12)
        np.testing.assert_array_equal(gv[:,26],cv[:,14])
        np.testing.assert_array_equal(cv[:,4:7],cv[:,1:4]+np.clip(park[:,11:17:2]-cv[:,1:4],-.01,.01))
        np.testing.assert_array_equal(park[:,17:23],embed_unbounded(cv[:,1:4]))
        np.testing.assert_array_equal(park[:,23:29],embed_unbounded(cv[:,4:7]))
        np.testing.assert_array_equal(old[:,383:389],embed_unbounded(c[0,:,1:4]).astype(np.float32))
        packet=saved['terminal_packet'];route_dt=np.full(n,.02)
        route_dt[done]=saved['state'][done,0]*.0005-self.clock[done]
        velocity=lateral_velocity(packet);self.y+=.5*(self.velocity+velocity)*route_dt
        self.velocity=velocity;self.clock+=route_dt
        endpoint=np.column_stack((packet,self.y.astype(np.float32)))
        frames=old[:,:390].reshape(n,10,39).copy();commands=old[:,390:462].reshape(n,9,8).copy()
        elapsed=old[:,462:471].copy();valid_history=old[:,471:].copy()
        frames[:,:-1]=frames[:,1:].copy();frames[:,-1]=endpoint
        commands[:,:-1]=commands[:,1:].copy();commands[:,-1]=0
        mean=np.divide(intervals['command_integral'],dt[:,None],out=np.zeros((n,6)),where=dt[:,None]>0)
        commands[:,-1,:6]=mean/SCALES;elapsed[:,:-1]=elapsed[:,1:].copy();elapsed[:,-1]=dt/.02
        valid_history[:,:-1]=valid_history[:,1:].copy();valid_history[:,-1]=1
        terminal=np.concatenate((frames.reshape(n,390),commands.reshape(n,72),elapsed,valid_history),axis=1)
        for w in np.flatnonzero(done):
            np.testing.assert_array_equal(infos[w]['terminal_observation'],terminal[w])
            self.commands[w].clear();self.y[w]=self.clock[w]=0
        reset=np.column_stack((public_after,np.where(done,0,self.y).astype(np.float32)))
        frames[done]=reset[done,None,:];commands[done]=0;elapsed[done]=0;valid_history[done]=0;valid_history[done,-1]=1
        expected=np.concatenate((frames.reshape(n,390),commands.reshape(n,72),elapsed,valid_history),axis=1)
        np.testing.assert_array_equal(returned,expected)
        np.testing.assert_array_equal(public_after[~done],packet[~done])
        self.velocity[done]=lateral_velocity(reset[done,:38]);self.previous=end;self.previous[done]=0
        self.counts=count;self.counts[done]=0
        return terminal


def embed_unbounded(a):return np.column_stack((a[:,0],-a[:,0],a[:,1],-a[:,1],a[:,2],-a[:,2]))


def flush(directory,records,blocks):
    if not records:return
    file=directory/f'block_{len(blocks):03d}.npz'
    assert not file.exists();np.savez_compressed(file,**{k:np.stack([r[k] for r in records]) for k in records[0]})
    blocks.append(dict(path=file.name,sha256=sha(file),packets=len(records)));records.clear()


def rollout(p,env,raw,history,obs,directory,budget):
    records=[];blocks=[];episodes=[];replay=Replay(raw.obs.numpy());ticks=np.zeros(10,np.int64)
    events=dict(partial_resets=0,stage2_rows=0,stage3_rows=0,positive_delay_rows=0)
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
                print('LIGHT368',directory.name,k+1,events,'graph',budget.graph_steps,flush=True)
        assert env.policy_steps==12010
        status='producer_passed_pending_independent_receipt' if all(events.values()) else 'inconclusive_missing_registered_events'
        return dict(status=status,packets=1200,events=events,blocks=blocks,episodes=len(episodes),task_successes=sum(bool(r['success']) for r in episodes))
    finally:
        if 'last_actor' in locals():np.savez_compressed(directory/'last_submitted_actor.npz',**last_actor)
        flush(directory,records,blocks)
        atomic_json(directory/'episodes.json',dict(episodes=episodes,transitions=env.transition_log,events=events,blocks=blocks))


def preserve(raw,history,directory):
    np.savez_compressed(directory/'final_buffers.npz',**snapshot(raw,history),
        **{f'cart_{k}':v.numpy() for k,v in getattr(raw,'_cartesian_buffers',{}).items()})
    saved=getattr(raw,'_cartesian_last',None)
    if saved is not None:np.savez_compressed(directory/'last_before_autoreset.npz',**saved)


def run():
    with (ROOT/'.git/project-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,a=verify(committed=True);QUEUE.mkdir(exist_ok=False)
        budget=Budget(p);started=time.perf_counter();holders=[];reports={};baseline=None;initial_cart=None
        atomic_json(QUEUE/'started.json',dict(pid=os.getpid(),round=368,admission_sha256=sha(OUT/'source_admission.json'),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),implicit_retry=False))
        capture=wp.capture_launch;fd=mujoco.mjd_transitionFD
        try:
            with patch.object(wp,'capture_launch',lambda *x,**y:budget.capture(capture,*x,**y)),\
                 patch.object(mujoco,'mjd_transitionFD',lambda *x,**y:budget.fd(fd,*x,**y)):
                for arm in ['B0',*p['arms']]:
                    verify();directory=QUEUE/arm;directory.mkdir()
                    env,raw,history=build(p,arm);holders.append((arm,env,raw,history,directory,None))
                    obs=history.reset();before=snapshot(raw,history);np.savez_compressed(directory/'initial.npz',**before)
                    if baseline is not None:
                        for key in before:np.testing.assert_array_equal(before[key],baseline[0][key],err_msg='Zero initial '+key)
                    if arm!='B0':
                        own={k:raw._cartesian_buffers[k].numpy() for k in ('latent','reference','shadow')}
                        if initial_cart is None:initial_cart=own
                        else:
                            for key in own:np.testing.assert_array_equal(own[key],initial_cart[key])
                        np.savez_compressed(directory/'cartesian_initial.npz',**own)
                    zero=np.zeros((10,6),np.float32);obs,reward,done,infos=history.step(zero)
                    assert not done.any();physical(raw.state.numpy());after=snapshot(raw,history)
                    np.savez_compressed(directory/'zero_prefix.npz',**after,returned=obs,reward=reward,done=done)
                    if baseline is None:baseline=(before,after,obs.copy(),reward.copy())
                    else:
                        for key in after:np.testing.assert_array_equal(after[key],baseline[1][key],err_msg='Zero prefix '+key)
                        np.testing.assert_array_equal(obs,baseline[2]);np.testing.assert_array_equal(reward,baseline[3])
                    if arm=='B0':history.close()
                    else:
                        assert env.policy_steps==10;obs=history.reset();holders[-1]=(arm,env,raw,history,directory,obs)
                for arm,env,raw,history,directory,obs in holders[1:]:
                    verify();reports[arm]=rollout(p,env,raw,history,obs,directory,budget)
                    preserve(raw,history,directory);verify()
            assert budget.graph_steps==p['actual_graph_world_step_budget']
            atomic_json(QUEUE/'completion.json',dict(round=368,status='complete_pending_independent_receipt',reports=reports,
                actual_graph_world_steps=budget.graph_steps,constructor_FD_calls=budget.FD,wall_seconds=time.perf_counter()-started,
                proposal_sha256=sha(OUT/'proposal.json'),admission_sha256=sha(OUT/'source_admission.json'),
                new_learning_samples=0,new_model_forward_rows=0,formal5_admitted=False,scientific_training_admitted=False))
        except BaseException as error:
            failures=[]
            for arm,env,raw,history,directory,obs in holders:
                try:preserve(raw,history,directory)
                except BaseException as secondary:failures.append(dict(arm=arm,error=repr(secondary)))
            atomic_json(QUEUE/'failure.json',dict(round=368,error=repr(error),reports=reports,
                actual_graph_world_steps=budget.graph_steps,constructor_FD_calls=budget.FD,
                preservation_errors=failures,implicit_retry=False,wall_seconds=time.perf_counter()-started))
            raise
        finally:
            for arm,env,raw,history,directory,obs in holders:history.close()


def self_check():
    p=dict(worlds=10,actual_graph_world_step_budget=400,constructor_FD_budget=1);b=Budget(p);calls=[]
    b.capture(lambda *a,**k:calls.append('graph'));b.fd(lambda *a,**k:calls.append('FD'),None,None,.001)
    for action in (lambda:b.capture(lambda:calls.append('bad')),lambda:b.fd(lambda *a:calls.append('bad'),None,None,.001)):
        try:action()
        except RuntimeError:pass
        else:raise AssertionError('Budget overflow admitted')
    assert calls==['graph','FD'] and b.graph_steps==400 and len(b.FD)==1
    np.testing.assert_array_equal(script(np.zeros(3,int)),np.tile([0,0,.15,-.15,0,0],(3,1)).astype(np.float32))
    ticks=np.array([2,3,0]);np.testing.assert_array_equal(script(ticks)[2],script(np.zeros(1,int))[0])
    state=np.zeros((2,39));state[:,0]=state[:,37]=40;state[:,31:33]=HEIGHT_115_GEOMETRIC_MIN;physical(state)
    for column,value in ((37,39),(31,0),(33,-1e-9),(35,1e-5),(38,-1e-9)):
        bad=state.copy();bad[0,column]=value
        try:physical(bad)
        except AssertionError:pass
        else:raise AssertionError('Original physical gate weakened')
    packet=np.zeros((3,38),np.float32);old=np.zeros((3,481),np.float32);old[:,480]=1
    done=np.array([True,False,False]);delay=np.array([0,2,0]);dt=np.array([.002,.001,.002])
    state=np.zeros((3,39));state[:,0]=4
    g=np.zeros((40,3,62));c=np.zeros((40,3,38));park=np.zeros((40,3,30))
    g[:4,:,0]=c[:4,:,0]=park[:4,:,0]=1;g[:4,:,1]=np.arange(1,5)[:,None]
    g[:4,:,20:26]=1;g[:4,:,26]=c[:4,:,14]=1;c[:4,:,27:33]=1
    prefix=np.zeros((3,collector.RING,6));stamp=np.full((3,collector.RING),-1)
    prefix[:,:5]=np.arange(5)[None,:,None]*.0005;stamp[:,:5]=np.arange(5)
    saved=dict(guard=g,trace=c,parking=park,state=state,terminal_packet=packet,
        delay_steps=delay,previous_sensor_steps=np.zeros(3,int),command_stamp=stamp,command_prefix=prefix)
    intervals=dict(command_integral=np.repeat(dt[:,None],6,axis=1),sensor_elapsed_s=dt)
    terminal=old.copy();terminal[:,471:]=0;terminal[:,479:]=1
    terminal[:,454:460]=1/SCALES;terminal[:,470]=dt/.02
    public=packet.copy();public[0,12]=.25
    returned=terminal.copy();returned[0]=0;returned[0,:390]=np.tile(np.r_[public[0],0],10);returned[0,480]=1
    infos=[dict(terminal_observation=terminal[0]),{},{}]
    replay=Replay(packet);replay.check(old,returned,done,infos,saved,intervals,public)
    np.testing.assert_array_equal(replay.previous,[0,2,4]);np.testing.assert_array_equal(replay.counts,[0,4,4])
    assert len(replay.commands[0])==0 and len(replay.commands[1])==4
    for field in ('command_integral','reset_packet'):
        bad=returned.copy();bad_intervals={k:v.copy() for k,v in intervals.items()}
        if field=='command_integral':bad_intervals[field][1,0]+=.001
        else:bad[0,363]=.5
        try:Replay(packet).check(old,bad,done,infos,saved,bad_intervals,public)
        except AssertionError:pass
        else:raise AssertionError('Foreign integral/reset packet accepted')
    print('PASS368 graph/FD caps,episode script,physical gates,partial reset/delayed final-command history and corrupt packet rejection;0physics',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['check','freeze','run']);args=parser.parse_args()
    {'check':self_check,'freeze':freeze,'run':run}[args.command]()
