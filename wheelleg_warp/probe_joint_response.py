"""Bounded full-state capture and same-state CPU one-step diagnostics.

Shadow models know the simulated geometry/parameters only for diagnosis.
They are not deployed predictors and supply no information to the actor.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json
import numpy as np
import mujoco
import warp as wp
from stable_baselines3 import PPO
import route_pilot_env as experiment
from native.controller import D,control_physical_nominal
from native.environment import command_step,after
from native.terrain import HeightTerrainScenario,model as build_model
from leg_residual_cone import supervise
from smoke_reward_training import weight_digest
from training_contract import digest
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/leg_cone_v1'
# Snapshot fields: pre q17/v16/warm16/time1/ctrl6/Nom6/Actor6, post q17/v16.
WIDTH=101


@wp.kernel
def save_pre(q:wp.array2d[float],v:wp.array2d[float],warm:wp.array2d[float],ctrl:wp.array2d[float],
             diag:wp.array2d[D],task:wp.array2d[D],active:wp.array[int],buffer:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    for j in range(17):buffer[w,j]=D(q[w,j])
    for j in range(16):buffer[w,17+j]=D(v[w,j]);buffer[w,33+j]=D(warm[w,j])
    buffer[w,49]=task[w,0]*D(.0005)
    for j in range(6):buffer[w,50+j]=D(ctrl[w,j]);buffer[w,56+j]=diag[w,j];buffer[w,62+j]=diag[w,6+j]


@wp.kernel
def save_events(q:wp.array2d[float],v:wp.array2d[float],ids:wp.array[int],active:wp.array[int],
                pre:wp.array2d[D],filled:wp.array2d[int],snapshots:wp.array3d[D]):
    w=wp.tid()
    if active[w]==0:return
    pre_margin=D(1.e30);post_margin=D(1.e30);predicted=D(1.e30)
    for j in range(4):
        position=pre[w,ids[j]];velocity=pre[w,17+ids[4+j]]
        pre_margin=wp.min(pre_margin,D(1.4)-wp.abs(position))
        predicted=wp.min(predicted,D(1.4)-wp.abs(position)-wp.max(D(0),wp.sign(position)*velocity)*D(.02))
        post_margin=wp.min(post_margin,D(1.4)-wp.abs(D(q[w,ids[j]])))
    for event in range(3):
        selected=(event==0 and predicted<=D(0)) or (event==1 and pre_margin<=D(1.e-5)) or (event==2 and post_margin<D(0))
        if selected and filled[w,event]==0:
            filled[w,event]=1
            for j in range(68):snapshots[w,event,j]=pre[w,j]
            for j in range(17):snapshots[w,event,68+j]=D(q[w,j])
            for j in range(16):snapshots[w,event,85+j]=D(v[w,j])


def capture(cases,agent,normalization,out,label):
    old_factory=experiment.raw_env;old_launch=wp.launch;captured={}
    def factory(selected,mode):
        assert selected==cases and mode=='diff3'
        n=len(cases);pre=wp.zeros((n,WIDTH),dtype=D);filled=wp.zeros((n,3),dtype=int);snapshots=wp.zeros((n,3,WIDTH),dtype=D);stats=wp.zeros((n,7),dtype=D)
        warm=None;task=None
        def launch(kernel,dim,inputs=None,**kwargs):
            nonlocal warm,task
            if kernel is command_step:warm=inputs[6];task=inputs[0]
            if kernel is after:old_launch(save_events,n,[inputs[0],inputs[1],inputs[6],inputs[13],pre,filled,snapshots])
            answer=old_launch(kernel,dim,**kwargs) if inputs is None else old_launch(kernel,dim,inputs,**kwargs)
            if kernel is control_physical_nominal:
                old_launch(supervise,n,[1,inputs[0],inputs[1],inputs[7],inputs[5],inputs[6],inputs[15],inputs[14],inputs[18],stats])
                old_launch(save_pre,n,[inputs[0],inputs[1],warm,inputs[14],inputs[15],task,inputs[5],pre])
            return answer
        wp.launch=launch
        try:env=old_factory(cases,mode)
        finally:wp.launch=old_launch
        assert env.cpu.nq==17 and env.cpu.nv==16 and env.cpu.na==0
        assert np.all(env.data.qfrc_applied.numpy()==0) and np.all(env.data.xfrc_applied.numpy()==0)
        assert not np.any(env.cpu.body_mocapid>=0),'Moving geometry state is not captured'
        pending=set(range(n));saved=[];wait=env.step_wait
        def step_wait():
            result=wait()
            for w in list(pending):
                if result[2][w]:
                    marks=filled.numpy();values=snapshots.numpy()
                    for event in range(3):
                        if marks[w,event]:saved.append(dict(world=w,event=event,state=values[w,event].copy()))
                    marks[w,:]=1;filled.assign(marks);pending.remove(w)
            return result
        env.step_wait=step_wait;captured['saved']=saved;captured['ids']=env.ids.numpy();return env
    experiment.raw_env=factory;before=(agent.num_timesteps,agent._n_updates,weight_digest(agent))
    try:result=experiment.evaluate(cases,'L2-route',agent,normalization)
    finally:experiment.raw_env=old_factory;wp.launch=old_launch
    assert before==(agent.num_timesteps,agent._n_updates,weight_digest(agent))
    saved=captured['saved'];arrays=np.stack([r['state'] for r in saved]) if saved else np.empty((0,WIDTH))
    worlds=np.array([r['world'] for r in saved],int);events=np.array([r['event'] for r in saved],int)
    assert np.isfinite(arrays).all() and len(saved)<=len(cases)*3
    np.savez_compressed(out/(label+'_states.npz'),state=arrays,world=worlds,event=events,ids=captured['ids'])
    atomic_json(out/(label+'_episodes.json'),result)
    return arrays,worlds,events,captured['ids']


def shadow(cases,states,worlds,events,ids):
    models={};result=[]
    for i,(state,world,event) in enumerate(zip(states,worlds,events)):
        scene=HeightTerrainScenario(**cases[world]['scenario'])
        if world not in models:models[world]=build_model(scene)
        actual_model=models[world];assert actual_model.na==0 and actual_model.nmocap==0
        q=state[:17];v=state[17:33];warm=state[33:49];actual=state[50:56];nominal=state[56:62]
        branches={'actual_command':actual,'nominal_only':nominal}
        # One held motor-increment step, not an optimized controller. Physical
        # command bounds remain enforced by the original speed curve.
        from state_estimation import leg_kinematics
        matrix=np.vstack([leg_kinematics(q[ids[:2]],np.zeros(2))[3]*[3.4335,1],-leg_kinematics(q[ids[2:4]],np.zeros(2))[3]*[3.4335,1]])
        effective=np.linalg.lstsq(matrix,state[62:66],rcond=None)[0]
        np.testing.assert_allclose(matrix@effective,state[62:66],rtol=0,atol=1e-8);assert abs(effective).max()<=1+1e-8
        for column,coord in enumerate(['Fdiff','Hdiff']):
            for sign in [-1,1]:
                u=np.clip(effective+sign*.01*np.eye(2)[column],-1,1);increment=np.r_[matrix@u,0.,0.];lam=1.
                for j in range(6):
                    limit=experiment.sim.hw.torque_limit(float('inf'),float(v[ids[4+j]]),j<4,0.,.0005)[0]/(1 if j<4 else 1.05)
                    if increment[j]>0:lam=min(lam,(limit-nominal[j])/increment[j])
                    elif increment[j]<0:lam=min(lam,(-limit-nominal[j])/increment[j])
                branches[f'{coord}_{sign}']=nominal+np.clip(lam,0,1)*increment
        outcomes={}
        for name,command in branches.items():
            data=mujoco.MjData(actual_model);data.qpos[:]=q;data.qvel[:]=v;data.qacc_warmstart[:]=warm;data.ctrl[:]=command;data.time=state[49]
            mujoco.mj_step(actual_model,data)
            outcomes[name]=dict(active_q=data.qpos[ids[:4]].tolist(),active_v=data.qvel[ids[4:8]].tolist(),
                effective_acceleration=((data.qvel[ids[4:8]]-v[ids[4:8]])/.0005).tolist(),ctrl=command.tolist())
        gpu_postq=state[68:85];gpu_postv=state[85:101]
        cpu=outcomes['actual_command'];qerror=np.asarray(cpu['active_q'])-gpu_postq[ids[:4]];verror=np.asarray(cpu['active_v'])-gpu_postv[ids[4:8]]
        result.append(dict(index=i,world=int(world),event=int(event),case=cases[world]['seed'],time_s=float(state[49]),
            observed_gpu_active_q=gpu_postq[ids[:4]].tolist(),observed_gpu_active_v=gpu_postv[ids[4:8]].tolist(),
            cpu_gpu_active_q_error=qerror.tolist(),cpu_gpu_active_v_error=verror.tolist(),branches=outcomes))
    return result


def run(out):
    p=json.loads((BASE/'registration.json').read_text());review=json.loads((BASE/'review.json').read_text());assert review['verified']
    assert all(digest(ROOT/n)==v for n,v in p['source_sha256'].items())
    out.mkdir(parents=True,exist_ok=False)
    reg=dict(version='joint-response-diagnostic-v1',cases=p['cases'],models=p['models'],budget_episodes=81,maximum_snapshots=243,
        maximum_cpu_shadow_steps=243*6,events=['first20ms kinematic forecast','first pre-margin<=1e-5rad','first post crossing'],
        training_updates=0,control='Same previously stopped cone fixed-policy experiment, no new horizon/control/constraint changes.',
        snapshots='Pre full q17/v16/warm16, held executed/Nom/Actor commands, step-derived time; post q17/v16. No actuator activation,mocap or applied forces; static model/contacts recomputed by shadow.',
        shadow='CPU MuJoCo actual simulated model, warmstart and held commands: actual,Nom alone,±one original0.01 increment of normalized differential F/H within original boxes. Diagnostic privileged geometry/parameters never fed to controller/actor; not a deployed nominal predictor.',
        limits='CPU float64 and GPU float32/contact ordering may differ. Compare retained GPU next state, report errors without replacing observed state/gate. Empirical maxima not prospective bounds.',
        source_sha256={**p['source_sha256'],'wheelleg_warp/probe_joint_response.py':digest(__file__)},old_gate_or_final_used=False)
    atomic_json(out/'registration.json',reg);completed=0;records=[]
    try:
        for m in p['models']:
            prefix=Path(m['prefix']);assert digest(prefix.with_suffix('.zip'))==m['checkpoint']['checkpoint_sha256'] and digest(prefix.with_suffix('.pkl'))==m['checkpoint']['normalization_sha256']
            agent=PPO.load(str(prefix)+'.zip',device='cuda');label=str(m['seed'])
            atomic_json(out/'progress.json',dict(status='running',completed_episodes=completed,pending_job=label))
            states,worlds,events,ids=capture(p['cases'],agent,str(prefix)+'.pkl',out,label)
            results=shadow(p['cases'],states,worlds,events,ids);atomic_json(out/(label+'_response.json'),dict(snapshots=len(states),shadow_steps=len(states)*6,outcomes=results))
            records.append(dict(label=label,snapshots=len(states),states_sha256=digest(out/(label+'_states.npz')),episodes_sha256=digest(out/(label+'_episodes.json')),response_sha256=digest(out/(label+'_response.json'))));completed+=27
            atomic_json(out/'completed_jobs.json',dict(completed_episodes=completed,records=records));print('COMPLETED',completed,label,'snapshots',len(states),flush=True)
        assert completed==81
        atomic_json(out/'completion.json',dict(completed_episodes=completed,records=records,training_updates=0,registration_sha256=digest(out/'registration.json')))
        atomic_json(out/'progress.json',dict(status='complete',completed_episodes=completed))
    except BaseException as e:
        atomic_json(out/'interruption.json',dict(completed_episodes=completed,error=str(e),pending_job_consumption_unknown=True,silently_resumable=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
