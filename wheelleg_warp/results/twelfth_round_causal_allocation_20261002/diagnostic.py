"""State-dependent5ms allocation of a feasible nominal guide; original domain/gates."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from scipy.optimize import linprog
from native.environment import NativeEnv,begin,command_step,reduce_contacts,collect_physical,after
from native.controller import D
from probe_braking_feedback import Forecaster,record_forecast
from probe_height_115_margin import cases
from select_braking_common_action import batch_constraint_margins,task_forecast
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent
PLANS=OUT.parent/'braking_trajectory_slsqp_20261001'


def margins(p,trace,q,v,state):
    pos=trace[:,:,:p.nq];vel=trace[:,:,p.nq:p.nq+p.nv]
    physical=batch_constraint_margins(p.nom,pos,vel,trace[:,:,-12:-6],trace[:,:,-6:],v)
    task=task_forecast(pos,vel,q,v,state)
    return np.c_[physical,.6-task['peak_distance_m'],.03-task['peak_tail_speed_m_s']]


def allocate(p,env,current,wanted):
    # ponytail: local FD constrained map, followed by nonlinear recheck; no recursive certificate.
    sign=np.where(current+.001<=1,1.,-1.);samples=[]
    for w in range(2):
        arms=np.tile(current[w],(13,1))
        for j in range(6):
            arms[1+j,j]+=.001*sign[w,j];arms[7+j,j]+=.0005*sign[w,j]
        samples.append(arms)
    samples=np.concatenate(samples);trace=p.forecast_nominal(env,samples)
    q=np.repeat(env.data.qpos.numpy(),13,axis=0);v=np.repeat(env.data.qvel.numpy(),13,axis=0);state=np.repeat(env.state.numpy(),13,axis=0)
    g=margins(p,trace,q,v,state);chosen=[];rows=[]
    for w in range(2):
        sl=slice(13*w,13*(w+1));values=g[sl];base=values[0]
        jac=((values[1:7]-base)/(.001*sign[w,:,None])).T
        half=((values[7:]-base)/(.0005*sign[w,:,None])).T
        A=[];b=[]
        for j in range(6):
            row=np.zeros(13);row[j]=1;row[6]=-1;A.append(row);b.append(wanted[w,j])
            row=np.zeros(13);row[j]=-1;row[6]=-1;A.append(row);b.append(-wanted[w,j])
            row=np.zeros(13);row[j]=1;row[7+j]=-1;A.append(row);b.append(current[w,j])
            row=np.zeros(13);row[j]=-1;row[7+j]=-1;A.append(row);b.append(-current[w,j])
        row=np.zeros(13);row[7:]=1;A.append(row);b.append(.1)
        A.extend(np.c_[-jac,np.zeros((len(base),7))]);b.extend(base-jac@current[w])
        c=np.zeros(13);c[6]=1
        solve=linprog(c,A_ub=A,b_ub=b,bounds=[(-1,1)]*6+[(0,None)]*7,method='highs')
        row=dict(world=w,solver_status=int(solve.status),solver_message=solve.message,current=current[w].tolist(),wanted=wanted[w].tolist(),
            nominal_zero_change_minimum_margin=float(base.min()),fd_half_max_difference=float(abs(jac-half).max()),
            baseline_margins=base.tolist(),jacobian=jac.tolist())
        if not solve.success:return None,dict(reason='no_original_domain_sampled_solution',**row)
        u=solve.x[:6];assert np.max(abs(u))<=1+1e-12 and np.sum(abs(u-current[w]))<=.1+1e-12
        row['chosen']=u.tolist();row['max_linear_excess']=float(np.max(np.array(A)@solve.x-b));chosen.append(u);rows.append(row)
    chosen=np.array(chosen)
    exact=p.forecast_nominal(env,np.repeat(chosen,13,axis=0));validated=margins(p,exact,q,v,state)
    for w in range(2):
        row=rows[w];row['nonlinear_margins']=validated[13*w].tolist()
        if np.min(validated[13*w])<0:return None,dict(reason='nonlinear_candidate_rejected',**row)
    return chosen,dict(rows=rows,exact_prediction=exact[:,::13].copy())


def run():
    assert not (OUT/'verification.json').exists()
    plans=[np.load(PLANS/f'world{w}_best.npz',allow_pickle=False)['schedule'] for w in range(2)]
    env=NativeEnv.height115_candidate(n=2,scenario=cases()[4:6],residual_scale=0,nominal_correction=True)
    try:
        env.reset();d=env.data;actual=wp.zeros((10,2,env.cpu.nq+env.cpu.nv+12),dtype=D)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,2,[env.reward])
            for slot in range(10):
                wp.launch(command_step,2,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
                wp.launch(env.control_kernel,2,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
                mjw.step(env.model,d);wp.launch(record_forecast,2,[slot,d.qpos,d.qvel,d.ctrl,d.actuator_force,actual])
                wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags]);wp.launch(collect_physical,2,env.physical_args)
                wp.launch(after,2,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        for _ in range(1600):
            wp.capture_launch(capture.graph)
            if np.all(env.state.numpy()[:,1]>=0):break
            assert not env.done.numpy().any()
        else:raise RuntimeError('Original arrival budget exceeded')
        p=Forecaster(env,candidate_count=13,nominal_boundary=True);current=np.zeros((2,6));decisions=[];traces=[];errors=[];failure=None
        for step in range(400):
            started=perf_counter();wanted=np.stack([plan[min(step,len(plan)-1)] for plan in plans])
            chosen,info=allocate(p,env,current,wanted)
            if chosen is None:
                failure=dict(step=step,**info)
                np.savez_compressed(OUT/'failed_state.npz',q=d.qpos.numpy(),v=d.qvel.numpy(),memory=env.k['state'].numpy(),sensor=d.sensordata.numpy(),task=env.state.numpy(),current=current,wanted=wanted)
                break
            prediction=info.pop('exact_prediction');env.set_nominal_correction(chosen);wp.capture_launch(capture.graph);real=actual.numpy()
            qe=float(abs(real[:,:,:p.nq]-prediction[:,:,:p.nq]).max());ve=float(abs(real[:,:,p.nq:p.nq+p.nv]-prediction[:,:,p.nq:p.nq+p.nv]).max())
            ce=float(abs(real[:,:,-12:-6]-prediction[:,:,-12:-6]).max())
            errors.append(dict(qpos=qe,qvel=ve,command=ce))
            traces.append(real.copy());decisions.append(dict(step=step,elapsed_ms=(perf_counter()-started)*1000,request=chosen.tolist(),**info));current=chosen
            if qe>2e-6 or ve>1e-3 or ce>1e-5:
                failure=dict(step=step,reason='original_predictor_pairing_gate_failed',**errors[-1])
                np.savez_compressed(OUT/'failed_pair.npz',predicted=prediction,actual=real);break
            if np.all(env.done.numpy()!=0):break
            if step%50==0:print('PROGRESS',step,errors[-1],flush=True)
        completed=bool(np.all(env.done.numpy()!=0));episodes=[{k:v for k,v in row.items() if k!='terminal_observation'} for row in env.step_wait()[3]] if completed else []
        if traces:np.savez_compressed(OUT/'actual.npz',trace=np.concatenate(traces),requests=np.array([r['request'] for r in decisions]))
        result=dict(role='causal_shared_nominal_sampled_constraint_feedback',completed=completed,failure=failure,episodes=episodes,decisions=decisions,pairing_errors=errors,
            original_pairing_gates=dict(qpos=2e-6,qvel=1e-3,command=1e-5),default_promoted=False,learning_performed=False,
            limitations='Fixed public7kg flat predictor; current full simulator q/v and known past controller/sensor/Actor target only, own forward warmstart. Local5ms FD constraints and nonlinear candidate recheck, not recursive/full-horizon or uncertainty proof. Original1Nm and L1 increment, joint/geometry/task thresholds. Diagnostic stop is not physical safety fallback.')
    finally:env.close()
    result['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_braking_feedback.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',completed,failure,flush=True)


if __name__=='__main__':run()
