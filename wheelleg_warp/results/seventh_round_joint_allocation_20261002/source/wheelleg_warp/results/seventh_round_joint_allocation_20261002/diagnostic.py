"""Run-local nominal-state joint allocation diagnostic; no trained-policy or default change."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw
from scipy.optimize import linprog
from native.controller import D
from native.environment import NativeEnv,begin,command_step,reduce_contacts,collect_physical,after
from native.terrain import model,HeightTerrainScenario
from probe_braking_feedback import execute_extra
from probe_height_115_margin import cases
from probe_height_115_passive import JOINTS
from probe_height_115_braking_budget import barriers
from probe_height_115_radial_authority import torque_box
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent
PREVIOUS=OUT.parent/'sixth_round_coupled_feedback_20261002'
spec=importlib.util.spec_from_file_location('coupled_request',PREVIOUS/'diagnostic.py')
request=importlib.util.module_from_spec(spec);spec.loader.exec_module(request)


def allocate(m,q,v,nominal,wanted,current):
    """Nearest motor request in the original box, with local stopping/position rows."""
    dofs=[m.joint(n).dofadr[0] for n in JOINTS];qa=[m.joint(n).qposadr[0] for n in JOINTS]
    d=mujoco.MjData(m)
    def acceleration(u):
        d.qpos[:]=q;d.qvel[:]=v;d.ctrl[:]=u;d.qacc_warmstart[:]=0
        mujoco.mj_forward(m,d)
        return d.qacc[dofs].copy()
    a=acceleration(nominal)
    g=np.column_stack([(acceleration(nominal+eps)-acceleration(nominal-eps))/.002 for eps in np.eye(6)*.001])
    gh=np.column_stack([(acceleration(nominal+eps)-acceleration(nominal-eps))/.001 for eps in np.eye(6)*.0005])
    assert np.isfinite(g).all() and np.allclose(g,gh,rtol=.01,atol=.01)
    A=[];rhs=[]
    def add(coeff,b):A.append(coeff);rhs.append(b)
    # Variables: six motor extras, L-infinity request distance, six slew auxiliaries.
    for j in range(6):
        row=np.zeros(13);row[j]=1;row[6]=-1;add(row,wanted[j])
        row=np.zeros(13);row[j]=-1;row[6]=-1;add(row,-wanted[j])
        row=np.zeros(13);row[j]=1;row[7+j]=-1;add(row,current[j])
        row=np.zeros(13);row[j]=-1;row[7+j]=-1;add(row,-current[j])
    row=np.zeros(13);row[7:]=1;add(row,.1)
    caps=np.array([1.4,1.4,2.5,2.5,1.4,1.4,2.5,2.5]);pos=q[qa];vel=v[dofs]
    if np.any(caps-abs(pos)<=0):return None,dict(reason='original_design_position_outside')
    # Existing constant stopping-acceleration rule; full q/v dynamic validation remains required.
    for j in range(8):
        for sign in (-1,1):
            h=caps[j]+sign*pos[j];rate=sign*vel[j]
            if rate<0:
                row=np.zeros(13);row[:6]=-sign*g[j];add(row,sign*a[j]-rate*rate/(2*h))
    # Same leg-length stopping rows as the existing assembler, with its original numeric padding.
    active=[0,1,4,5];bs,_=barriers(pos[active],vel[active],a[active],g[active],(0.,.1))
    if any(b['h']<=0 for b in bs[:2]):return None,dict(reason='original_leg_padding_outside')
    for b in bs[:2]:
        if b['rate']<0:
            row=np.zeros(13);row[:6]=-b['g'];add(row,b['a0']-b['rate']**2/(2*b['h']))
    # Every substep in the held5ms interval, then the existing200/s terminal joint cone.
    for t in np.arange(1,11)*.0005:
        for sign in (-1,1):
            for j in range(8):
                row=np.zeros(13);row[:6]=-sign*.5*t*t*g[j]
                add(row,caps[j]+sign*(pos[j]+t*vel[j]+.5*t*t*a[j]))
    t=.005;k=200.
    for sign in (-1,1):
        for j in range(8):
            row=np.zeros(13);row[:6]=-sign*(k*.5*t*t+t)*g[j]
            add(row,k*(caps[j]+sign*(pos[j]+t*vel[j]+.5*t*t*a[j]))+sign*(vel[j]+t*a[j]))
    bound=torque_box(m,v)/np.array([1.,1.,1.,1.,1.05,1.05])
    box=[(max(-1.,-bound[j]-nominal[j]),min(1.,bound[j]-nominal[j])) for j in range(6)]
    cost=np.zeros(13);cost[6]=1
    solved=linprog(cost,A_ub=A,b_ub=rhs,bounds=box+[(0,None)]+[(0,None)]*6,method='highs')
    info=dict(status=int(solved.status),message=solved.message,a0=a.tolist(),G=g.tolist(),q_joint=pos.tolist(),v_joint=vel.tolist(),
        wanted=wanted.tolist(),current=current.tolist(),public_motor_bound=bound.tolist(),fd_half_max_error=float(abs(g-gh).max()))
    if not solved.success:return None,dict(reason='no_original_domain_local_solution',**info)
    x=solved.x;excess=float(np.max(np.array(A)@x-rhs))
    assert excess<=1e-7 and np.sum(abs(x[:6]-current))<=.1+1e-7 and np.max(abs(x[:6]))<=1+1e-7
    return x[:6],dict(max_constraint_excess=excess,request_distance=float(x[6]),**info)


def run():
    assert not (OUT/'verification.json').exists()
    old=json.loads((PREVIOUS/'verification.json').read_text());assert sha(PREVIOUS/'diagnostic.py')==old['source_sha256'][str((PREVIOUS/'diagnostic.py').relative_to(ROOT))]
    gains=np.load(PREVIOUS/'design.npz')['gains8'];scenes=cases()[:6]
    (OUT/'registered_cases.json').write_text(json.dumps([asdict(s) for s in scenes],indent=2)+'\n')
    env=NativeEnv.height115_candidate(n=6,scenario=scenes,residual_scale=0);nom=model(HeightTerrainScenario(stand_height_m=.115))
    try:
        env.reset();d=env.data;extra=wp.zeros((6,6),dtype=D);wanted=wp.zeros_like(extra);kg=wp.array(gains,dtype=D)
        scratch=wp.zeros((1,6,23),dtype=D);trace=wp.zeros((10,6,23),dtype=D)
        def prepare():
            wp.launch(command_step,6,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(env.control_kernel,6,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
        with wp.ScopedCapture() as prepared:
            wp.launch(begin,6,[env.reward]);prepare();wp.copy(wanted,extra)
            # Reuse the frozen request generator: ten0.01 increments give at most0.1 per5ms.
            for _ in range(10):wp.launch(request.update,6,[0,d.qpos,d.qvel,d.sensordata,env.ids,env.active,env.state,env.k['state'],env.command,env.k['heights'],env.k['angles'],env.k['reference'],kg,env.k['feed'],d.ctrl,env.diag,wanted,scratch])
        with wp.ScopedCapture() as advanced:
            for slot in range(10):
                if slot>0:prepare()
                wp.launch(execute_extra,6,[d.qvel,env.ids,env.control_extra[0],extra,env.active,d.ctrl,env.diag]);mjw.step(env.model,d)
                wp.launch(request.record,6,[slot,d.qpos,env.ids,env.diag,trace])
                wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags]);wp.launch(collect_physical,6,env.physical_args)
                wp.launch(after,6,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        current=np.zeros((6,6));chunks=[];logs=[];failure=None;durations=[]
        for iteration in range(2800):
            wp.capture_launch(prepared.graph);state=env.state.numpy();active=env.active.numpy()!=0
            stopped=(state[:,1]>=0)&active
            if stopped.any():
                q=d.qpos.numpy().astype(float);v=d.qvel.numpy().astype(float);base=d.ctrl.numpy().astype(float);target=wanted.numpy()
                proposed=current.copy();decisions=[]
                for w in np.flatnonzero(stopped):
                    tic=perf_counter();u,info=allocate(nom,q[w],v[w],base[w],target[w],current[w]);durations.append((perf_counter()-tic)*1000)
                    if u is None:failure=dict(iteration=iteration,world=int(w),time_s=float(state[w,0]*.0005),arrival_s=float(state[w,1]),**info);break
                    proposed[w]=u;decisions.append(dict(world=int(w),**info,chosen=u.tolist()))
                if failure is not None:
                    np.savez_compressed(OUT/'failed_state.npz',q=q,v=v,nominal=base,wanted=target,current=current,task_state=state,controller_state=env.k['state'].numpy());break
                logs.append(dict(iteration=iteration,decisions=decisions));current=proposed;extra.assign(current)
            wp.capture_launch(advanced.graph);chunks.append(dict(joints=trace.numpy()[:,:,18:22].copy(),active=active.copy(),extra=current.copy()))
            if np.all(env.done.numpy()!=0):break
            if iteration%200==0:print('PROGRESS',iteration,'allocations',len(logs),flush=True)
        completed=bool(np.all(env.done.numpy()!=0));rows=[{k:v for k,v in row.items() if k!='terminal_observation'} for row in env.step_wait()[3]] if completed else []
        np.savez_compressed(OUT/'trace.npz',joints=np.stack([c['joints'] for c in chunks]),active=np.stack([c['active'] for c in chunks]),extra=np.stack([c['extra'] for c in chunks]))
        result=dict(role='fixed_nominal_joint_allocation_diagnostic',completed=completed,failure=failure,episodes=rows,scenarios=[asdict(s) for s in scenes],decisions=logs,
            physical_pass_count=sum(r['physical_safety_passed'] for r in rows),task_pass_count=sum(r['success'] for r in rows),
            solver_ms=dict(median=float(np.median(durations)),p95=float(np.percentile(durations,95))) if durations else None,
            trace_sha256=sha(OUT/'trace.npz'),default_changed=False,original_gates_unchanged=True,
            limitations='One fixed7kg flat CPU nominal model, known current full simulator state and Nom, own forward solve without plant warmstart. Local constant acceleration/stopping inequalities, held5ms extra with Nom recomputed each0.5ms; not an invariance or full-horizon task certificate. Existing eight-state request only, no gain tuning, no final holdout, no PPO rights expansion. On infeasibility diagnostic stops without executing an unverified allocation; this is not a physical safety fallback.')
    finally:env.close()
    sources={}
    for p in (Path(__file__),PREVIOUS/'diagnostic.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/probe_height_115_braking_budget.py'):
        name=str(p.relative_to(ROOT));dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());sources[name]=sha(p)
    result['source_sha256']=sources
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED',completed,'physical',result['physical_pass_count'],'task',result['task_pass_count'],'failure',None if failure is None else {k:failure[k] for k in ('iteration','world','time_s','reason')},flush=True)


def check():
    r=json.loads((OUT/'verification.json').read_text());z=np.load(OUT/'trace.npz');u=z['extra'];valid=z['active']
    assert sha(OUT/'trace.npz')==r['trace_sha256'] and np.isfinite(u).all()
    assert np.max(abs(u[valid]))<=1.+1e-7
    previous=np.concatenate([np.zeros_like(u[:1]),u[:-1]])
    assert np.max(np.sum(abs(u-previous),axis=2)[valid])<=.1+1e-7
    for name,h in r['source_sha256'].items():assert sha(OUT/'source'/name)==h
    if r['failure'] is not None:
        s=np.load(OUT/'failed_state.npz');f=r['failure'];w=f['world']
        m=model(HeightTerrainScenario(stand_height_m=.115))
        chosen,info=allocate(m,s['q'][w],s['v'][w],s['nominal'][w],s['wanted'][w],s['current'][w])
        assert chosen is None and info['reason']==f['reason'] and not r['completed']
    print('CHECKED domain, source, trace and frozen infeasibility; completed',r['completed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    check() if args.check else run()
