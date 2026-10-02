"""One frozen nominal plan, same-version full episodes and registered parameter panel."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from native.controller import D,control_physical,V2,fk,polar_jac,inverse2,PI
from native.environment import NativeEnv,begin,command_step,reduce_contacts,collect_physical,after
from native.terrain import HeightTerrainScenario
from probe_height_115_margin import cases
from probe_braking_feedback import execute_extra
from check_braking_trajectory import check as check_original
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'braking_trajectory_slsqp_20261001'
REGISTRY=OUT/'single_factor_cases.json'


@wp.kernel
def record(slot:int,q:wp.array2d[float],ids:wp.array[int],active:wp.array[int],diag:wp.array2d[D],ctrl:wp.array2d[float],shadow:wp.array2d[float],out:wp.array3d[D]):
    w=wp.tid();out[slot,w,4]=D(active[w])
    for j in range(4):out[slot,w,j]=D(q[w,ids[j]])
    amount=D(0)
    for j in range(6):amount+=wp.abs(diag[w,6+j])
    out[slot,w,5]=amount;out[slot,w,6]=diag[w,12]
    difference=D(0)
    for j in range(6):difference=wp.max(difference,wp.abs(D(ctrl[w,j])-D(shadow[w,j])))
    out[slot,w,7]=difference



@wp.kernel
def record_reference(slot:int,q:wp.array2d[float],ids:wp.array[int],active:wp.array[int],diag:wp.array2d[D],
                     before:wp.array2d[float],after_ctrl:wp.array2d[float],heights:wp.array[D],gains:wp.array3d[D],
                     reference:wp.array2d[D],out:wp.array3d[D]):
    w=wp.tid()
    out[slot,w,0]=D(active[w])
    left=fk(D(q[w,ids[0]]),D(q[w,ids[1]]));right=fk(D(q[w,ids[2]]),D(q[w,ids[3]]))
    length=(left[3]+right[3])/D(2);index=int(0)
    for knot in range(1,heights.shape[0]-1):
        if length>heights[knot]:index=knot
    ratio=wp.clamp((length-heights[index])/(heights[index+1]-heights[index]),D(0),D(1))
    kg=(D(1)-ratio)*gains[index,1,0]+ratio*gains[index+1,1,0]
    out[slot,w,1]=diag[w,21];out[slot,w,2]=diag[w,22];out[slot,w,3]=reference[w,5];out[slot,w,4]=kg
    out[slot,w,5]=left[2]+D(PI)/D(2);out[slot,w,6]=right[2]+D(PI)/D(2)
    out[slot,w,7]=diag[w,27];out[slot,w,8]=diag[w,25];out[slot,w,9]=diag[w,26]
    for side in range(2):
        j=2*side;matrix=inverse2(polar_jac(D(q[w,ids[j]]),D(q[w,ids[j+1]])))
        a=matrix*V2(D(before[w,j]),D(before[w,j+1]))
        b=matrix*V2(D(after_ctrl[w,j]),D(after_ctrl[w,j+1]))
        out[slot,w,10+side]=out[slot,w,5+side]+(a[1]-diag[w,27])/kg
        out[slot,w,12+side]=out[slot,w,5+side]+(b[1]-diag[w,27])/kg
        out[slot,w,14+side]=b[0]-a[0];out[slot,w,16+side]=b[1]-a[1]
    for j in range(4):out[slot,w,18+j]=D(q[w,ids[j]])
    for j in range(6):
        out[slot,w,22+j]=D(before[w,j]);out[slot,w,28+j]=D(after_ctrl[w,j])


def run():
    assert not (OUT/'verification.json').exists()
    variations=[('nominal',{}),('mass_only',dict(mass=7.5)),('friction_only',dict(mu_l=.6,mu_r=1.)),('drive_only',dict(drive_difference=.03))]
    scenes=[replace(s,**change) for _,change in variations for s in cases()[:6]]
    REGISTRY.write_text(json.dumps(dict(labels=[label for label,_ in variations for _ in range(6)],scenarios=[asdict(s) for s in scenes]),indent=2)+'\n')
    check_original(SOURCE)
    plans=[np.load(SOURCE/f'world{w}_best.npz',allow_pickle=False)['schedule'] for w in range(2)]
    registered=json.loads(REGISTRY.read_text());scenes=[HeightTerrainScenario(**s) for s in registered['scenarios']];n=len(scenes)
    assert n==24 and all(s.stand_height_m==.115 for s in scenes)
    selected=[0 if s.speed==1. else 1 if s.speed==-1. else -1 for s in scenes]
    registration=dict(selection='registered four single-factor settings crossed with original six normal cases; original plans, no gain or plan search',parameter_groups=registered['labels'],scenarios=[asdict(s) for s in scenes],
        plan_by_world=selected,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [REGISTRY,SOURCE/'verification.json']+[SOURCE/f'world{w}_best.npz' for w in range(2)]})
    (OUT/'registered_cases.json').write_text(json.dumps(registration,indent=2)+'\n')
    env=NativeEnv.height115_candidate(n=n,scenario=scenes,residual_scale=0,nominal_correction=True)
    try:
        assert env.observation_space.shape==(38,)
        env.reset();d=env.data;extra=wp.zeros((n,6),dtype=D);trace=wp.zeros((10,n,8),dtype=D)
        reference_trace=wp.zeros((10,n,34),dtype=D);reference_chunks=[];base_ctrl=wp.zeros_like(d.ctrl)
        shadow_state=wp.zeros_like(env.k['state']);shadow_ctrl=wp.zeros_like(d.ctrl);shadow_diag=wp.zeros_like(env.diag)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,n,[env.reward])
            for slot in range(10):
                wp.launch(command_step,n,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
                wp.copy(shadow_state,env.k['state'])
                wp.launch(control_physical,n,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,shadow_state,env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],shadow_ctrl,shadow_diag,0,0]+env.control_extra[:1],block_dim=32)
                wp.copy(base_ctrl,shadow_ctrl)
                wp.launch(execute_extra,n,[d.qvel,env.ids,env.control_extra[0],env.nominal_correction,env.active,shadow_ctrl,shadow_diag])
                wp.launch(env.control_kernel,n,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
                wp.launch(record_reference,n,[slot,d.qpos,env.ids,env.active,env.diag,base_ctrl,d.ctrl,env.k['heights'],env.k['gains'],env.k['reference'],reference_trace])
                mjw.step(env.model,d)
                wp.launch(record,n,[slot,d.qpos,env.ids,env.active,env.diag,d.ctrl,shadow_ctrl,trace])
                wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags]);wp.launch(collect_physical,n,env.physical_args)
                wp.launch(after,n,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        cursor=np.full(n,-1,int);current=np.zeros((n,6));chunks=[];actions=[];boundaries=[]
        for iteration in range(2800):
            state=env.state.numpy();active=env.active.numpy()!=0
            for w in np.flatnonzero(active):
                if selected[w]<0 or state[w,1]<0:continue
                if cursor[w]<0:boundaries.append(dict(world=int(w),start_s=float(state[w,0]*.0005),arrival_s=float(state[w,1])))
                cursor[w]+=1;plan=plans[selected[w]];current[w]=plan[min(cursor[w],len(plan)-1)]
            env.set_nominal_correction(current);wp.capture_launch(capture.graph);chunks.append(trace.numpy().copy());reference_chunks.append(reference_trace.numpy().copy());actions.append(current.copy())
            if np.all(env.done.numpy()!=0):break
            if iteration%400==0:print('PROGRESS',iteration,'started plans',len(boundaries),flush=True)
        assert np.all(env.done.numpy()!=0)
        rows=[{k:v for k,v in row.items() if k!='terminal_observation'} for row in env.step_wait()[3]]
        z=np.concatenate(chunks);margin=[float(np.min(1.4-abs(z[z[:,w,4]>0,w,:4]))) for w in range(n)]
        np.savez_compressed(OUT/'trace.npz',trace=z,extra=np.stack(actions))
        np.savez_compressed(OUT/'reference_trace.npz',trace=np.concatenate(reference_chunks))
        gates=[bool(r['success'] and r['physical_safety_passed'] and m>=0) for r,m in zip(rows,margin)]
        groups={name:dict(physical_pass_count=sum(rows[w]['physical_safety_passed'] for w,label in enumerate(registered['labels']) if label==name),task_pass_count=sum(rows[w]['success'] for w,label in enumerate(registered['labels']) if label==name),
            full_design_and_task_pass_count=sum(gates[w] for w,label in enumerate(registered['labels']) if label==name)) for name in dict.fromkeys(registered['labels'])}
        result=dict(role='read_only_reference_order_instrumentation',base_candidate_version=env.baseline_version,observation_spec=env.observation_spec,registration=registration,episodes=rows,
            active_design_margin_rad=margin,original_full_gates=gates,groups=groups,start_boundaries=boundaries,trace_sha256=sha(OUT/'trace.npz'),
            actual_future_used_for_selection=False,plans_retuned=False,default_promoted=False,learning_performed=False,
            limitations='Frozen7kg nominal two-second plans, selected only by public commanded speed/sign at known arrival phase. Other four scenarios have zero added action. Current actor is zero: delay labels do not prove policy/closed-loop delay robustness. Common correction is now part of accepted Nom before Actor; residual monitor and penalty contain Actor only. This remains an explicit non-default frozen-plan candidate, not robust control or training admission. No online adaptation, recursive safety, full-height, IID robustness or paper method advantage claim.')
    finally:env.close()
    result['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/design.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/probe_braking_feedback.py',ROOT/'wheelleg_warp/training_contract.py')}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETED',groups,flush=True);print('HIGH SPEED',[(w,rows[w]['stop_distance_m'],rows[w]['tail_speed_m_s'],margin[w]) for w,p in enumerate(selected) if p>=0],flush=True)
    check()


def check():
    r=json.loads((OUT/'verification.json').read_text());z=np.load(OUT/'trace.npz');t=z['trace'];u=z['extra'];valid=t[:,:,4]>0
    assert sha(OUT/'trace.npz')==r['trace_sha256'] and np.isfinite(t[valid]).all() and np.isfinite(u).all()
    counts=valid.sum(axis=0);assert counts.tolist()==[row['physical_steps'] for row in r['episodes']]
    assert np.all(t[:,:,5][valid]==0), 'Classical correction charged as Actor residual'
    assert np.max(t[:,:,7][valid])<=1e-6, 'Same-input total-command correspondence failed'
    assert all(row['physical_steps']==row['physical_evidence_steps'] for row in r['episodes'])
    delta=np.diff(np.concatenate([np.zeros_like(u[:1]),u]),axis=0)
    assert np.max(abs(u))<=1.+1e-12 and np.max(np.sum(abs(delta),axis=2))<=.1+1e-12
    margin=[float(np.min(1.4-abs(t[valid[:,w],w,:4]))) for w in range(len(r['episodes']))]
    np.testing.assert_allclose(margin,r['active_design_margin_rad'],rtol=0,atol=1e-12)
    gates=[bool(row['success'] and row['physical_safety_passed'] and m>=0) for row,m in zip(r['episodes'],margin)]
    assert gates==r['original_full_gates']
    for name,h in r['registration']['input_sha256'].items():assert sha(ROOT/name)==h
    print('CHECKED complete2kHz evidence, original per-motor/slew domain, active design gates and frozen inputs; full gates',sum(gates),'/',len(gates),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    check() if a.check else run()
