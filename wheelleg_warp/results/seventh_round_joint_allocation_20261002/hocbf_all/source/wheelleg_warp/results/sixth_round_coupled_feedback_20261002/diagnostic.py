"""Run-local eight-state braking comparison; retains six-state Nom and original extra box."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from native.design import sim,ml,nominal_input_jacobian
from native.controller import D,V2,V6,PI,MASS,fk,polar_jac
from native.environment import NativeEnv,begin,command_step,reduce_contacts,collect_physical,after
from probe_braking_feedback import execute_extra
from probe_height_115_margin import cases
OUT=Path(__file__).resolve().parent
V8=wp.types.vector(length=8,dtype=D)


@wp.kernel
def update(slot:int,q:wp.array2d[float],v:wp.array2d[float],sensor:wp.array2d[float],ids:wp.array[int],
           active:wp.array[int],task:wp.array2d[D],memory:wp.array2d[D],command:wp.array[D],
           heights:wp.array[D],angles:wp.array[D],reference:wp.array2d[D],gains:wp.array3d[D],feed:wp.array2d[D],
           ctrl:wp.array2d[float],diag:wp.array2d[D],extra:wp.array2d[D],trace:wp.array3d[D]):
    w=wp.tid();trace[slot,w,22]=D(active[w])
    for j in range(6):trace[slot,w,j]=extra[w,j]
    if active[w]==0:return
    desired=V6()
    if task[w,1]>=D(0):
        qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
        pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
        yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
        jl=polar_jac(D(q[w,ids[0]]),D(q[w,ids[1]]));jr=polar_jac(D(q[w,ids[2]]),D(q[w,ids[3]]))
        left=fk(D(q[w,ids[0]]),D(q[w,ids[1]]));right=fk(D(q[w,ids[2]]),D(q[w,ids[3]]))
        length=(left[3]+right[3])/D(2)
        rate=(jl[0,0]*D(v[w,ids[4]])+jl[1,0]*D(v[w,ids[5]])+jr[0,0]*D(v[w,ids[6]])+jr[1,0]*D(v[w,ids[7]]))/D(2)
        arate=(jl[0,1]*D(v[w,ids[4]])+jl[1,1]*D(v[w,ids[5]])+jr[0,1]*D(v[w,ids[6]])+jr[1,1]*D(v[w,ids[7]]))/D(2)
        index=int(0)
        for knot in range(1,heights.shape[0]-1):
            if length>heights[knot]:index=knot
        ratio=wp.clamp((length-heights[index])/(heights[index+1]-heights[index]),D(0),D(1))
        angle=(D(1)-ratio)*angles[index]+ratio*angles[index+1]
        position=D(0)
        if memory[w,13]>D(0):position=wp.cos(yaw)*(D(q[w,0])-memory[w,14])+wp.sin(yaw)*(D(q[w,1])-memory[w,15])
        pdot=D(sensor[w,ids[10]+1]);vx=wp.cos(yaw)*D(v[w,0])+wp.sin(yaw)*D(v[w,1])
        x=V8((left[2]+right[2])/D(2)+D(PI)/D(2)-pitch-angle,arate-pdot,position,vx-command[w],pitch,pdot,length-reference[w,2],rate)
        virtual=wp.vec3d()
        for i in range(3):
            virtual[i]=(D(1)-ratio)*feed[index,i]+ratio*feed[index+1,i]
            for j in range(8):virtual[i]-=((D(1)-ratio)*gains[index,i,j]+ratio*gains[index+1,i,j])*x[j]
        # Preserve the existing unilateral outward guard and differential roll/yaw Nom.
        support=(D(1)-ratio)*feed[index,2]+ratio*feed[index+1,2]
        old_force=support+D(MASS)*(D(500)*(reference[w,2]-length)-D(25)*rate)
        delta=V2(virtual[2]-old_force,virtual[1]-(diag[w,29]+diag[w,30])/D(2))
        dl=jl*delta;dr=jr*delta
        dw=virtual[0]-(D(ctrl[w,4])+D(ctrl[w,5]))/D(2)
        desired=V6(dl[0],dl[1],dr[0],dr[1],dw,dw)
    change=V6();total=D(0)
    for j in range(6):
        change[j]=wp.clamp(desired[j],D(-1),D(1))-extra[w,j];total+=wp.abs(change[j])
    scale=D(1)
    if total>D(.01):scale=D(.01)/total
    for j in range(6):extra[w,j]+=scale*change[j];trace[slot,w,6+j]=extra[w,j]


@wp.kernel
def record(slot:int,q:wp.array2d[float],ids:wp.array[int],diag:wp.array2d[D],trace:wp.array3d[D]):
    w=wp.tid()
    for j in range(6):trace[slot,w,12+j]=diag[w,6+j]
    for j in range(4):trace[slot,w,18+j]=D(q[w,ids[j]])


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    r=json.loads((OUT/'verification.json').read_text());z=np.load(OUT/'trace.npz')['trace']
    valid=z[:,:,22]>0
    assert np.isfinite(z[valid]).all()
    assert np.max(abs(z[:,:,6:12][valid]))<=1.+1e-12
    assert np.max(np.sum(abs(z[:,:,6:12]-z[:,:,:6]),axis=2)[valid])<=.01+1e-12
    counts=valid.sum(axis=0)
    assert counts.tolist()==[row['physical_steps'] for row in r['episodes']]
    assert all(row['physical_steps']==row['physical_evidence_steps'] for row in r['episodes'])
    assert sha(OUT/'trace.npz')==r['trace_sha256']
    for name,h in r['source_sha256'].items():assert sha(OUT/'source'/name)==h,name
    margin=[float(np.min(1.4-abs(z[valid[:,w],w,18:22]))) for w in range(6)]
    assert np.allclose(margin,r['active_design_margin_rad'],rtol=0,atol=1e-12)
    passed=bool(all(row['success'] and row['physical_safety_passed'] for row in r['episodes']) and min(margin)>=0)
    assert passed==r['full_normal_gate_pass']
    print('CHECKED request box/slew, complete evidence, archived sources, active design reserve; full gate',passed,flush=True)


def run():
    assert not (OUT/'verification.json').exists()
    model,_=sim.load_model(ml.XML,True);gains=[];reports=[]
    for h in (.115,sim.L_SQUAT_MIN,sim.L_PREP,sim.L_STAND,sim.L_MAX):
        ref,a,b,_=ml.design(model,h,min_height=.115);c,g,_=ml.vmc_coordinates(ref)
        outer=np.zeros((3,15));outer[2]=sim.hw.DESIGN_MASS/sim.hw.BASELINE_MASS*(sim.KP_LEG*c[6]+sim.KD_LEG*c[7])
        gain,r=ml.reduced_design(model,ref,a+b@(nominal_input_jacobian(model,ref)+g@outer),b,8)
        assert r['linear_pass'];k=np.linalg.solve(g,gain)@np.linalg.pinv(c)
        assert np.allclose(g@k@c,gain,rtol=1e-8,atol=1e-8)
        gains.append(k);reports.append(dict(height_m=h,**r))
    scenes=cases()[:6]
    (OUT/'registered_cases.json').write_text(json.dumps([asdict(s) for s in scenes],indent=2)+'\n')
    env=NativeEnv.height115_candidate(n=6,scenario=scenes,residual_scale=0)
    try:
        env.reset();d=env.data;extra=wp.zeros((6,6),dtype=D);kg=wp.array(np.stack(gains),dtype=D);trace=wp.zeros((40,6,23),dtype=D)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,6,[env.reward])
            for slot in range(40):
                wp.launch(command_step,6,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
                wp.launch(env.control_kernel,6,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
                wp.launch(update,6,[slot,d.qpos,d.qvel,d.sensordata,env.ids,env.active,env.state,env.k['state'],env.command,env.k['heights'],env.k['angles'],env.k['reference'],kg,env.k['feed'],d.ctrl,env.diag,extra,trace])
                wp.launch(execute_extra,6,[d.qvel,env.ids,env.control_extra[0],extra,env.active,d.ctrl,env.diag]);mjw.step(env.model,d)
                wp.launch(record,6,[slot,d.qpos,env.ids,env.diag,trace])
                wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags]);wp.launch(collect_physical,6,env.physical_args)
                wp.launch(after,6,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        chunks=[]
        for iteration in range(700):
            wp.capture_launch(capture.graph);chunks.append(trace.numpy().copy())
            if np.all(env.done.numpy()!=0):break
            if iteration%100==0:print('PROGRESS',iteration,flush=True)
        assert np.all(env.done.numpy()!=0)
        rows=[{k:v for k,v in row.items() if k!='terminal_observation'} for row in env.step_wait()[3]]
        z=np.concatenate(chunks);margin=[float(np.min(1.4-abs(z[z[:,w,22]>0,w,18:22]))) for w in range(6)]
        np.savez_compressed(OUT/'trace.npz',trace=z)
        np.savez_compressed(OUT/'design.npz',gains8=np.stack(gains),gains6=env.k['gains'].numpy(),feed=env.k['feed'].numpy(),angles=env.k['angles'].numpy(),heights=env.k['heights'].numpy())
    finally:env.close()
    sources={}
    for p in (Path(__file__),ROOT/'wheelleg_warp/native/design.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/probe_braking_feedback.py',ROOT/'wheelleg_warp/training_contract.py',ROOT/'wheelleg_ppo/tools/model_lqr.py'):
        name=str(p.relative_to(ROOT));dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());sources[name]=sha(p)
    result=dict(role='bounded_eight_state_common_braking_diagnostic',scenarios=[asdict(s) for s in scenes],episodes=rows,linear_design_reports=reports,active_design_margin_rad=margin,
        physical_pass_count=sum(r['physical_safety_passed'] for r in rows),task_pass_count=sum(r['success'] for r in rows),
        full_normal_gate_pass=bool(all(r['success'] and r['physical_safety_passed'] for r in rows) and min(margin)>=0),
        trace_sha256=sha(OUT/'trace.npz'),source_sha256=sources,default_changed=False,
        limitations='One fixed7kg design with existing eight-state Q/R, braking-only correction of six-state Nom; per-motor1Nm and per0.5ms L1 change0.01Nm. Guard retained. Whole-node linear results do not certify this bounded nonlinear branch. Six public nominal scenarios only, no robustness or all-height claim, no PPO action expansion or default promotion.')
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED physical',result['physical_pass_count'],'task',result['task_pass_count'],'design margin',margin,flush=True)
    print([(r['stop_distance_m'],r['tail_speed_m_s']) for r in rows],flush=True);check()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    check() if args.check else run()
