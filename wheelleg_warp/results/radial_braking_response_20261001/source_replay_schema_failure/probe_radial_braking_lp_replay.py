"""Nonlinear CPU replay of the fixed nominal braking LP; stop at first physical failure."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
import wheelleg_sim as sim
from state_estimation import leg_kinematics
from model_lqr import sagittal_basis,vmc_coordinates
from native.terrain import model as terrain_model,TerrainScenario,HEIGHT_115_GEOMETRIC_MIN as LIMIT
from probe_height_115_passive import geometry,JOINTS
from probe_height_115_action_predict_loow import sha


def run(source):
    target=source/'nonlinear_lp_replay.json';assert not target.exists()
    meta=json.loads((source/'verification.json').read_text());columns=meta['columns']
    lp=np.load(source/'linear_braking_lp.npz',allow_pickle=False)
    archive=np.load(source/'windows.npz',allow_pickle=False);r=archive['w4_braking_radial']
    initial=archive['w4_braking_pre'][np.flatnonzero(r[:,2]==0)[0]]
    m=terrain_model(TerrainScenario(**meta['scenarios'][4]));d=mujoco.MjData(m)
    d.qpos[:]=initial[1:columns['qvel_start']];d.qvel[:]=initial[columns['qvel_start']:columns['warmstart_start']]
    d.qacc_warmstart[:]=initial[columns['warmstart_start']:columns['ctrl_start']];mujoco.mj_forward(m,d)
    ref=mujoco.MjData(m);ref.qpos[:]=lp['reference_q'];ref.ctrl[:]=lp['reference_ctrl']
    _,g,_=vmc_coordinates(ref);_,mapping=sagittal_basis(m);feed=np.linalg.solve(g,np.linalg.pinv(mapping)@ref.ctrl)
    indices=np.array([m.joint(name).qposadr[0] for name in JOINTS]);ranges=np.array([m.jnt_range[m.joint(name).id] for name in JOINTS])
    mid=np.array([(m.body_pos[m.body('leg'+s).id]+m.body_pos[m.body('leg'+s+'_D').id])/2 for s in ('L','R')])
    qp=[];vp=[];torques=[];metrics=[];origin=d.qpos[:2].copy();clipped=0;failure=None
    for step in range(4000):
        virtual=lp['inputs'][step//10];issued=np.zeros(6);bounds=np.zeros(6)
        for side,suffix in enumerate(('L','R')):
            joints=[m.joint(n+suffix) for n in ('alpha','beta')]
            q=np.array([d.qpos[j.qposadr[0]] for j in joints]);v=np.array([d.qvel[j.dofadr[0]] for j in joints])
            leg,_,_,jac=leg_kinematics(q,v);rate=(jac.T@v)[0]
            force=feed[2]+sim.hw.DESIGN_MASS/sim.hw.BASELINE_MASS*(500*(.115-np.linalg.norm(leg))-25*rate)
            issued[2*side:2*side+2]=jac@np.array([force,virtual[1]])
        issued[4:]=virtual[0]
        for j,name in enumerate(('alphaL','betaL','alphaR','betaR','wheel1','wheel2')):
            speed=d.qvel[m.joint(name).dofadr[0]]
            bounds[j]=sim.hw.torque_limit(1e6,speed,j<4,0.,.0005)[0]/(1.05 if j>=4 else 1.)
        clipped+=int(np.max(abs(issued)-bounds)>1e-6);d.ctrl[:]=np.clip(issued,-bounds,bounds)
        mujoco.mj_step(m,d);q=d.qpos.copy();v=d.qvel.copy();geo=geometry(m,q)
        actual=float(min(geo[1].min(),np.linalg.norm(geo[5]-mid,axis=-1).min()))
        margin=float(np.minimum(q[indices]-ranges[:,0],ranges[:,1]-q[indices]).min());att=float(abs(np.rad2deg(sim.euler(d))).max())
        force_excess=float(np.max(abs(d.actuator_force)-bounds));distance=float(np.linalg.norm(q[:2]-origin));speed=float(np.linalg.norm(v[:2]))
        qp.append(q);vp.append(v);torques.append(d.ctrl.copy());metrics.append([actual,margin,att,force_excess,distance,speed])
        failed=[]
        if not np.isfinite(np.r_[q,v]).all():failed.append('nonfinite')
        if actual<LIMIT:failed.append('actual_A_B_geometry')
        if margin<0:failed.append('eight_joint_range')
        if att>5:failed.append('attitude')
        if force_excess>1e-6:failed.append('actual_torque')
        if distance>.6:failed.append('stop_distance')
        if step>=2999 and speed>.03:failed.append('tail_speed')
        if failed:failure=dict(step=step+1,time_s=(step+1)*.0005,reasons=failed);break
    metrics=np.asarray(metrics);full=len(metrics)==4000 and failure is None
    np.savez_compressed(source/'nonlinear_lp_replay.npz',qpos=qp,qvel=vp,ctrl=torques,metrics=metrics)
    report=dict(role='nominal_two_input_lp_nonlinear_cpu_replay',full_braking_gate_pass=full,completed_steps=len(metrics),first_failure=failure,
        min_actual_A_B_m=float(metrics[:,0].min()),min_eight_joint_margin_rad=float(metrics[:,1].min()),max_attitude_deg=float(metrics[:,2].max()),
        max_actual_torque_excess_Nm=float(metrics[:,3].max()),max_distance_m=float(metrics[:,4].max()),command_clipped_steps=clipped,
        limitations='Offline measured Warp initial state and planned5ms inputs, nominal CPU plant. Radial PD matches linear model, no experimental radial guard. Stops at first failure; no failed-prefix endpoint can count as a complete stop. No feedback, deployment, paired backend or six-task claim.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/audit_radial_braking_response.py')})
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    run(parser.parse_args().source)
