"""Registered400step CPU nominal-flow check;not actualNom controller."""
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import model_lqr as ml
import wheelleg_sim as sim
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/relative_reference_qualification_v1'
JOINTS=('alphaL','betaL','alphaR','betaR','passA_L','passC_L','passA_R','passC_R')
MOTORS=('alphaL','betaL','alphaR','betaR','wheel1','wheel2')
COL=['actual_A_L','actual_A_R','actual_B_L','actual_B_R','loop_L','loop_R',
     'active_design_margin_rad','eight_joint_margin_rad','command_excess_Nm','force_excess_Nm',
     'pre_gyro_x','pre_gyro_y','pre_gyro_z','post_roll','post_pitch','post_yaw']


def contacts(m,d):
    result=[]
    for i in range(d.ncon):
        c=d.contact[i];force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        velocities=[]
        for geom in c.geom:
            v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,int(geom),v,0)
            velocities.append(v[3:]+np.cross(v[:3],c.pos-d.geom_xpos[int(geom)]))
        relative=velocities[1]-velocities[0]
        result.append(dict(geom=list(map(int,c.geom)),point=c.pos.tolist(),frame=c.frame.tolist(),
                          local_force_torque=force.tolist(),relative_point_velocity_contact_frame=(c.frame.reshape(3,3)@relative).tolist()))
    return result


def run():
    assert not any((OUT/n).exists() for n in ('source_contract.json','completion.json','failure.json'))
    p=json.loads((OUT/'proposal.json').read_text());assert p['cpu_integration_step_budget']==400
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    for r in p['reference_inputs']:assert sha(ROOT/r['path'])==r['sha256']
    m,_=sim.load_model(ml.XML,True)
    assert (m.nq,m.nv,m.nu,m.na,m.nmocap,m.nplugin)==(17,16,6,0,0,0)
    assert m.opt.timestep==.0005 and m.opt.iterations==100 and int(m.opt.integrator)==3
    assert abs(m.body_mass.sum()-7.)<1e-12
    assert mujoco.get_mjcb_control() is None and mujoco.get_mjcb_passive() is None
    qa=np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
    va=np.array([m.jnt_dofadr[m.joint(n).id] for n in MOTORS])
    limits=m.jnt_range[[m.joint(n).id for n in JOINTS]]
    gyro=m.sensor('body_gyro').adr[0];sources={**p['source_sha256'],'wheelleg_warp/qualify_relative_reference.py':sha(__file__),str(Path(ml.XML).relative_to(ROOT)):sha(ml.XML)}
    atomic_json(OUT/'source_contract.json',dict(verified=True,proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,
        mujoco_version=mujoco.__version__,step_budget=400,transitionFD_calls=0,
        metric_time='Contactforce/frame/velocity/gyro from solver preintegration configuration;geometry/roll and q/v from postintegration. Kinematics-only refresh,not extra forward solve.'))
    records=[];steps=0
    try:
        for i,ref in enumerate(p['reference_inputs']):
            with np.load(ROOT/ref['path'],allow_pickle=False) as z:q=z['reference_q'].copy();control=z['reference_ctrl'].copy()
            for j,speed in enumerate(p['speeds_m_s']):
                d=mujoco.MjData(m);d.qpos[:]=q;d.qvel[:]=0;d.qvel[0]=speed;d.qvel[va[-2:]]=speed/sim.hw.WHEEL_RADIUS
                d.ctrl[:]=control;d.qacc_warmstart[:]=0;d.time=0
                assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
                mujoco.mj_forward(m,d);initial_q=d.qpos.copy();initial_v=d.qvel.copy()
                initial_contacts=contacts(m,d);initial_qacc=d.qacc.copy()
                pre=[];post=[];predictions=[];qerrors=[];verrors=[];metrics=[];contact_rows=[]
                for k in range(40):
                    before_v=d.qvel.copy();pre.append(np.r_[d.qpos,before_v,d.ctrl]);steps+=1;mujoco.mj_step(m,d)
                    contact_rows.append(contacts(m,d));pre_gyro=d.sensordata[gyro:gyro+3].copy()
                    actual_force=d.actuator_force.copy();assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
                    assert d.time==(k+1)*.0005 or abs(d.time-(k+1)*.0005)<1e-15
                    expected_q=initial_q.copy();mujoco.mj_integratePos(m,expected_q,initial_v,(k+1)*.0005)
                    difference=np.empty(16);mujoco.mj_differentiatePos(m,difference,1.,expected_q,d.qpos)
                    post.append(np.r_[d.qpos,d.qvel]);predictions.append(expected_q);qerrors.append(difference);verrors.append(d.qvel-initial_v)
                    # Update post-step positions for true-chain metrics only;do not recompute forces/sensors.
                    mujoco.mj_kinematics(m,d)
                    a=[];b=[];loops=[]
                    for side in ('L','R'):
                        hip=(d.xanchor[m.joint('alpha'+side).id]+d.xanchor[m.joint('beta'+side).id])/2
                        wheel=d.xpos[m.body('wheel'+side).id];end=d.site_xpos[m.site('couplerB_'+side+'_end').id]
                        a.append(float(np.linalg.norm(wheel-hip)));b.append(float(np.linalg.norm(end-hip)));loops.append(float(np.linalg.norm(wheel-end)))
                    bounds=np.array([sim.hw.torque_limit(float('inf'),float(before_v[n]),idx<4,0.,.0005)[0] for idx,n in enumerate(va)])
                    margin=float(np.minimum(d.qpos[qa]-limits[:,0],limits[:,1]-d.qpos[qa]).min())
                    metrics.append(a+b+loops+[float((1.4-abs(d.qpos[qa[:4]])).min()),margin,
                        float(np.maximum(abs(control)-bounds,0).max()),float(np.maximum(abs(actual_force)-bounds,0).max())]+pre_gyro.tolist()+list(sim.euler(d)))
                metric=np.asarray(metrics);path=OUT/f'height_{i}_speed_{j}.npz'
                np.savez_compressed(path,initial_q=initial_q,initial_v=initial_v,control=control,initial_qacc=initial_qacc,
                    pre=np.asarray(pre),post=np.asarray(post),expected_q=np.asarray(predictions),q_error_tangent=np.asarray(qerrors),
                    v_error=np.asarray(verrors),metrics=metric,metric_columns=np.asarray(COL),speed=np.array(speed),height=np.array(ref['height_m']))
                cp=OUT/f'height_{i}_speed_{j}_contacts.json';atomic_json(cp,dict(initial_contacts=initial_contacts,preintegration_step_contacts=contact_rows))
                record=dict(height_m=ref['height_m'],speed_m_s=speed,steps=40,path=path.name,sha256=sha(path),contacts_path=cp.name,contacts_sha256=sha(cp),
                    initial_qacc_max=float(abs(initial_qacc).max()),q_tangent_error_max=float(abs(np.array(qerrors)).max()),
                    v_error_max=float(abs(np.array(verrors)).max()),minimum_actual_leg_m=float(metric[:,:4].min()),
                    maximum_loop_error_m=float(metric[:,4:6].max()),minimum_active_design_margin_rad=float(metric[:,6].min()),
                    minimum_eight_joint_margin_rad=float(metric[:,7].min()),maximum_command_excess_Nm=float(metric[:,8].max()),
                    maximum_force_excess_Nm=float(metric[:,9].max()),maximum_roll_pitch_yaw_rad=np.max(abs(metric[:,13:16]),axis=0).tolist(),
                    physical_design_limits_passed=bool(metric[:,:4].min()>=.1147044660616607 and metric[:,6:8].min()>=0 and metric[:,8:10].max()<=1e-6))
                records.append(record);atomic_json(OUT/'progress.json',dict(completed_cpu_steps=steps,completed_states=len(records)))
                print('RELATIVE',ref['height_m'],speed,'qerr',record['q_tangent_error_max'],'verr',record['v_error_max'],flush=True)
        assert steps==400 and len(records)==10 and all(sha(ROOT/n)==v for n,v in sources.items())
        atomic_json(OUT/'completion.json',dict(verified=True,cpu_integration_steps=400,records=records,source_contract_sha256=sha(OUT/'source_contract.json'),
            transitionFD_calls=0,new_gpu_rollouts=0,new_training_samples=0,controller_admitted=False,
            limits='Nominalgravity-control rolling-flow diagnostic20ms only,notactualNomclosedloop/uncertaincontact/continuous-domain proof. No fitted relative-error threshold;260 finite decision required.'))
        print('PASS259 registered400CPUsteps/all10states;noPPO/FD',flush=True)
    except BaseException as e:
        atomic_json(OUT/'failure.json',dict(error=repr(e),attempted_cpu_steps=steps,records=records,implicit_retry=False));raise


if __name__=='__main__':run()
