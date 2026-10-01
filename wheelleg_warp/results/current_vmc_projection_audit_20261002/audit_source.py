"""Bounded equilibrium command audit: online interpolation and unilateral radial guard."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
import warp as wp
import wheelleg_sim as sim
import model_lqr as ml
from state_estimation import leg_kinematics
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario,bank_height_115
from native.controller import D
from probe_current_vmc_design import current_vmc_table
from probe_height_115_action_predict_loow import sha
from audit_radial_braking_response import nominal_input_jacobian


def audit_projection(source,output):
    source=source.resolve();output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    meta=json.loads((source/'verification.json').read_text());assert meta['current_vmc_design']
    assert sha(source/'windows.npz')==meta['windows_sha256']
    data=np.load(source/'windows.npz',allow_pickle=False);cfg=np.load(source/'controller_inputs.npz',allow_pickle=False)
    h=cfg['heights'];kgain=cfg['gains'];ff=cfg['feed'];eq=cfg['angles'];ids=cfg['ids'];cols=meta['columns'];rows=[];derived={}
    for world in range(6):
        key=f'w{world}_braking';pre=data[key+'_pre'];rad=data[key+'_radial'];diag=data[key+'_diagnostic'];memory=data[key+'_controller_after_nominal']
        assert np.allclose(np.diff(rad[:,0]),.0005,atol=1e-10)
        q=pre[:,cols['qpos_start']:cols['qvel_start']];v=pre[:,cols['qvel_start']:cols['warmstart_start']]
        length=rad[:,4:6].mean(axis=1);index=np.clip(np.searchsorted(h,length,side='left')-1,0,len(h)-2);ratio=np.clip((length-h[index])/(h[index+1]-h[index]),0,1)
        k=(1-ratio[:,None,None])*kgain[index]+ratio[:,None,None]*kgain[index+1];feed=(1-ratio[:,None])*ff[index]+ratio[:,None]*ff[index+1];theta=(1-ratio)*eq[index]+ratio*eq[index+1]
        delta_h=k[:,1,0]*(diag[:,25:27]-diag[:,23:25]).mean(axis=1);delta_v=delta_h/k[:,1,3];delta_w=k[:,0,3]*delta_v
        pitch=np.arcsin(np.clip(2*(q[:,3]*q[:,5]-q[:,6]*q[:,4]),-1,1));yaw=np.arctan2(2*(q[:,3]*q[:,6]+q[:,4]*q[:,5]),1-2*(q[:,5]**2+q[:,6]**2))
        vx=np.cos(yaw)*v[:,0]+np.sin(yaw)*v[:,1];angles=[];angular_rates=[]
        for pose,velocity in zip(q,v):
            a=[];r=[]
            for side in range(2):
                qa=pose[ids[2*side:2*side+2]];dq=velocity[ids[4+2*side:6+2*side]]
                a.append(sim.fk_joints(*qa)['phi5']+np.pi/2);r.append((leg_kinematics(qa,dq)[3].T@dq)[1])
            angles.append(np.mean(a));angular_rates.append(np.mean(r))
        # Exact inversion of the known controller pitch-rate EMA; first row
        # has no preceding memory and is excluded from the algebra check.
        gyro=(memory[1:,6]-.95*memory[:-1,6])/.05
        position=np.where(memory[:,13]>0,np.cos(yaw)*(q[:,0]-memory[:,14])+np.sin(yaw)*(q[:,1]-memory[:,15]),0.)
        x=np.c_[np.array(angles)[1:]-pitch[1:]-theta[1:],np.array(angular_rates)[1:]-gyro,position[1:],vx[1:]-rad[1:,2],pitch[1:],gyro]
        original_w=feed[1:,0]-np.einsum('ni,ni->n',k[1:,0],x)
        actual_raw_w=diag[1:,19:21].mean(axis=1);error=float(abs(actual_raw_w-original_w-delta_w[1:]).max());assert error<=1e-5,error
        issued=pre[:,cols['ctrl_start']:];clipped=np.max(abs(diag[:,15:21]-issued),axis=1)>1e-6
        stopped=rad[:,2]==0.;clipped_pose=diag[:,28]>0;arrival=meta['events'][world]['arrival_s'];early=stopped&(rad[:,0]-arrival<=.75)
        row=dict(world=world,algebra_error_Nm=error,braking_steps=int(stopped.sum()),projected_steps=int(np.sum(stopped&clipped_pose)),
            projected_fraction=float(np.mean(clipped_pose[stopped])),early_effective_reference_min_m_s=float((delta_v[early]*np.sign(meta['scenarios'][world]['speed'])).min()),
            early_effective_reference_max_m_s=float((delta_v[early]*np.sign(meta['scenarios'][world]['speed'])).max()),
            peak_wheel_projection_delta_Nm=float(abs(delta_w[stopped]).max()),motor_clipped_braking_steps=int(np.sum(clipped&stopped)),
            original_wheel_mean_early_Nm=float(np.mean(original_w[early[1:]])),actual_raw_wheel_mean_early_Nm=float(np.mean(actual_raw_w[early[1:]])),
            stop_distance_m=meta['complete_episodes'][world]['stop_distance_m'],task_pass=meta['complete_episodes'][world]['success'])
        rows.append(row);derived[key]=np.c_[rad[:,0],delta_h,delta_w,delta_v,vx,clipped_pose,clipped]
    np.savez_compressed(output/'derived.npz',**derived)
    result=dict(role='recorded_projection_coordinate_audit',rows=rows,physical6=sum(e['physical_safety_passed'] for e in meta['complete_episodes']),
        limitations='Reconstructs recorded commands and equivalent velocity-reference shifts only. Unprojected commands are counterfactual algebra, not safe executable trajectories; no causal rollout or robustness claim.',
        input_sha256={str((source/name).relative_to(ROOT)):sha(source/name) for name in ('windows.npz','controller_inputs.npz','verification.json')},source_sha256=sha(Path(__file__)))
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PROJECTION AUDIT',rows,flush=True)


def run(output):
    output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    env=NativeEnv(n=18,scenario=[HeightTerrainScenario(stand_height_m=.115)]*18,bank_factory=bank_height_115,
        height_conditioned=True,height_design='range115',residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=True)
    try:
        table,_=current_vmc_table();env.k['gains'].assign(np.stack([t[0] for t in table]));env.k['feed'].assign(np.stack([t[1] for t in table]));env.k['angles'].assign(np.array([t[2] for t in table]))
        ref,a,b,_=ml.design(env.cpu,.115,min_height=.115);basis,_=ml.sagittal_basis(env.cpu);c,g,_=ml.vmc_coordinates(ref);inverse=np.linalg.pinv(c)
        samples=[('base',np.zeros(15))]
        for axis,name in ((6,'length'),(7,'rate')):
            for size in (1e-5,5e-6):
                for sign in (-1,1):samples.append((f'{name}_{size}_{sign}',inverse[:,axis]*size*sign))
        poses=[];velocities=[];expected=[];features=[]
        ids=env.ids.numpy();heights=env.k['heights'].numpy();gains=env.k['gains'].numpy();feeds=env.k['feed'].numpy();angles=env.k['angles'].numpy()
        for name,delta in samples:
            full=basis@delta;q=ref.qpos.copy();mujoco.mj_integratePos(env.cpu,q,full[:env.cpu.nv],1.);v=ref.qvel+full[env.cpu.nv:]
            q=q.astype(np.float32);v=v.astype(np.float32);active_q=q[ids[:2]];active_v=v[ids[4:6]]
            leg,_,_,jac=leg_kinematics(active_q,active_v);length=np.linalg.norm(leg);angle=sim.fk_joints(*active_q)['phi5']+np.pi/2;rate=(jac.T@active_v)[0]
            index=int(np.clip(np.searchsorted(heights,length,side='left')-1,0,len(heights)-2));ratio=np.clip((length-heights[index])/(heights[index+1]-heights[index]),0,1)
            k=(1-ratio)*gains[index]+ratio*gains[index+1];feed=(1-ratio)*feeds[index]+ratio*feeds[index+1];theta=(1-ratio)*angles[index]+ratio*angles[index+1]
            x=np.array([angle-theta,(jac.T@active_v)[1],0.,0.,0.,0.]);virtual=feed[:2]-k@x
            force=feed[2]+sim.hw.DESIGN_MASS/sim.hw.BASELINE_MASS*(500*(.115-length)-25*rate)
            hip=jac@np.array([force,virtual[1]]);expected.append(np.r_[hip,hip,virtual[0],virtual[0]])
            poses.extend([q,q]);velocities.extend([v,v]);features.append((name,float(length),float(rate),float(force)))
        poses=np.array(poses);velocities=np.array(velocities);memory=env.k['state'].numpy();memory[:]=0.;memory[:,0]=3.;memory[:,12]=3.;memory[:,10]=1.;memory[:,13]=1.
        for sample,(_,length,rate,_) in enumerate(features):memory[2*sample:2*sample+2,1]=length;memory[2*sample:2*sample+2,3]=rate
        env.data.qpos.assign(poses);env.data.qvel.assign(velocities);env.data.sensordata.zero_();env.k['state'].assign(memory)
        references=env.k['reference'].numpy();references[::2,8]=0.;env.k['reference'].assign(references)
        wp.launch(env.control_kernel,18,[env.data.qpos,env.data.qvel,env.data.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
            env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],env.data.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
        ctrl=env.data.ctrl.numpy().astype(float);diag=env.diag.numpy();error=float(np.max(abs(ctrl[::2]-np.array(expected))))
        assert error<=1e-5,error;assert np.all(diag[:,28]==0) and np.max(abs(ctrl-diag[:,15:21]))<=1e-5
        rows=[]
        for i,(name,length,rate,force) in enumerate(features):
            difference=ctrl[2*i+1]-ctrl[2*i];jac=leg_kinematics(poses[2*i,ids[:2]],velocities[2*i,ids[4:6]])[3]
            extra=np.linalg.inv(jac)@difference[:2];assert abs(extra[1])<1e-5
            assert abs(extra[0]-diag[2*i+1,32])<1e-5
            rows.append(dict(sample=name,length_m=length,rate_m_s=rate,base_radial_force_N=force,guard_force_N=float(extra[0]),
                guard_requested_N=float(diag[2*i+1,31]),effective_mass_kg=float(diag[2*i+1,33]),motor_delta_peak_Nm=float(abs(difference).max())))
        assert any(r['guard_force_N']>.1 for r in rows)
        fixed_input=nominal_input_jacobian(env.cpu,ref)-g[:,:2]@table[0][0]@c[:6]
        jac=leg_kinematics(ref.qpos[[7,10]],np.zeros(2))[3]
        effective_mass=8+sim.hw.HIP_OUTPUT_INERTIA*np.sum(np.linalg.inv(jac)[0]**2)
        guard_jacobian=-np.outer(g[:,2],effective_mass*(200**2*c[6]+400*c[7]))
        sector=dict(inactive_radius=float(max(abs(np.linalg.eigvals(a+b@fixed_input)))),active_radius=float(max(abs(np.linalg.eigvals(a+b@(fixed_input+guard_jacobian))))))
        assert sector['inactive_radius']<1 and sector['active_radius']<1
        np.savez_compressed(output/'commands.npz',qpos=poses,qvel=velocities,controller_before=memory,references=references,commands=ctrl,diagnostic=diag,c=c,a=a,b=b)
        result=dict(role='local_command_model_consistency_not_physics_or_robust_certificate',no_guard_command_error_max_Nm=error,rows=rows,
            fixed_node_sector_check=sector,
            no_projection_or_clipping_active=True,radial_guard_is_in_current_linear_design=False,
            conclusion='Online scheduled current-J nominal commands match the independent algebra locally; unilateral max(0,...) radial protection adds a state-dependent command absent from the design Jacobian. No unique two-sided Jacobian at activation. This alone does not prove it causes the highspeed failure.',
            original_costs_limits_and_default_controller_unchanged=True,
            limitations='Nine small symmetric sagittal samples, fixed7kg/0.115m, commands only. Both fixed-node affine sectors stable, but this excludes interpolation derivatives, switching, projections and saturation; no full-trajectory or random-domain proof.',
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/probe_current_vmc_design.py',ROOT/'wheelleg_warp/audit_radial_braking_response.py',ROOT/'wheelleg_ppo/tools/model_lqr.py')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('COMMAND AUDIT PASS',error,rows,flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--trajectory-source',type=Path);args=p.parse_args()
    if args.trajectory_source:audit_projection(args.trajectory_source,args.output)
    else:run(args.output)
