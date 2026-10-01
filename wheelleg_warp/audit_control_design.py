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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
