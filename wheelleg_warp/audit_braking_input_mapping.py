"""Verify current-J VMC chain rule and reanalyse the rejected braking prefix; no new trajectory."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
import wheelleg_sim as sim
from model_lqr import sagittal_basis,vmc_coordinates,linearize
from state_estimation import leg_kinematics
from native.design import mapped_input,nominal_input_jacobian
from probe_height_115_action_predict_loow import sha


def run(source,output):
    assert not output.exists();output.mkdir(parents=True)
    metadata=json.loads((source/'verification.json').read_text());col=metadata['columns']
    lp=np.load(source/'linear_braking_lp.npz',allow_pickle=False)
    model,_=sim.load_model(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'),True)
    reference=mujoco.MjData(model);reference.qpos[:]=lp['reference_q'];reference.ctrl[:]=lp['reference_ctrl'];mujoco.mj_forward(model,reference)
    a,b=linearize(model,reference);basis,inputs=sagittal_basis(model);project=np.linalg.pinv(basis);c,g,_=vmc_coordinates(reference)
    outer=np.zeros((3,15));outer[2]=sim.hw.DESIGN_MASS/sim.hw.BASELINE_MASS*(sim.KP_LEG*c[6]+sim.KD_LEG*c[7])
    old_u=-g@outer;new_u=nominal_input_jacobian(model,reference)
    half=nominal_input_jacobian(model,reference,5e-7);assert np.allclose(new_u,half,rtol=.01,atol=1e-4)
    old_a=a+b@old_u;new_a=a+b@new_u;bv=b@g[:,:2]
    def step(dx,virtual):
        data=mujoco.MjData(model);mujoco.mj_copyData(data,model,reference)
        full=basis@dx;mujoco.mj_integratePos(model,data.qpos,full[:model.nv],1.);data.qvel[:]+=full[model.nv:]
        data.ctrl[:]=inputs@mapped_input(model,reference,dx,virtual);mujoco.mj_step(model,data)
        delta=np.empty(model.nv);mujoco.mj_differentiatePos(model,delta,1.,reference.qpos,data.qpos)
        return project@np.r_[delta,data.qvel-reference.qvel]
    baseline=step(np.zeros(15),np.zeros(2));errors=[]
    rng=np.random.default_rng(115031)
    for _ in range(32):
        dx=rng.normal(size=15)*1e-7;du=rng.normal(size=2)*1e-7
        actual=step(dx,du)-baseline
        predicted=(old_a@dx+bv@du,new_a@dx+bv@du)
        errors.append([np.linalg.norm(p[7:]-actual[7:])/np.linalg.norm(actual[7:]) for p in predicted])
    errors=np.array(errors);assert errors[:,1].max()<.005
    archive=np.load(source/'windows.npz',allow_pickle=False);rad=archive['w4_braking_radial']
    start=archive['w4_braking_pre'][np.flatnonzero(rad[:,2]==0)[0]]
    replay=np.load(source/'nonlinear_lp_replay.npz',allow_pickle=False)
    q=np.r_[start[None,1:col['qvel_start']],replay['qpos']];v=np.r_[start[None,col['qvel_start']:col['warmstart_start']],replay['qvel']]
    refq=reference.qpos.copy();refq[0]=q[0,0];actual=[]
    for pose,vel in zip(q,v):
        delta=np.empty(model.nv);mujoco.mj_differentiatePos(model,delta,1.,refq,pose);actual.append(project@np.r_[delta,vel])
    actual=np.array(actual);physical_feed=np.linalg.pinv(inputs)@reference.ctrl
    issued=replay['ctrl'];motor_input=np.c_[issued[:,4:6].mean(axis=1),issued[:,[0,2]].mean(axis=1),issued[:,[1,3]].mean(axis=1)]
    predicted={label:[actual[0].copy()] for label in ('old_virtual','current_J_equilibrium','executed_motor')}
    one_step=[];raw=[];first_clip=None
    for k in range(len(issued)):
        du=lp['inputs'][k//10]
        predicted['old_virtual'].append(old_a@predicted['old_virtual'][-1]+bv@du)
        predicted['current_J_equilibrium'].append(new_a@predicted['current_J_equilibrium'][-1]+bv@du)
        predicted['executed_motor'].append(a@predicted['executed_motor'][-1]+b@(motor_input[k]-physical_feed))
        one_step.append(a@actual[k]+b@(motor_input[k]-physical_feed))
        motor=mapped_input(model,reference,actual[k],du);raw.append(motor)
        if first_clip is None and np.max(abs(motor-motor_input[k]))>1e-5:first_clip=k
    raw=np.array(raw);one_step=np.array(one_step);summary={}
    for label,sequence in predicted.items():
        values=np.array(sequence);error=values[1:]-actual[1:]
        summary[label]=dict(max_vx_error_m_s=float(abs(error[:,7]).max()),max_pitch_error_deg=float(np.rad2deg(abs(error[:,2]).max())),
            max_active_joint_error_rad=float(abs(error[:,[3,5]]).max()),max_linear_length_error_m=float(abs(error@c[6]).max()))
    result=dict(role='current_J_nominal_chain_rule_and_executed_input_audit',
        missing_physical_input_jacobian_max_Nm_per_state=float(abs(new_u-old_u).max()),
        small_perturbation_samples=32,small_perturbation_velocity_relative_error_max=errors.max(axis=0).tolist(),
        chain_rule_gate_pass=True,first_mapping_or_clip_difference_s=(first_clip+1)*.0005 if first_clip is not None else None,
        planned_to_executed_motor_difference_max_Nm=float(abs(raw-motor_input).max()),recursive_prediction=summary,
        executed_input_one_step_max_vx_error_m_s=float(abs(one_step[:,7]-actual[1:,7]).max()),
        executed_input_one_step_max_active_joint_velocity_error_rad_s=float(abs(one_step[:,[10,12]]-actual[1:,[10,12]]).max()),
        limitations='Reuses the rejected74-step CPU prefix. First-order current-J correction is verified only near equilibrium. Executed motor history is known at each step; measured-state one-step check is not recursive safety. Symmetric projection omits differences; this audit does not certify larger inputs or an online controller.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/audit_radial_braking_response.py',ROOT/'wheelleg_ppo/tools/model_lqr.py')})
    np.savez_compressed(output/'prediction.npz',actual=actual,one_step=one_step,raw_motor=raw,motor_history=motor_input,small_errors=errors,
                        **{k:np.array(v) for k,v in predicted.items()})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
