"""Fixed original15-state LQR cost and constraints; select ONLY from nominal predicted traces."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.spatial.transform import Rotation
import wheelleg_sim as sim
import model_lqr as ml
from native.terrain import model,HeightTerrainScenario,HEIGHT_115_GEOMETRIC_MIN as LIMIT
from probe_braking_motor_response import features
from probe_height_115_passive import JOINTS
from probe_height_115_action_predict_loow import sha


def lqr_cost(m):
    ref,a,b,gain=ml.design(m,.115,min_height=.115)
    scales=np.array([.03,.005,.03,.1,.1,.1,.1,.3,.1,.3,2.,2.,2.,2.,20.])
    Q=np.diag(scales**-2);mass=sum(m.body_mass)/sim.hw.BASELINE_MASS
    R=np.diag((mass*np.array([.2,.5,.5]))**-2);P=solve_discrete_are(a,b,Q,R)
    assert np.all(np.linalg.eigvalsh(P)>0) and np.allclose(P,P.T,atol=1e-8)
    recovered=np.linalg.solve(R+b.T@P@b,b.T@P@a)
    assert np.allclose(recovered,gain,atol=1e-8,rtol=1e-7)
    return ref,Q,R,P


def trajectory_cost(m,q,v,commands,initial_q,initial_v,memory,ref,Q,R,P,project,input_project):
    reference=ref.qpos.copy();reference[0]=memory[14] if memory[13]>0 else initial_q[0]
    states=[]
    for pose,vel in zip(np.r_[initial_q[None],q],np.r_[initial_v[None],v]):
        delta=np.empty(m.nv);mujoco.mj_differentiatePos(m,delta,1.,reference,pose);states.append(project@np.r_[delta,vel])
    states=np.array(states);du=(commands-ref.ctrl)@input_project.T
    assert len(states)==len(commands)+1
    return float(np.einsum('ti,ij,tj->',states[:-1],Q,states[:-1])+np.einsum('ti,ij,tj->',du,R,du)+states[-1]@P@states[-1])


def constraint_rejections(m,q,v):
    f=features(m,q,v);joints=q[:,[m.joint(name).qposadr[0] for name in JOINTS]]
    caps=np.array([1.4,1.4,2.5,2.5,1.4,1.4,2.5,2.5]);reasons=[]
    if np.any(abs(joints)>caps):reasons.append('design_joint_cap')
    if np.any(f[:,:4]<LIMIT):reasons.append('actual_A_B_geometry')
    speeds=v[-1,[m.joint(name).dofadr[0] for name in JOINTS]]
    if np.any(200*(caps-joints[-1])-speeds<0) or np.any(200*(caps+joints[-1])+speeds<0):reasons.append('terminal_joint_cone')
    if not np.isfinite(np.c_[q,v]).all():reasons.append('nonfinite_state')
    return reasons


def batch_scores(m,q,v,commands,force,initial_q,initial_v,memory,ref,Q,R,P,project,input_project):
    """Same discrete objective and gates for all candidate traces in one batch."""
    steps,count=q.shape[:2];poses=np.concatenate([initial_q[None],q]);vels=np.concatenate([initial_v[None],v])
    reference=np.tile(ref.qpos,(count,1));reference[:,0]=np.where(memory[:,13]>0,memory[:,14],initial_q[:,0])
    tangent=np.zeros((steps+1,count,m.nv));tangent[:,:,:3]=poses[:,:,:3]-reference[None,:,:3]
    rotations=Rotation.from_quat(poses[:,:,[4,5,6,3]].reshape(-1,4))
    base=Rotation.from_quat(reference[:,[4,5,6,3]])
    tangent[:,:,3:6]=(Rotation.from_quat(np.tile(base.inv().as_quat(),(steps+1,1)))*rotations).as_rotvec().reshape(steps+1,count,3)
    tangent[:,:,6:]=poses[:,:,7:]-reference[None,:,7:]
    states=np.concatenate([tangent,vels],axis=2)@project.T;du=(commands-ref.ctrl)@input_project.T
    costs=np.einsum('tni,ij,tnj->n',states[:-1],Q,states[:-1])+np.einsum('tni,ij,tnj->n',du,R,du)+np.einsum('ni,ij,nj->n',states[-1],P,states[-1])
    f=features(m,q.reshape(-1,m.nq),v.reshape(-1,m.nv)).reshape(steps,count,-1)
    joint=f[:,:,4:12];caps=np.array([1.4,1.4,2.5,2.5,1.4,1.4,2.5,2.5]);dofs=[m.joint(n).dofadr[0] for n in JOINTS]
    valid=np.all(abs(joint)<=caps,axis=(0,2))&np.all(f[:,:,:4]>=LIMIT,axis=(0,2))
    valid &= np.all(200*(caps-joint[-1])-v[-1,:,dofs].T>=0,axis=1)&np.all(200*(caps+joint[-1])+v[-1,:,dofs].T>=0,axis=1)
    quat=q[:,:,3:7];pitch=np.arcsin(np.clip(2*(quat[:,:,0]*quat[:,:,2]-quat[:,:,3]*quat[:,:,1]),-1,1))
    roll=np.arctan2(2*(quat[:,:,0]*quat[:,:,1]+quat[:,:,2]*quat[:,:,3]),1-2*(quat[:,:,1]**2+quat[:,:,2]**2))
    yaw=np.arctan2(2*(quat[:,:,0]*quat[:,:,3]+quat[:,:,1]*quat[:,:,2]),1-2*(quat[:,:,2]**2+quat[:,:,3]**2))
    valid &= np.all(np.maximum.reduce([abs(roll),abs(pitch),abs(yaw)])<=np.deg2rad(5),axis=0)
    motor=[m.joint(n).dofadr[0] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
    rpm=abs(vels[:-1][:,:,motor])*30/np.pi;rated=np.array([sim.hw.HIP_RATED_RPM]*4+[sim.hw.MOTOR_RATED_RPM]*2)
    no_load=np.array([sim.hw.HIP_NO_LOAD_RPM]*4+[sim.hw.MOTOR_NO_LOAD_RPM]*2)
    bound=np.array([40.]*4+[4.5]*2)*np.clip((no_load-rpm)/(no_load-rated),0,1)
    valid &= np.all(abs(force)<=bound+1e-6,axis=(0,2))&np.all(abs(commands)<=bound/np.array([1.,1.,1.,1.,1.05,1.05])+1e-6,axis=(0,2))
    valid &= np.isfinite(costs)&np.isfinite(np.concatenate([q,v,commands,force],axis=2)).all(axis=(0,2))
    return costs,valid


def run(source,output):
    source=source.resolve();output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    meta=json.loads((source/'verification.json').read_text());data=np.load(source/'rollout.npz',allow_pickle=False)
    assert meta['all_pairs_pass'] and meta['all_physical_steps_pass'] and len(meta['arms'])==13
    m=model(HeightTerrainScenario(stand_height_m=.115));ref,Q,R,P=lqr_cost(m)
    basis,inputs=ml.sagittal_basis(m);project=np.linalg.pinv(basis);input_project=np.linalg.pinv(inputs)
    count=len(meta['samples']);assert count==234
    q=data['qpos'][:,count:];v=data['qvel'][:,count:];cmd=data['commands'][:,count:]
    initial=data['initial_qpos'][count:];memory=data['initial_controller'][count:]
    cost=np.zeros(count);valid=np.ones(count,bool);reasons=[[] for _ in range(count)]
    for i in range(count):
        cost[i]=trajectory_cost(m,q[:,i],v[:,i],cmd[:,i],initial[i],data['initial_qvel'][count+i],memory[i],ref,Q,R,P,project,input_project)
        reasons[i]=constraint_rejections(m,q[:,i],v[:,i])
        if not np.isfinite(cost[i]):reasons[i].append('nonfinite_cost')
        valid[i]=not reasons[i]
    # Already independently recorded physical trace verifies posture/motor boxes.
    for row in meta['physical']:
        if row['world']>=count and row['safe_arms']!=1:
            i=row['world']-count;valid[i]=False;reasons[i].append('original_physical_gate')
    selected=[];rows=[]
    for group in range(18):
        sl=slice(group*13,(group+1)*13);local=np.flatnonzero(valid[sl]);assert len(local),('no valid candidate',group)
        choice=int(local[np.argmin(cost[sl][local])]);index=group*13+choice;selected.append(index)
        rows.append(dict(world=meta['samples'][index]['world'],elapsed_s=meta['samples'][index]['elapsed_s'],
            selected_arm=meta['arms'][choice],valid_arms=int(valid[sl].sum()),baseline_valid=bool(valid[group*13]),
            model_cost=float(cost[index]),model_zero_cost=float(cost[group*13]),model_cost_reduction=float(cost[group*13]-cost[index]),
            candidate_rejections={meta['arms'][j]:reasons[group*13+j] for j in range(13) if reasons[group*13+j]}))
    # Actual future states are read ONLY AFTER all model decisions are frozen.
    actual_cost=[];actual_zero=[]
    for group,i in enumerate(selected):
        actual_cost.append(trajectory_cost(m,data['qpos'][:,i],data['qvel'][:,i],data['commands'][:,i],initial[i],data['initial_qvel'][i],memory[i],ref,Q,R,P,project,input_project))
        z=group*13;actual_zero.append(trajectory_cost(m,data['qpos'][:,z],data['qvel'][:,z],data['commands'][:,z],initial[z],data['initial_qvel'][z],memory[z],ref,Q,R,P,project,input_project))
    rejected=ROOT/'wheelleg_warp/results/radial_braking_response_20261001/nonlinear_lp_replay.npz'
    bad=np.load(rejected,allow_pickle=False);rejection=constraint_rejections(m,bad['qpos'],bad['qvel'])
    assert 'actual_A_B_geometry' in rejection and 'design_joint_cap' in rejection
    np.savez_compressed(output/'selection.npz',indices=selected,extra=data['known_extra'][count+np.array(selected)],candidate_costs=cost.reshape(18,13),valid=valid.reshape(18,13),actual_selected_cost=actual_cost,actual_zero_cost=actual_zero,Q=Q,R=R,P=P)
    result=dict(role='nominal_prediction_only_fixed_LQR_candidate_selection',rows=rows,
        selected_nonzero_count=sum(r['selected_arm']!='zero' for r in rows),original_cost_recovered=True,
        original_limits_unchanged=True,active_design_cap_rad=1.4,terminal_joint_cone_k=200.,actual_future_used_for_selection=False,
        actual_selected_cost_error_max=float(np.max(abs(np.array(actual_cost)-np.array([r['model_cost'] for r in rows])))),
        actual_selected_cost_not_worse_count=int(np.sum(np.array(actual_cost)<=actual_zero)),
        actual_cost_reduction_min=float(np.min(np.array(actual_zero)-actual_cost)),archived_failed_plan_rejected=rejection,
        limitations='Finite13 candidates,18 states,5ms model prediction, known extraL1<=0.1Nm. Cost-to-go from original full15-state physical LQR; moving position target follows existing lock semantics. Original physical gates and1.4rad design cap/cone applied. No continuous optimization, feedback rollout, sensor-only or full-task result.',
        input_sha256={str((source/f).relative_to(ROOT)):sha(source/f) for f in ('verification.json','rollout.npz')},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_ppo/tools/model_lqr.py',ROOT/'wheelleg_warp/probe_braking_motor_response.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
