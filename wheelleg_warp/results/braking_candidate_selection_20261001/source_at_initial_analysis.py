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
        reference=ref.qpos.copy();reference[0]=memory[i,14] if memory[i,13]>0 else initial[i,0]
        states=[]
        for pose,vel in zip(q[:,i],v[:,i]):
            delta=np.empty(m.nv);mujoco.mj_differentiatePos(m,delta,1.,reference,pose);states.append(project@np.r_[delta,vel])
        states=np.array(states);du=(cmd[:,i]-ref.ctrl)@input_project.T
        cost[i]=np.einsum('ti,ij,tj->',states,Q,states)+np.einsum('ti,ij,tj->',du,R,du)+states[-1]@P@states[-1]
        f=features(m,q[:,i],v[:,i]);joints=q[:,i,[m.joint(name).qposadr[0] for name in JOINTS]]
        caps=np.array([1.4,1.4,2.5,2.5,1.4,1.4,2.5,2.5])
        if np.any(abs(joints)>caps):reasons[i].append('design_joint_cap')
        if np.any(f[:,:4]<LIMIT):reasons[i].append('actual_A_B_geometry')
        # Terminal joint stopping cone on the existing5ms scale, no margin scan.
        speeds=v[-1,i,[m.joint(name).dofadr[0] for name in JOINTS]]
        if np.any(200*(caps-joints[-1])-speeds<0) or np.any(200*(caps+joints[-1])+speeds<0):reasons[i].append('terminal_joint_cone')
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
    actual_cost=[]
    for i in selected:
        reference=ref.qpos.copy();reference[0]=memory[i,14] if memory[i,13]>0 else initial[i,0]
        states=[]
        for pose,vel in zip(data['qpos'][:,i],data['qvel'][:,i]):
            delta=np.empty(m.nv);mujoco.mj_differentiatePos(m,delta,1.,reference,pose);states.append(project@np.r_[delta,vel])
        states=np.array(states);du=(data['commands'][:,i]-ref.ctrl)@input_project.T
        actual_cost.append(float(np.einsum('ti,ij,tj->',states,Q,states)+np.einsum('ti,ij,tj->',du,R,du)+states[-1]@P@states[-1]))
    np.savez_compressed(output/'selection.npz',indices=selected,extra=data['known_extra'][count+np.array(selected)],candidate_costs=cost.reshape(18,13),valid=valid.reshape(18,13),actual_selected_cost=actual_cost,Q=Q,R=R,P=P)
    result=dict(role='nominal_prediction_only_fixed_LQR_candidate_selection',rows=rows,
        selected_nonzero_count=sum(r['selected_arm']!='zero' for r in rows),original_cost_recovered=True,
        original_limits_unchanged=True,active_design_cap_rad=1.4,terminal_joint_cone_k=200.,actual_future_used_for_selection=False,
        actual_selected_cost_error_max=float(np.max(abs(np.array(actual_cost)-np.array([r['model_cost'] for r in rows])))),
        limitations='Finite13 candidates,18 states,5ms model prediction, known extraL1<=0.1Nm. Cost-to-go from original full15-state physical LQR; moving position target follows existing lock semantics. Original physical gates and1.4rad design cap/cone applied. No continuous optimization, feedback rollout, sensor-only or full-task result.',
        input_sha256={str((source/f).relative_to(ROOT)):sha(source/f) for f in ('verification.json','rollout.npz')},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_ppo/tools/model_lqr.py',ROOT/'wheelleg_warp/probe_braking_motor_response.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
