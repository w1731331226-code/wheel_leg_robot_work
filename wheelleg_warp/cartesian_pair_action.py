"""Offline candidate: opposite endpoint force commands with bounded virtual-H sum."""
import itertools
import json
from pathlib import Path
import numpy as np
from state_estimation import leg_kinematics
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

F0=.1*7*9.81/2
H0=1.
W0=.3
H_SUM=.1
BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'
OUT=BASE/'cartesian_pair_action_candidate_v1'


def request(legs,action):
    r=np.asarray(legs,float);a=np.asarray(action,float)
    if r.shape!=(2,2) or a.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(a).all() or np.any(abs(a)>1):
        raise ValueError('Finite paired sagittal geometry and bounded latent3 required')
    length=np.linalg.norm(r,axis=1)
    if not np.isfinite(length).all() or np.any(length<=1e-6):raise ValueError('Degenerate leg geometry')
    f=F0*a[:2]
    radial=r@f/length
    moment=r[:,0]*f[1]-r[:,1]*f[0]
    u=np.array([radial[0]/F0,-radial[1]/F0,moment[0]/H0,-moment[1]/H0,a[2],-a[2]])
    alpha=1/max(1.,float(abs(u).max()),abs(H0*(u[2]+u[3]))/H_SUM)
    return alpha*u,alpha


def decode(legs,u):
    r=np.asarray(legs,float);length=np.linalg.norm(r,axis=1)
    tangent=np.column_stack((-r[:,1],r[:,0]))
    return r/length[:,None]*(F0*np.asarray(u[:2]))[:,None]+tangent/(length**2)[:,None]*(H0*np.asarray(u[2:4]))[:,None]


def constraints(legs):
    # Row scales make force and moment residuals dimensionless; no mixed-unit rank tolerance.
    eye=np.eye(6)
    force=np.column_stack([decode(legs,u).sum(axis=0)/F0 for u in eye])
    return np.vstack((force,[0,0,1,1,0,0],[0,0,0,0,1,1]))


def self_check():
    corners=np.array(list(itertools.product([-1.,0.,1.],repeat=3)))
    for legs in (np.array([[0.,-.3],[0.,-.3]]),np.array([[.04,-.115],[-.03,-.38]])):
        for a in corners:
            u,alpha=request(legs,a);f=decode(legs,u)
            np.testing.assert_allclose(f[0],alpha*F0*a[:2],rtol=0,atol=1e-12)
            np.testing.assert_allclose(f.sum(axis=0),0,rtol=0,atol=1e-12)
            assert abs(u).max()<=1+1e-12 and abs(H0*(u[2]+u[3]))<=H_SUM+1e-12 and u[4]+u[5]==0
    for legs,a in [(np.zeros((2,2)),np.zeros(3)),(np.ones((2,2)),np.ones(3)*1.01),(np.ones((2,2))*np.nan,np.zeros(3))]:
        try:request(legs,a)
        except ValueError:pass
        else:raise AssertionError('Invalid geometry/action accepted')
    same=np.array([[0.,-.3],[0.,-.3]]);c=constraints(same);assert np.linalg.matrix_rank(c,tol=1e-10)==3
    direction=np.array([1.,-1.,0.,0.,0.,0.]);p=np.eye(6)-np.linalg.pinv(c,rcond=1e-12)@c
    np.testing.assert_allclose(p@direction,direction,atol=1e-12,rtol=0)
    jumps=[]
    for epsilon in (1e-3,1e-4,1e-5):
        unequal=same.copy();unequal[1,0]=epsilon;c=constraints(unequal)
        assert np.linalg.matrix_rank(c,tol=1e-10)==4
        projection=np.eye(6)-np.linalg.pinv(c,rcond=1e-12)@c
        jumps.append(dict(epsilon_m=epsilon,strict_projection_change=float(np.linalg.norm(projection@direction-direction)),
            bounded_map_change=float(np.linalg.norm(request(unequal,[0,1,0])[0]-request(same,[0,1,0])[0]))))
    assert all(x['strict_projection_change']>1.4 for x in jumps)
    assert jumps[-1]['bounded_map_change']<jumps[0]['bounded_map_change']
    return jumps


def audit():
    jumps=self_check();p=json.loads((OUT/'proposal.json').read_text());assert not (OUT/'review.json').exists()
    assert p['parameters']==dict(force_scale_N=F0,hip_scale_Nm=H0,wheel_scale_Nm=W0,hip_sum_limit_Nm=H_SUM)
    for mapping in (p['source_sha256'],p['evidence_sha256']):assert all(sha(ROOT/f)==h for f,h in mapping.items())
    data=BASE/'reflection_retention_falsification_v1';review=json.loads((data/'review.json').read_text());prior=json.loads((data/'proposal.json').read_text())
    jobs=[x for x in prior['conditions'] if x['operator']=='original'];assert len(jobs)==8 and review['verified']
    corners=np.array(list(itertools.product([-1.,0.,1.],repeat=3)));records=[];hashes={};worst=0.;power_error=0.;old_leak=0.;hip_peak=0.;alphas=[];ranks=[]
    for job in jobs:
        file=data/job['label']/'result.json';assert sha(file)==review['raw_sha256'][str(file.relative_to(ROOT))]
        row=json.loads(file.read_text())['runs'][0];assert row['success']
        file=file.parent/row['actor_trace']['path'];assert sha(file)==review['raw_sha256'][str(file.relative_to(ROOT))]
        hashes[str(file.relative_to(ROOT))]=sha(file)
        with np.load(file) as z:actor=z['trace']
        indices=np.linspace(0,len(actor)-1,64,dtype=int);assert len(set(indices))==64
        for index in indices:
            frame=actor[index,351:390];parts=[leg_kinematics(frame[12+2*s:14+2*s].astype(float),frame[16+2*s:18+2*s].astype(float)) for s in range(2)]
            legs=np.array([x[0] for x in parts]);rank=int(np.linalg.matrix_rank(constraints(legs),tol=1e-10));ranks.append(rank)
            state_leak=0.;state_min=1.
            for a in corners:
                u,alpha=request(legs,a);f=decode(legs,u);error=float(abs(f.sum(axis=0)).max());worst=max(worst,error)
                assert error<=1e-10 and abs(u).max()<=1+1e-12 and abs(H0*(u[2]+u[3]))<=H_SUM+1e-12 and u[4]+u[5]==0
                alphas.append(alpha);state_min=min(state_min,alpha);hip_peak=max(hip_peak,abs(H0*(u[2]+u[3])))
                original=np.array([a[0],-a[0],a[1],-a[1],a[2],-a[2]])
                state_leak=max(state_leak,float(np.linalg.norm(decode(legs,original).sum(axis=0))))
                for side,(_,velocity,_,jac) in enumerate(parts):
                    torque=jac@np.array([u[side]*F0,u[2+side]*H0]);dq=frame[16+2*side:18+2*side]
                    power_error=max(power_error,abs(float(torque@dq)-float(f[side]@velocity)))
            old_leak=max(old_leak,state_leak)
            records.append(dict(condition=job['label'],actor_index=int(index),legs_xz_m=legs.tolist(),strict_constraint_rank=rank,
                old_D3_max_resultant_N=state_leak,candidate_min_scale=state_min))
    assert len(records)==512 and len(alphas)==13824 and power_error<=1e-10
    atomic_json(OUT/'review.json',dict(round=361,verified=True,geometry_rows=len(records),offline_map_evaluations=len(alphas),
        maximum_resultant_component_error_N=worst,maximum_virtual_work_error_W=power_error,maximum_virtual_H_sum_Nm=hip_peak,
        original_D3_max_resultant_norm_N=old_leak,minimum_candidate_scale=min(alphas),
        strict_rank_counts={str(k):ranks.count(k) for k in sorted(set(ranks))},strict_projection_boundary=jumps,records=records,
        proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sha(__file__),input_sha256=hashes,
        new_physics_steps=0,new_model_forward_rows=0,new_learning_samples=0,runtime_admitted=False,formal5_admitted=False,
        limits='Only ideal commanded virtual endpoint force identities. Not actual support/body wrench, closed-loop decoupling, task improvement or policy equivariance. Seen success poses are not a generalization evaluation. New latent directions/scale alter exploration and authority usage; equal Box bounds do not mean equal distributions.',
        next='Before any runtime branch, decide whether the observed command-coordinate difference warrants a fully matched delivery/mechanism protocol. Do not infer benefit from algebra or deploy through old six independent slew states.'))
    print('PASS361',len(records),'geometries;13824 maps;resultant',worst,'oldD3 max',old_leak,'Hsum',hip_peak,'minscale',min(alphas),'ranks',sorted(set(ranks)),flush=True)


if __name__=='__main__':audit()
