"""Independent saved-array/coordinate review;zero additional transitionFD."""
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

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/full_mode_model_admission_v1'


def run():
    p=json.loads((OUT/'proposal.json').read_text());done=json.loads((OUT/'completion.json').read_text())
    c=json.loads((OUT/'source_contract.json').read_text())
    assert done['verified'] and done['derivative_calls']==10 and len(done['records'])==10
    assert done['source_contract_sha256']==sha(OUT/'source_contract.json') and c['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/n)==v for n,v in c['source_sha256'].items())
    m,_=sim.load_model(ml.XML,True);assert m.nq==17 and m.nv==16 and m.nu==6
    # Build from joint addresses, without calling collector.coordinates or ml.sagittal_basis.
    t=np.zeros((32,30));u=np.zeros((6,6));pairs=[('alphaL','alphaR'),('passA_L','passA_R'),('betaL','betaR'),('passC_L','passC_R')]
    scale=1/np.sqrt(2.)
    for block,axes,sign in [(0,[0,2,4],1),(15,[1,3,5],-1)]:
        for j,dof in enumerate(axes):t[dof,block+j]=1;t[16+dof,block+7+j]=1
        for j,(left,right) in enumerate(pairs):
            a,b=[int(m.jnt_dofadr[m.joint(n).id]) for n in (left,right)]
            t[a,block+3+j]=scale;t[b,block+3+j]=sign*scale
            t[16+a,block+10+j]=scale;t[16+b,block+10+j]=sign*scale
        for n,s in [('wheel1',1),('wheel2',sign)]:t[16+m.jnt_dofadr[m.joint(n).id],block+14]=s*scale
    for j,(left,right) in enumerate([('motor_wheelL','motor_wheelR'),('motor_alphaL','motor_alphaR'),('motor_betaL','motor_betaR')]):
        a,b=[m.actuator(n).id for n in (left,right)];u[a,j]=u[b,j]=scale;u[a,j+3]=scale;u[b,j+3]=-scale
    phases=np.array([m.jnt_dofadr[m.joint(n).id] for n in ('wheel1','wheel2')]);keep=np.setdiff1d(np.arange(32),phases)
    assert np.linalg.matrix_rank(t)==30 and np.linalg.matrix_rank(u)==6
    np.testing.assert_allclose(t.T@t,np.eye(30),rtol=0,atol=3e-16)
    np.testing.assert_array_equal(t[np.r_[1,3,5,17,19,21],:15],0)
    records=[];arrays=[]
    for i,r in enumerate(done['records']):
        assert (r['height_m'],r['eps'])==(p['heights'][i//2],p['derivative_eps'][i%2])
        path=OUT/r['path'];assert sha(path)==r['sha256']
        with np.load(path,allow_pickle=False) as z:
            a,b=z['A32'],z['B32'];assert a.shape==(32,32) and b.shape==(32,6) and np.isfinite(a).all() and np.isfinite(b).all()
            np.testing.assert_array_equal(t,z['T']);np.testing.assert_array_equal(u,z['U']);np.testing.assert_array_equal(phases,z['gauge_dofs'])
            a30=t.T@a@t;b30=t.T@b@u
            np.testing.assert_array_equal(a30,z['A30']);np.testing.assert_array_equal(b30,z['B30'])
            assert float(z['height'])==r['height_m'] and float(z['eps'])==r['eps'] and float(z['timestep'])==.0005
            d=mujoco.MjData(m);d.qpos[:]=z['reference_q'];d.qvel[:]=z['reference_v'];d.ctrl[:]=z['reference_ctrl'];mujoco.mj_forward(m,d)
            assert max(abs(d.qacc))<=1e-4 and d.ncon==2
            floor=m.geom('floor').id;wheelids={m.geom('wheel_collide_L').id,m.geom('wheel_collide_R').id}
            assert all(floor in set(contact.geom) and (set(contact.geom)-{floor}).issubset(wheelids) for contact in d.contact)
            metrics=dict(omitted_phase_to_nongauge_A_max_abs=float(abs(a[np.ix_(keep,phases)]).max()),
                         common_from_difference_A_max_abs=float(abs(a30[:15,15:]).max()),
                         difference_from_common_A_max_abs=float(abs(a30[15:,:15]).max()))
            for name,value in metrics.items():assert value==r[name]
            np.testing.assert_array_equal(b30[[16,17]],r['roll_yaw_position_B']);np.testing.assert_array_equal(b30[[23,24]],r['roll_yaw_velocity_B'])
            np.testing.assert_allclose(np.linalg.svd(b30[15:],compute_uv=False),r['difference_B_singular_values'],rtol=0,atol=1e-15)
            arrays.append((a.copy(),b.copy()));records.append(dict(path=r['path'],sha256=r['sha256'],height_m=r['height_m'],eps=r['eps'],reference_qacc_max=float(abs(d.qacc).max()),**metrics))
        if i%2:
            assert float(abs(arrays[-2][0]-arrays[-1][0]).max())==r['eps_A32_max_abs_difference']
            assert float(abs(arrays[-2][1]-arrays[-1][1]).max())==r['eps_B32_max_abs_difference']
    atomic_json(OUT/'review.json',dict(verified=True,records=records,state_basis_rank=30,input_basis_rank=6,
        completion_sha256=sha(OUT/'completion.json'),reviewer_sha256=sha(__file__),additional_transitionFD_calls=0,
        forward_only_reference_checks=10,new_gpu_rollouts=0,new_training_samples=0,controller_admitted=False,
        strict_phase_influence_zero=all(r['omitted_phase_to_nongauge_A_max_abs']==0 for r in records),
        decision='Savedarrays/coordinatecoverage admittedonly.Full32retained;30projection not exactclosedplant. No actualNom/contact/gain stabilitycertificate.',
        limits='Independenttransforms/reportedmetrics andstaticreferenceforwardchecks only,not independentderivative recomputation or nonlineartrajectory validation. 258 finite scope/controller-contract decision required.'))
    print('PASS257 independent10array/coordinates/reference review;0extraFD;full32retained',flush=True)


if __name__=='__main__':run()
