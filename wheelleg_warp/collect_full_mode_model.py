"""Bounded nominal full32 transition derivatives;no controller admission."""
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
from review_yaw_sector import ROOT,sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import wheelleg_sim as sim
import model_lqr as ml
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/full_mode_model_admission_v1'


def coordinates(m):
    common,inputs=ml.sagittal_basis(m);difference=common.copy();diff_inputs=inputs.copy()
    for offset in (0,m.nv):
        difference[offset+np.array([1,3,5])]=common[offset+np.array([0,2,4])]
        difference[offset+np.array([0,2,4])]=0
        right=[m.jnt_dofadr[m.joint(n).id] for n in ('alphaR','passA_R','betaR','passC_R','wheel2')]
        difference[offset+np.array(right)] *= -1
    for n in ('motor_wheelR','motor_alphaR','motor_betaR'):diff_inputs[m.actuator(n).id] *= -1
    common=common/np.linalg.norm(common,axis=0);difference=difference/np.linalg.norm(difference,axis=0)
    transform=np.c_[common,difference];input_map=np.c_[inputs,diff_inputs]/np.sqrt(2)
    phases=np.array([m.jnt_dofadr[m.joint(n).id] for n in ('wheel1','wheel2')])
    np.testing.assert_allclose(transform.T@transform,np.eye(30),atol=3e-16,rtol=0)
    np.testing.assert_allclose(input_map.T@input_map,np.eye(6),atol=3e-16,rtol=0)
    assert np.linalg.matrix_rank(transform)==30 and np.linalg.matrix_rank(input_map)==6
    np.testing.assert_array_equal(transform[phases],0)
    gauge=np.eye(32)[:,phases];np.testing.assert_allclose(np.c_[transform,gauge].T@np.c_[transform,gauge],np.eye(32),atol=3e-16,rtol=0)
    return transform,input_map,phases


def run():
    assert not any((OUT/n).exists() for n in ('source_contract.json','completion.json','failure.json'))
    p=json.loads((OUT/'proposal.json').read_text());assert p['maximum_full_transition_derivative_calls']==10
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    m,_=sim.load_model(ml.XML,True)
    assert (m.nq,m.nv,m.nu,m.na,m.nmocap,m.nplugin)==(17,16,6,0,0,0)
    assert m.opt.timestep==.0005 and m.opt.iterations==100 and int(m.opt.integrator)==3
    assert abs(m.body_mass.sum()-7.)<1e-12
    assert mujoco.get_mjcb_control() is None and mujoco.get_mjcb_passive() is None
    transform,input_map,phases=coordinates(m)
    sources={**p['source_sha256'],'wheelleg_warp/collect_full_mode_model.py':sha(__file__),str(Path(ml.XML).relative_to(ROOT)):sha(ml.XML)}
    atomic_json(OUT/'source_contract.json',dict(verified=True,proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,
        mujoco_version=mujoco.__version__,numpy_version=np.__version__,shape=dict(nq=17,nv=16,nu=6),
        model_options=dict(timestep=.0005,iterations=100,integrator=3),derivative_call_budget=10,
        scope='Nominalflatbilateralstationaryplant;notactualNomclosedloop/uncertaincontacts. FD internallyperformsCPUphysics,notGPUrollout.'))
    records=[];calls=0
    try:
        for i,height in enumerate(p['heights']):
            ref=ml.equilibrium(m,height,min_height=.115)
            before=[ref.qpos.copy(),ref.qvel.copy(),ref.ctrl.copy(),ref.qacc_warmstart.copy(),float(ref.time)]
            matrices=[]
            for j,eps in enumerate(p['derivative_eps']):
                scratch=mujoco.MjData(m);mujoco.mj_copyData(scratch,m,ref)
                a=np.empty((32,32));b=np.empty((32,6));calls+=1
                mujoco.mjd_transitionFD(m,scratch,eps,True,a,b,None,None)
                assert np.isfinite(a).all() and np.isfinite(b).all()
                for x,y in zip(before[:-1],[ref.qpos,ref.qvel,ref.ctrl,ref.qacc_warmstart]):np.testing.assert_array_equal(x,y)
                assert ref.time==before[-1]
                a30=transform.T@a@transform;b30=transform.T@b@input_map
                path=OUT/f'node_{i}_eps_{j}.npz'
                np.savez_compressed(path,A32=a,B32=b,A30=a30,B30=b30,T=transform,U=input_map,
                    gauge_dofs=phases,reference_q=ref.qpos,reference_v=ref.qvel,reference_ctrl=ref.ctrl,
                    timestep=np.array(m.opt.timestep),height=np.array(height),eps=np.array(eps))
                keep=np.setdiff1d(np.arange(32),phases)
                record=dict(height_m=height,eps=eps,path=path.name,sha256=sha(path),
                    omitted_phase_to_nongauge_A_max_abs=float(abs(a[np.ix_(keep,phases)]).max()),
                    common_from_difference_A_max_abs=float(abs(a30[:15,15:]).max()),
                    difference_from_common_A_max_abs=float(abs(a30[15:,:15]).max()),
                    roll_yaw_position_B=a30.shape and b30[[16,17]].tolist(),
                    roll_yaw_velocity_B=b30[[23,24]].tolist(),
                    difference_B_singular_values=np.linalg.svd(b30[15:],compute_uv=False).tolist(),
                    phase_removed_unconditionally=False,reference_unchanged=True)
                matrices.append((a,b));records.append(record)
                atomic_json(OUT/'progress.json',dict(status='derivatives',completed_calls=calls,height_m=height))
            records[-1]['eps_A32_max_abs_difference']=float(abs(matrices[0][0]-matrices[1][0]).max())
            records[-1]['eps_B32_max_abs_difference']=float(abs(matrices[0][1]-matrices[1][1]).max())
            print('NODE',height,'calls',calls,'phase influence',record['omitted_phase_to_nongauge_A_max_abs'],flush=True)
        assert calls==10 and len(records)==10 and all(sha(ROOT/n)==v for n,v in sources.items())
        atomic_json(OUT/'completion.json',dict(verified=True,derivative_calls=10,records=records,
            source_contract_sha256=sha(OUT/'source_contract.json'),state_basis_rank=30,input_map_rank=6,
            new_gpu_rollouts=0,new_training_samples=0,controller_admitted=False,
            limits='Full32retained,30projectionforcomparisononly;reportedcoupling/gauge/sensitivity notstability orcontrollabilitycertificate. Frozenflatnodes andnoactualNomfilter/guard/projection/memory. Independentreviewpending.'))
        atomic_json(OUT/'progress.json',dict(status='complete',completed_calls=10));print('PASS25610full32FDcalls;noPPO/controlleradmission',flush=True)
    except BaseException as e:
        atomic_json(OUT/'failure.json',dict(error=repr(e),attempted_derivative_calls=calls,records=records,implicit_retry=False));raise


if __name__=='__main__':run()
