"""One known-case initializer change; no old batch restart or gate changes."""
import json
import math
import time
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
import solve_moving_equilibrium as original
from solve_continuous_nominal_references import constraints
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = original.OUT.parent / 'geometry_seed_diagnostic_v1'
sim = original.sim


def geometry(m, height):
    # Reuse the closed target geometry from model_lqr.equilibrium, without its optimizer.
    alpha, beta = sim.ik(height)
    bpos = sim.L1*np.array([math.cos(sim.PHI1_STAND-alpha),math.sin(sim.PHI1_STAND-alpha)])
    dpos = np.array([sim.L5,0.])+sim.L4*np.array([math.cos(sim.PHI4_STAND-beta),math.sin(sim.PHI4_STAND-beta)])
    cpos = np.array([sim.L5/2,-height])
    avec = m.body_pos[m.body('wheelL').id][[0,2]]
    cvec = m.site_pos[m.site('couplerB_L_end').id][[0,2]]
    pa = math.atan2(avec[1],avec[0])-math.atan2(*(cpos-bpos)[::-1])-alpha
    pc = math.atan2(cvec[1],cvec[0])-math.atan2(*(cpos-dpos)[::-1])-beta
    return np.array([alpha,pa,beta,pc])


def rotate(point, angle):
    c,s = math.cos(angle),math.sin(angle)
    return np.array([c*point[0]+s*point[2],point[1],-s*point[0]+c*point[2]])


def unit(m):
    joints=[m.joint(n).id for n in ('alphaL','passA_L','betaL','passC_L')]
    records=[]
    for height in np.linspace(.115,.38,19):
        q=geometry(m,float(height))
        assert np.all((q>m.jnt_range[joints,0])&(q<m.jnt_range[joints,1]))
        assert max(abs(q[[0,2]]))<=1.4
        a=m.body_pos[m.body('legL').id]+rotate(m.body_pos[m.body('kneeA_L').id],q[0])+rotate(m.body_pos[m.body('wheelL').id],q[0]+q[1])
        b=m.body_pos[m.body('legL_D').id]+rotate(m.body_pos[m.body('kneeB_L').id],q[2])+rotate(m.site_pos[m.site('couplerB_L_end').id],q[2]+q[3])
        middle=(m.body_pos[m.body('legL').id]+m.body_pos[m.body('legL_D').id])/2
        assert abs(sim.fk_joints(q[0],q[2])['leg_len']-height)<=1e-12
        assert abs(np.linalg.norm(a-middle)-height)<=1e-12 and np.linalg.norm(a-b)<=1e-12
        records.append(dict(height=float(height),joint_values=q.tolist(),loop_error=float(np.linalg.norm(a-b))))
    return records


def run():
    assert not any((OUT/n).exists() for n in ('source_admission.json','started.json','completion.json','failure.json'))
    started=time.monotonic()
    p=json.loads((OUT/'proposal.json').read_text())
    assert p['optimizer_points']==1 and p['max_nfev']==100 and p['total_residual_calls_maximum']==1200
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    assert sha(ROOT/p['old_outcome']['path'])==p['old_outcome']['sha256']
    assert sha(ROOT/p['initial_reference']['path'])==p['initial_reference']['sha256']
    m,_=sim.load_model(original.ml.XML,True)
    assert (m.nq,m.nv,m.nu)==(17,16,6) and abs(m.body_mass.sum()-7.)<1e-12
    assert m.opt.timestep==.0005 and m.opt.iterations==100 and int(m.opt.integrator)==3
    assert mujoco.get_mjcb_control() is None and mujoco.get_mjcb_passive() is None
    checks=unit(m)
    with np.load(ROOT/p['initial_reference']['path']) as z:
        reference=dict(q=z['q'].copy(),ctrl=z['ctrl'].copy())
    with np.load(ROOT/p['old_outcome']['path']) as z:
        previous_initial=z['initial'].copy()
    counter=dict(used=0,limit=1200)
    old,lo,hi,f,data=original.problem(m,reference,p['height'],p['speed'],counter)
    np.testing.assert_array_equal(old,previous_initial)
    new=old.copy()
    old_height=sim.fk_joints(old[2],old[4])['leg_len']
    new[0]+=p['height']-old_height
    new[2:6]=geometry(m,p['height'])
    np.testing.assert_array_equal(new[[1,6,7,8,9]],old[[1,6,7,8,9]])
    assert np.all((new>lo)&(new<hi))
    assert abs(sim.fk_joints(new[2],new[4])['leg_len']-p['height'])<=1e-12
    atomic_json(OUT/'source_admission.json',dict(
        verified=True,proposal_sha256=sha(OUT/'proposal.json'),runner_sha256=sha(__file__),
        original_constraints_helper_sha256=sha(ROOT/'wheelleg_warp/solve_continuous_nominal_references.py'),
        geometry_checks=checks,old_initial=old.tolist(),new_initial=new.tolist(),
        unchanged_indices=[1,6,7,8,9],total_residual_limit=1200,solver_max_nfev=100,
        optimization_calls=0,integration_steps=0,transitionFD_calls=0))
    atomic_json(OUT/'started.json',dict(source_admission_sha256=sha(OUT/'source_admission.json')))
    try:
        initial_residual=f(new)
        result=least_squares(f,new,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100,x_scale='jac')
        error=f(result.x);mujoco.mj_forward(m,data)
        values=constraints(m,data)
        passed=bool(result.success and abs(error).max()<=1e-6 and abs(data.qacc).max()<=1e-4 and
            values['active_margin_rad']>=0 and values['eight_margin_rad']>=0 and
            max(values['command_excess_Nm'],values['actual_torque_excess_Nm'])<=1e-6 and
            values['minimum_actual_leg_m']>=.1147044660616607 and values['bilateral_only_floor_support'])
        path=OUT/'outcome.npz'
        np.savez_compressed(path,old_initial=old,initial=new,initial_residual=initial_residual,
            solution=result.x,lower=lo,upper=hi,residual=error,q=data.qpos.copy(),v=data.qvel.copy(),
            ctrl=data.ctrl.copy(),qacc=data.qacc.copy(),actuator_force=data.actuator_force.copy(),
            qfrc_inverse=data.qfrc_inverse.copy(),qfrc_actuator=data.qfrc_actuator.copy(),
            qfrc_passive=data.qfrc_passive.copy(),qacc_warmstart=data.qacc_warmstart.copy())
        assert counter['used']<=1200
        atomic_json(OUT/'completion.json',dict(
            verified=True,passed=passed,solver_success=bool(result.success),solver_status=int(result.status),
            nfev=int(result.nfev),residual_calls=counter['used'],source_admission_sha256=sha(OUT/'source_admission.json'),
            outcome_sha256=sha(path),combined_residual_max=float(abs(error).max()),
            projected_force_max=float(abs(error[:8]).max()),height_error_m=float(error[-2]/100),
            pitch_rad=float(error[-1]),forward_qacc_max=float(abs(data.qacc).max()),**values,
            total_seconds=time.monotonic()-started,optimization_states=1,integration_steps=0,
            transitionFD_calls=0,new_training_samples=0,old_batch_still_failed=True,map_admitted=False,
            limits='Known developmentcase, initializer only. Cached final forward, freshcold pending277. No oldbatch restart, freshholdout/interpolation/fulltask/PPO or novel contribution.'))
        print('DONE276 initializer-only point passed=',passed,'nfev=',result.nfev,'calls=',counter['used'],
              'combined residual=',float(abs(error).max()),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),residual_calls=counter['used'],implicit_retry=False))
        raise


if __name__=='__main__':
    run()
