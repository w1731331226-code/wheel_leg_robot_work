"""Fresh-data review of one known-case solution; no optimization or stepping."""
import json
import math
import numpy as np
import mujoco
import solve_moving_equilibrium as original
from solve_continuous_nominal_references import constraints
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = original.OUT.parent / 'geometry_seed_diagnostic_v1'


def run():
    assert not (OUT / 'cold_review.json').exists()
    p, a, c = [json.loads((OUT / n).read_text()) for n in
               ('proposal.json','source_admission.json','completion.json')]
    assert c['source_admission_sha256']==sha(OUT / 'source_admission.json')
    assert a['proposal_sha256']==sha(OUT / 'proposal.json')
    assert a['runner_sha256']==sha(ROOT / 'wheelleg_warp/diagnose_geometry_seed.py')
    assert a['original_constraints_helper_sha256']==sha(ROOT / 'wheelleg_warp/solve_continuous_nominal_references.py')
    assert all(sha(ROOT / n)==h for n,h in p['source_sha256'].items())
    assert sha(ROOT / p['old_outcome']['path'])==p['old_outcome']['sha256']
    assert c['passed'] and c['nfev']==10 and c['residual_calls']==112
    path = OUT / 'outcome.npz'; assert sha(path)==c['outcome_sha256']
    m, _ = original.sim.load_model(original.ml.XML,True)
    assert (m.nq,m.nv,m.nu)==(17,16,6) and abs(m.body_mass.sum()-7.)<1e-12
    with np.load(path,allow_pickle=False) as z, np.load(ROOT / p['old_outcome']['path']) as old:
        np.testing.assert_array_equal(z['old_initial'],old['initial'])
        np.testing.assert_array_equal(z['initial'][[1,6,7,8,9]],old['initial'][[1,6,7,8,9]])
        assert abs(original.sim.fk_joints(z['initial'][2],z['initial'][4])['leg_len']-p['height'])<=1e-12
        x=z['solution']; assert np.all((x>=z['lower'])&(x<=z['upper']))
        d=mujoco.MjData(m)
        d.qpos[2]=x[0];d.qpos[3:7]=[math.cos(x[1]/2),0.,math.sin(x[1]/2),0.]
        for side in ('L','R'):
            d.qpos[[m.jnt_qposadr[m.joint(n+side).id] for n in ('alpha','passA_','beta','passC_')]]=x[2:6]
            d.qvel[m.jnt_dofadr[m.joint('wheel1' if side=='L' else 'wheel2').id]]=x[9]
        d.qvel[0]=p['speed']
        for value,names in zip(x[6:9],(('motor_wheelL','motor_wheelR'),('motor_alphaL','motor_alphaR'),('motor_betaL','motor_betaR'))):
            for name in names:d.ctrl[m.actuator(name).id]=value
        np.testing.assert_array_equal(d.qpos,z['q'])
        np.testing.assert_array_equal(d.qvel,z['v'])
        np.testing.assert_array_equal(d.ctrl,z['ctrl'])
        assert not d.qacc_warmstart.any() and not d.qfrc_applied.any() and not d.xfrc_applied.any()
        mujoco.mj_forward(m,d);acc=d.qacc.copy();physical=constraints(m,d)
        d.qacc[:]=0.;mujoco.mj_inverse(m,d)
        raw=d.qfrc_inverse-d.qfrc_actuator
        forces=[raw[0],raw[2],raw[4]]
        for left,right in (('alphaL','alphaR'),('passA_L','passA_R'),('betaL','betaR'),('passC_L','passC_R'),('wheel1','wheel2')):
            forces.append(raw[m.jnt_dofadr[m.joint(left).id]]+raw[m.jnt_dofadr[m.joint(right).id]])
        height=original.sim.fk_joints(x[2],x[4])['leg_len']-p['height']
        residual=np.r_[forces,100*height,x[1]]
        np.testing.assert_allclose(residual,z['residual'],rtol=0,atol=1e-9)
        passed=bool(abs(acc).max()<=1e-4 and abs(residual).max()<=1e-6 and
            physical['active_margin_rad']>=0 and physical['eight_margin_rad']>=0 and
            max(physical['command_excess_Nm'],physical['actual_torque_excess_Nm'])<=1e-6 and
            physical['minimum_actual_leg_m']>=.1147044660616607 and physical['bilateral_only_floor_support'])
        assert passed
        atomic_json(OUT / 'cold_review.json',dict(
            verified=True,passed=passed,round=277,reviewer_sha256=sha(__file__),
            completion_sha256=sha(OUT / 'completion.json'),cold_combined_residual_max=float(abs(residual).max()),
            cold_projected_force_max=float(abs(np.array(forces)).max()),cold_full_qacc_max=float(abs(acc).max()),
            cold_minus_cached_qacc_max=float(abs(acc-z['qacc']).max()),height_error_m=float(height),**physical,
            initial_geometry_only_change_confirmed=True,fresh_data_forward_inverse_checks=1,
            new_optimizer_calls=0,integration_steps=0,transitionFD_calls=0,new_training_samples=0,
            map_admitted=False,old_batch_still_failed=True,
            decision='Close known-case diagnostic: targetgeometry improves convergence under unchanged original constraints. Only new predeclared unobserved references can validate initializer generality.',
            limits='Independent reconstruction and projectedforce; physical helper reused after prior independent review. Not a new heldoutcase, continuousmap/task stability or method contribution.'))
    print('PASS277 freshcold known-point and initializer-only evidence; no solves/steps/FD/PPO',flush=True)


if __name__=='__main__':
    run()
