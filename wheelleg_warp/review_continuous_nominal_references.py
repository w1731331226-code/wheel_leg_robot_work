"""Independent fresh-data review of terminal11, including the failed validation."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import mujoco
from review_yaw_sector import ROOT, sha
sys.path.insert(0, str(ROOT / 'wheelleg_ppo/tools'))
import model_lqr as ml
import wheelleg_sim as sim
from dashboard.live_env import atomic_json

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_reference_v1'


def run():
    assert not (OUT / 'cold_review.json').exists()
    p, a, f, terminal = [json.loads((OUT / name).read_text()) for name in
                         ('proposal.json','source_admission.json','failure.json','batch_terminal.json')]
    assert terminal['failure_sha256']==sha(OUT / 'failure.json')
    assert a['proposal_sha256']==sha(OUT / 'proposal.json')
    assert a['runner_sha256']==sha(ROOT / 'wheelleg_warp/solve_continuous_nominal_references.py')
    assert all(sha(ROOT / n)==h for n,h in p['source_sha256'].items())
    assert len(f['records'])==11 and terminal['unattempted_states']==p['new_reference_points'][11:]
    assert not (OUT / 'completion.json').exists()
    m, _ = sim.load_model(ml.XML, True)
    assert (m.nq,m.nv,m.nu)==(17,16,6) and abs(m.body_mass.sum()-7.)<1e-12
    assert m.opt.timestep==.0005 and m.opt.iterations==100 and int(m.opt.integrator)==3
    common = np.zeros((16,8)); common[[0,2,4],[0,1,2]]=1.
    for j,(left,right) in enumerate((('alphaL','alphaR'),('passA_L','passA_R'),
                                    ('betaL','betaR'),('passC_L','passC_R'),('wheel1','wheel2'))):
        common[[m.jnt_dofadr[m.joint(left).id],m.jnt_dofadr[m.joint(right).id]],j+3]=1.
    inputs = np.zeros((6,3))
    for j,names in enumerate((('motor_wheelL','motor_wheelR'),('motor_alphaL','motor_alphaR'),
                              ('motor_betaL','motor_betaR'))):
        for name in names: inputs[m.actuator(name).id,j]=1.
    left = [m.jnt_qposadr[m.joint(n).id] for n in ('alphaL','passA_L','betaL','passC_L')]
    right = [m.jnt_qposadr[m.joint(n).id] for n in ('alphaR','passA_R','betaR','passC_R')]
    joints = [m.joint(n).id for n in ('alphaL','betaL','alphaR','betaR','passA_L','passC_L','passA_R','passC_R')]
    dofs = [m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
    records = []
    for i,r in enumerate(f['records']):
        point = p['new_reference_points'][i]
        assert all(r[k]==point[k] for k in ('height','speed','role'))
        path = OUT / r['path']; assert sha(path)==r['sha256']
        with np.load(path,allow_pickle=False) as z:
            x = z['solution']
            assert np.isfinite(x).all() and np.all((x>=z['lower'])&(x<=z['upper']))
            d = mujoco.MjData(m)
            d.qpos[2]=x[0]; d.qpos[3:7]=[math.cos(x[1]/2),0.,math.sin(x[1]/2),0.]
            d.qpos[left]=d.qpos[right]=x[2:6]
            d.qvel[0]=r['speed']
            d.qvel[[m.jnt_dofadr[m.joint(n).id] for n in ('wheel1','wheel2')]]=x[9]
            d.ctrl[:]=inputs@x[6:9]
            np.testing.assert_array_equal(d.qpos,z['q'])
            np.testing.assert_array_equal(d.qvel,z['v'])
            np.testing.assert_array_equal(d.ctrl,z['ctrl'])
            assert not d.qacc_warmstart.any() and not d.qfrc_applied.any() and not d.xfrc_applied.any()
            mujoco.mj_forward(m,d); acc=d.qacc.copy()
            bounds=np.array([sim.hw.torque_limit(float('inf'),float(d.qvel[n]),j<4,0.,.0005)[0] for j,n in enumerate(dofs)])
            excess=max(float(np.maximum(abs(d.ctrl)-bounds,0.).max()),
                       float(np.maximum(abs(d.actuator_force)-bounds,0.).max()))
            qj=d.qpos[m.jnt_qposadr[joints]]
            margin=float(np.minimum(qj-m.jnt_range[joints,0],m.jnt_range[joints,1]-qj).min())
            active=float((1.4-abs(qj[:4])).min())
            lengths,loops=[],[]
            for side in ('L','R'):
                hip=(d.xanchor[m.joint('alpha'+side).id]+d.xanchor[m.joint('beta'+side).id])/2
                center=d.xpos[m.body('wheel'+side).id]
                end=d.site_xpos[m.site('couplerB_'+side+'_end').id]
                lengths.extend([float(np.linalg.norm(center-hip)),float(np.linalg.norm(end-hip))])
                loops.append(float(np.linalg.norm(center-end)))
            floor=m.geom('floor').id
            wheels={m.geom('wheel_collide_L').id,m.geom('wheel_collide_R').id}
            support=d.ncon==2 and all(floor in set(c.geom) and (set(c.geom)-{floor}).issubset(wheels) for c in d.contact)
            d.qacc[:]=0.; mujoco.mj_inverse(m,d)
            forces=common.T@(d.qfrc_inverse-d.qfrc_actuator)
            height_error=sim.fk_joints(x[2],x[4])['leg_len']-r['height']
            residual=np.r_[forces,100*height_error,x[1]]
            np.testing.assert_allclose(residual,z['residual'],rtol=0,atol=1e-9)
            cold_pass=bool(abs(acc).max()<=1e-4 and abs(residual).max()<=1e-6 and
                min(lengths)>=.1147044660616607 and active>=0 and margin>=0 and excess<=1e-6 and support)
            assert cold_pass==r['passed']
            records.append(dict(**point,path=r['path'],sha256=r['sha256'],cold_pass=cold_pass,
                cold_qacc_max=float(abs(acc).max()),cold_projected_force_max=float(abs(forces).max()),
                cold_combined_residual_max=float(abs(residual).max()),height_error_m=float(height_error),
                pitch_rad=float(x[1]),actual_leg_min=min(lengths),loop_max=max(loops),
                active_margin=active,eight_joint_margin=margin,torque_excess=excess,
                cold_minus_cached_qacc_max=float(abs(acc-z['qacc']).max()),
                initial_fk_height_m=float(sim.fk_joints(z['initial'][2],z['initial'][4])['leg_len'])))
    assert sum(r['cold_pass'] for r in records)==10 and not records[-1]['cold_pass']
    atomic_json(OUT / 'cold_review.json',dict(
        verified=True,round=275,reviewer_sha256=sha(__file__),failure_sha256=sha(OUT / 'failure.json'),
        records=records,fresh_data_checks=11,passed=10,failed=1,unattempted=9,
        new_optimization_calls=0,integration_steps=0,transitionFD_calls=0,new_training_samples=0,
        map_admitted=False,reserved50query_admitted=False,
        decision='Close failed nearest-reference continuous-map qualification. Preserve boundary anchors only as nominal data. Failure is confirmed at freshcold point, not a global feasibility proof.',
        failure_interpretation='Reported force_residual_max included100*heighterror andpitch. Separate actual8force residual, height error and pitch here. max_nfev100/status0 gives no convergence or infeasibility certificate. Initial targetheight mismatch is a concrete seed-quality issue; do not retry this sealed batch.',
        limits='No table fit, interpolation, controller or task qualification. Initializer-only development diagnostic may be separately registered with unchanged objective/bounds/budget and failed case explicitly non-holdout. Not a novel paper contribution.'))
    print('PASS275 cold10anchors; failedvalidation confirmed; closedmap/no retry/physics/FD/PPO',flush=True)


if __name__=='__main__':
    run()
