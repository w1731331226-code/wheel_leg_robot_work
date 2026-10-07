"""Independent freshcold reconstruction of frozen20; no solver/helper invocation."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import mujoco
from review_yaw_sector import ROOT, sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import wheelleg_sim as sim
import model_lqr as ml
from dashboard.live_env import atomic_json

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/geometry_seed_validation_v1'


def run():
    assert not (OUT/'cold_review.json').exists()
    p,a,c=[json.loads((OUT/n).read_text()) for n in ('proposal.json','source_admission.json','completion.json')]
    assert c['verified'] and c['passed']==20 and c['no_fit']
    assert c['admission_sha256']==sha(OUT/'source_admission.json')
    assert a['proposal_sha256']==sha(OUT/'proposal.json')
    assert a['runner_sha256']==sha(ROOT/'wheelleg_warp/validate_geometry_seed.py')
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    old=json.loads((ROOT/'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_reference_v1/proposal.json').read_text())
    existing={(r['height'],r['speed']) for r in old['new_reference_points']+old['existing_moving_references']}
    assert not existing.intersection({(r['height'],r['speed']) for r in p['points']})
    assert len(c['records'])==20 and c['residual_calls']==sum(r['residual_calls'] for r in c['records'])<=24000
    m,_=sim.load_model(ml.XML,True)
    assert (m.nq,m.nv,m.nu)==(17,16,6) and abs(m.body_mass.sum()-7.)<1e-12
    assert m.opt.timestep==.0005 and m.opt.iterations==100 and int(m.opt.integrator)==3
    assert mujoco.get_mjcb_control() is None and mujoco.get_mjcb_passive() is None
    pairs=(('alphaL','alphaR'),('passA_L','passA_R'),('betaL','betaR'),('passC_L','passC_R'),('wheel1','wheel2'))
    joints=[m.joint(n).id for n in ('alphaL','betaL','alphaR','betaR','passA_L','passC_L','passA_R','passC_R')]
    dofs=[m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
    records=[]
    for i,r in enumerate(c['records']):
        assert all(r[k]==p['points'][i][k] for k in ('height','speed','role','height_cell','speed_cell'))
        assert r['passed'] and r['solver_success'] and r['nfev']<=100 and r['residual_calls']<=1200
        path=OUT/r['path'];assert sha(path)==r['sha256']
        assert sha(ROOT/r['donor_path'])==r['donor_sha256']
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['initial'][[1,6,7,8,9]],z['old_initial'][[1,6,7,8,9]])
            assert abs(sim.fk_joints(z['initial'][2],z['initial'][4])['leg_len']-r['height'])<=1e-12
            x=z['solution'];assert np.isfinite(x).all() and np.all((x>=z['lower'])&(x<=z['upper']))
            d=mujoco.MjData(m)
            d.qpos[2]=x[0];d.qpos[3:7]=[math.cos(x[1]/2),0.,math.sin(x[1]/2),0.]
            for side,wheel in (('L','wheel1'),('R','wheel2')):
                d.qpos[[m.jnt_qposadr[m.joint(n+side).id] for n in ('alpha','passA_','beta','passC_')]]=x[2:6]
                d.qvel[m.jnt_dofadr[m.joint(wheel).id]]=x[9]
            d.qvel[0]=r['speed']
            for value,names in zip(x[6:9],(('motor_wheelL','motor_wheelR'),('motor_alphaL','motor_alphaR'),('motor_betaL','motor_betaR'))):
                for name in names:d.ctrl[m.actuator(name).id]=value
            for value,name in ((d.qpos,'q'),(d.qvel,'v'),(d.ctrl,'ctrl')):np.testing.assert_array_equal(value,z[name])
            assert not d.qacc_warmstart.any() and not d.qfrc_applied.any() and not d.xfrc_applied.any()
            mujoco.mj_forward(m,d);acc=d.qacc.copy()
            limits=np.array([sim.hw.torque_limit(float('inf'),float(d.qvel[n]),j<4,0.,.0005)[0] for j,n in enumerate(dofs)])
            excess=max(float(np.maximum(abs(d.ctrl)-limits,0.).max()),float(np.maximum(abs(d.actuator_force)-limits,0.).max()))
            angles=d.qpos[m.jnt_qposadr[joints]]
            active=float((1.4-abs(angles[:4])).min())
            margin=float(np.minimum(angles-m.jnt_range[joints,0],m.jnt_range[joints,1]-angles).min())
            lengths,loops=[],[]
            for side in ('L','R'):
                hip=(d.xanchor[m.joint('alpha'+side).id]+d.xanchor[m.joint('beta'+side).id])/2
                center=d.xpos[m.body('wheel'+side).id];end=d.site_xpos[m.site('couplerB_'+side+'_end').id]
                lengths.extend([float(np.linalg.norm(center-hip)),float(np.linalg.norm(end-hip))])
                loops.append(float(np.linalg.norm(center-end)))
            floor=m.geom('floor').id;wheels={m.geom('wheel_collide_L').id,m.geom('wheel_collide_R').id}
            support=d.ncon==2 and all(floor in set(ct.geom) and (set(ct.geom)-{floor}).issubset(wheels) for ct in d.contact)
            d.qacc[:]=0.;mujoco.mj_inverse(m,d)
            raw=d.qfrc_inverse-d.qfrc_actuator
            forces=np.r_[[raw[0],raw[2],raw[4]],[sum(raw[m.jnt_dofadr[m.joint(n).id]] for n in pair) for pair in pairs]]
            height=sim.fk_joints(x[2],x[4])['leg_len']-r['height']
            residual=np.r_[forces,100*height,x[1]]
            np.testing.assert_allclose(residual,z['residual'],rtol=0,atol=1e-9)
            passed=bool(abs(acc).max()<=1e-4 and abs(residual).max()<=1e-6 and min(lengths)>=.1147044660616607 and
                        active>=0 and margin>=0 and excess<=1e-6 and support)
            assert passed
            records.append(dict(height=r['height'],speed=r['speed'],role=r['role'],path=r['path'],sha256=r['sha256'],
                passed=passed,cold_combined_residual_max=float(abs(residual).max()),
                cold_projected_force_max=float(abs(forces).max()),cold_full_qacc_max=float(abs(acc).max()),
                cold_minus_cached_qacc_max=float(abs(acc-z['qacc']).max()),height_error_m=float(height),
                actual_leg_min=min(lengths),loop_error_max=max(loops),active_margin=active,eight_joint_margin=margin,
                torque_excess=excess,bilateral_support=bool(support)))
    atomic_json(OUT/'cold_review.json',dict(
        verified=True,round=279,passed=20,records=records,completion_sha256=sha(OUT/'completion.json'),
        reviewer_sha256=sha(__file__),fresh_data_forward_inverse_checks=20,
        no_fit=True,new_optimization_calls=0,integration_steps=0,transitionFD_calls=0,new_training_samples=0,
        map_admitted=False,
        decision='Close geometry-initializer reference validation; accept only these nominal cold reference arrays.280 review decides finite continuous table/source/query,not extra reference solves or equilibrium trajectories.',
        limits='Freshcold reference/geometry only,no continuous-domain/contact/controller stability ornewmethod benefit. Twentydevelopment points neverfit; oldfailedbatch remainsfailed and finalOOD unopened.'))
    print('PASS279 independentfreshcold20/20; unchangedinitializerinputs/source; no solver/steps/FD/PPO',flush=True)


if __name__=='__main__':
    run()
