"""Four predeclared first-case saved-data source units, not full80 audit."""
import json
import sys
from pathlib import Path
import numpy as np
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
import wheel_momentum_information as info
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import model_lqr as ml
import wheelleg_sim as sim

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/wheel_momentum_information_v1'


def run():
    assert not (OUT/'source_admission.json').exists()
    p=json.loads((OUT/'proposal.json').read_text());assert p['episodes']==80
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    info.unit()
    m,_=sim.load_model(ml.XML,True)
    qids=[m.jnt_qposadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')]
    vids=[m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')]
    wheels=[m.jnt_dofadr[m.joint(n).id] for n in ('wheel1','wheel2')]
    parent=[[m.jnt_dofadr[m.joint(n+s).id] for n in ('alpha','passA_')] for s in ('L','R')]
    assert np.allclose(m.body_inertia[[m.body('wheel'+s).id for s in ('L','R')],1],info.SPIN_INERTIA,rtol=0,atol=1e-18)
    np.testing.assert_array_equal(m.dof_damping[wheels],.005)
    reports=[]
    for entry in p['input_results']:
        file=ROOT/entry['path'];assert sha(file)==entry['sha256']
        row=json.loads(file.read_text())['runs'][0];assert row['scenario']['delay_ms']==0.
        actor=file.parent/row['actor_trace']['path'];dense=file.parent/row['complete_trace']['path']
        assert sha(actor)==row['actor_trace']['sha256'] and sha(dense)==row['complete_trace']['sha256']
        with np.load(actor,allow_pickle=False) as a,np.load(dense,allow_pickle=False) as z:
            old,new,U,dt,valid=info.decode(a['trace']);k=np.flatnonzero(valid);end=k*40
            post=z['post'];pre=z['pre'];nq=m.nq
            assert np.all(end<=len(post))
            np.testing.assert_array_equal(new[k,12:16],post[end-1][:,qids])
            np.testing.assert_array_equal(new[k,16:20],post[end-1][:,nq+np.array(vids)])
            np.testing.assert_array_equal(new[k,20:22],post[end-1][:,nq+np.array(wheels)])
            ctrl=pre[:,-6:].astype(float)
            exact=np.array([ctrl[e-40:e].sum(axis=0)*.0005 for e in end])
            quantization=float(abs(U[k]-exact).max())
            np.testing.assert_allclose(U[k],exact,rtol=1e-6,atol=1e-8)
            carrier=info.carrier_rate(new[k])
            true=np.stack([post[end-1][:,nq+np.array(ids)].sum(axis=1) for ids in parent],axis=1)
            gyro_pre_error=float(abs(new[k,4]-pre[end-1,nq+4]).max())
            gyro_post_error=float(abs(new[k,4]-post[end-1,nq+4]).max())
            reports.append(dict(arm=entry['arm'],batch=entry['batch'],seed=row['seed'],
                source_result_sha256=entry['sha256'],actor_sha256=sha(actor),dense_sha256=sha(dense),
                complete_intervals=len(k),missing_final_terminal_packet=True,
                command_impulse_quantization_max=quantization,carrier_vs_true_max_rad_s=float(abs(carrier-true).max()),
                carrier_vs_true_rmse_rad_s=float(np.sqrt(np.mean((carrier-true)**2))),
                reversed_sign_rmse_rad_s=float(np.sqrt(np.mean((-carrier-true)**2))),
                gyro_matches_preintegration_max_rad_s=gyro_pre_error,gyro_vs_post_max_rad_s=gyro_post_error))
    atomic_json(OUT/'source_admission.json',dict(
        verified=True,round=291,proposal_sha256=sha(OUT/'proposal.json'),
        source_sha256={str(Path(f).relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'wheelleg_warp/wheel_momentum_information.py']},
        reports=reports,public_packet_wheel_and_active_state_post_exact=True,
        delivered_command_integral_same40step_interval=True,
        exact_synchronous_absolute_momentum_source_qualified=False,
        gyro_time_offset_s=-.0005,carrier_model_exact=False,normal_force_or_guarantee_admitted=False,
        approximate_information_audit_only_admitted=True,full_offline_episodes=80,
        new_simulation=0,new_training_samples=0,model_compile_only=True,
        limits='Fourfirst-case structuralchecks only. Gyropre versuspost wheel/joint mismatch andidealclosure carriererror explicit; no exactphysicalforce certificate orpositive normal bound. Finalterminal packet absent fromActorlogs,no private-state estimator fill-in.292mustreportapproximation/modelbias rather than silentlytightenbounds.'))
    print('PASS291 packet/command units; gyro0.5ms lag/carrier approximation retained; approximateauditonly',flush=True)


if __name__=='__main__':run()
