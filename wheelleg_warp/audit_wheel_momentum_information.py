"""One80episode offline comparison; estimate inputs restricted to public481."""
import json
import numpy as np
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
import wheel_momentum_information as info
import model_lqr as ml
import wheelleg_sim as sim

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/wheel_momentum_information_v1'


def run():
    assert not any((OUT/n).exists() for n in ('audit_started.json','audit_completion.json','audit_failure.json'))
    p=json.loads((OUT/'proposal.json').read_text());a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['approximate_information_audit_only_admitted']
    assert a['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/n)==h for n,h in {**p['source_sha256'],**a['source_sha256']}.items())
    m,_=sim.load_model(ml.XML,True);nq=m.nq
    wheels=[m.jnt_dofadr[m.joint(n).id] for n in ('wheel1','wheel2')]
    carriers=[[m.jnt_dofadr[m.joint(n+s).id] for n in ('alpha','passA_')] for s in ('L','R')]
    np.testing.assert_array_equal(m.dof_damping[wheels],.005)
    output=OUT/'audit';output.mkdir(exist_ok=False)
    atomic_json(OUT/'audit_started.json',dict(source_admission_sha256=sha(OUT/'source_admission.json'),
        auditor_sha256=sha(__file__),episodes=80,new_simulation=0))
    records=[];total=0
    try:
        for entry in p['input_results']:
            file=ROOT/entry['path'];assert sha(file)==entry['sha256']
            rows=json.loads(file.read_text())['runs'];assert len(rows)==20
            for row in rows:
                actor=file.parent/row['actor_trace']['path']
                dense=file.parent/row['complete_trace']['path']
                assert sha(actor)==row['actor_trace']['sha256'] and sha(dense)==row['complete_trace']['sha256']
                with np.load(actor,allow_pickle=False) as z:trace=z['trace'].copy()
                old,new,U,dt,valid=info.decode(trace);k=np.flatnonzero(valid)
                assert np.array_equal(k,np.arange(1,len(trace))) and row['scenario']['delay_ms']==0.
                # Finish every public estimate BEFORE reading diagnostic private state/torque.
                estimate=info.estimates(old[k],new[k],U[k],dt[k])
                with np.load(dense,allow_pickle=False) as z:
                    pre=z['pre'];post=z['post'];end=k*40;start=end-40
                    assert np.all(end<=len(post)) and np.all(dt[k]==.02)
                    actual=post[:,-2:]
                    relative=pre[:,nq+np.array(wheels)]
                    after=post[end-1][:,nq+np.array(wheels)]+post[end-1,nq+4,None]
                    before=pre[start][:,nq+np.array(wheels)]+pre[start,nq+4,None]
                    for j,ids in enumerate(carriers):
                        after[:,j]+=post[end-1][:,nq+np.array(ids)].sum(axis=1)
                        before[:,j]+=pre[start][:,nq+np.array(ids)].sum(axis=1)
                    actual_impulse=np.array([actual[s:e].sum(axis=0)*.0005 for s,e in zip(start,end)])
                    damping_impulse=np.array([.005*relative[s:e].sum(axis=0)*.0005 for s,e in zip(start,end)])
                    reference=(actual_impulse-info.SPIN_INERTIA*(after-before)-damping_impulse)/dt[k,None]
                    exact_command=np.array([pre[s:e,-6:].sum(axis=0)*.0005 for s,e in zip(start,end)])
                    np.testing.assert_allclose(U[k],exact_command,rtol=1e-6,atol=1e-8)
                errors={name:estimate[name]-reference for name in
                        ('relative_proxy_Nm','carrier_proxy_Nm','corrected_proxy_Nm')}
                coverage=(reference>=estimate['gain_interval_lower_Nm'])&(reference<=estimate['gain_interval_upper_Nm'])
                path=output/f'{entry["arm"]}_{row["seed"]}.npz'
                np.savez_compressed(path,actor_rows=k,sensor_end_steps=end,dt=dt[k],public_old=old[k],
                    public_new=new[k],public_command_integral=U[k],**estimate,
                    diagnostic_spin_axis_balance_Nm=reference,diagnostic_actual_impulse=actual_impulse,
                    diagnostic_true_absolute_omega_before=before,diagnostic_true_absolute_omega_after=after,
                    diagnostic_damping_impulse=damping_impulse,**{n+'_error_Nm':e for n,e in errors.items()},
                    gain_interval_contains_diagnostic=coverage)
                result=dict(arm=entry['arm'],seed=row['seed'],task_success=row['success'],
                    intervals=len(k),excluded_initial_packet=1,terminal_packet_absent=True,
                    path=str(path.relative_to(OUT)),sha256=sha(path),actor_sha256=sha(actor),dense_sha256=sha(dense),
                    rmse_Nm={n:float(np.sqrt(np.mean(e**2))) for n,e in errors.items()},
                    mean_bias_Nm={n:float(e.mean()) for n,e in errors.items()},
                    max_abs_error_Nm={n:float(abs(e).max()) for n,e in errors.items()},
                    gain_interval_coverage=float(coverage.mean()))
                records.append(result);total+=len(k)
            print('AUDITED',entry['arm'],entry['batch'],'episodes',len(records),'intervals',total,flush=True)
        assert len(records)==80
        summary={}
        for arm in p['arms']:
            values=[r for r in records if r['arm']==arm]
            count=sum(r['intervals'] for r in values)
            summary[arm]=dict(episodes=len(values),intervals=count,
                pooled_rmse_Nm={n:float(np.sqrt(sum(r['rmse_Nm'][n]**2*r['intervals'] for r in values)/count))
                                for n in records[0]['rmse_Nm']},
                corrected_rmse_lower_than_relative_cases=sum(r['rmse_Nm']['corrected_proxy_Nm']<r['rmse_Nm']['relative_proxy_Nm'] for r in values),
                gain_only_interval_coverage=sum(r['gain_interval_coverage']*r['intervals'] for r in values)/count)
        atomic_json(OUT/'audit_completion.json',dict(
            verified=True,round=292,records=records,summary=summary,episodes=80,intervals=total,
            source_admission_sha256=sha(OUT/'source_admission.json'),auditor_sha256=sha(__file__),
            estimator_inputs_public_only=True,diagnostic_private_state_used_after_estimates=True,
            new_simulation=0,new_training_samples=0,new_optimizer=0,
            interpretation='Reference is a model spin-axis mechanicalbalance diagnostic fromactualtorque/truecarrier anddamping,notindependentlymeasuredcontactwrench ornormal. Gyro0.5ms/softclosure/trapezoidalω andpublicgaininterval limitations retained; coverage isnotcertification.',
            exact_force_or_normal_admitted=False,controller_admitted=False,
            next='293independent packet/time/algebra/recompute saved80 andfiniteinformationutility decision. No fitthreshold/J/damping/bounds ornewPPO.'))
        print('DONE29280offlineapproximationaudit',summary,flush=True)
    except BaseException as error:
        atomic_json(OUT/'audit_failure.json',dict(error=repr(error),records=records,intervals=total,implicit_retry=False))
        raise


if __name__=='__main__':run()
