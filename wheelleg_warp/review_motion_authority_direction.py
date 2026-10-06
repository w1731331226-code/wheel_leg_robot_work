"""Round215 strong-reference comparison and public motor-envelope audit."""
import json
import numpy as np
import run_parking_withdrawal as runner
from audit_remaining_motion import first_over
import hardware_profile as hw


def wheel_bound(speed,gain):
    available,_=hw.torque_limit(1e6,speed,False,0.,.0005)
    return available/max(1.,gain)


def run():
    assert wheel_bound(0,1)==4.5 and wheel_bound(0,1.05)==4.5/1.05
    assert wheel_bound(hw.MOTOR_NO_LOAD_RPM*2*np.pi/60,1)<1e-12
    p=runner.verify();out=runner.OUT;root=runner.ROOT;f=out/'round214_remaining_motion.json';motion=json.loads(f.read_text())
    assert motion['verified'] and runner.sha(root/'wheelleg_warp/audit_remaining_motion.py')==motion['source_sha256']
    assert all(runner.sha(root/n)==s for n,s in motion['input_sha256'].items())
    inputs={str(f.relative_to(root)):runner.sha(f)};records=[];cases={6301000,6301001,6301002,6301003,6301005,6301007}
    original=json.loads((runner.main.OUT/'proposal.json').read_text());rated=hw.MOTOR_RATED_RPM*2*np.pi/60
    for ref in original['references']['phase_result_refs']:
        if ref['panel']!='controlled':continue
        file=root/ref['path'];assert runner.sha(file)==ref['sha256'];inputs[ref['path']]=ref['sha256']
        for row in json.loads(file.read_text())['runs']:
            if row['seed'] not in cases:continue
            trace=file.parent/row['complete_trace']['path'];assert runner.sha(trace)==row['complete_trace']['sha256'];inputs[str(trace.relative_to(root))]=runner.sha(trace)
            with np.load(trace,allow_pickle=False) as z:t=z['trace'];cols={str(k):i for i,k in enumerate(z['columns'])}
            moving=t[:,cols['speed_command']]!=0
            omega=t[:,56:58];bounds=t[:,58:60];gains=np.array(row['actuator_gain_upper'])[-2:]
            expected=np.array([[wheel_bound(w,g) for w,g in zip(ws,gains)] for ws in omega])
            np.testing.assert_allclose(bounds,expected,atol=1e-12,rtol=1e-12)
            yaw=first_over(t[:,cols['yaw']],np.deg2rad(5));maximum=int(np.argmax(np.max(abs(omega)*moving[:,None],axis=1)))
            def event(i):
                fast=int(np.argmax(np.abs(omega[i])))
                return dict(step=i+1,post_s=float(t[i,cols['post_s']]),command=float(t[i,cols['speed_command']]),
                    yaw_deg=float(np.rad2deg(t[i,cols['yaw']])),pre_wheel_speed_rad_s=omega[i].tolist(),
                    command_bounds_Nm=bounds[i].tolist(),wheel_command_Nm=t[i,37:39].tolist(),
                    wheel_normal_N=t[i,[cols['left_normal_N'],cols['right_normal_N']]].tolist(),
                    fast_side=fast,fast_side_sampled_normal_zero=bool(t[i,cols['left_normal_N'] if fast==0 else cols['right_normal_N']]==0))
            records.append(dict(law=ref['label'],case=row['seed'],scenario=row['scenario'],success=row['success'],physical=row['physical_safety_passed'],design=row['design_joint_passed'],
                first_yaw_failure=event(yaw) if yaw is not None else None,max_moving_wheel_speed=event(maximum),
                above_rated_moving_samples=int(np.any(abs(omega)>rated,axis=1)[moving].sum()),
                total_moving_samples=int(moving.sum()),motor_envelope_matches_independent_CPU_profile=True))
    assert len(records)==12
    learned=[]
    for r in motion['remaining_strict_cases']:
        f=out/'runs'/str(r['seed'])/'parking_request_withdrawal/batch_0/result.json';row=next(x for x in json.loads(f.read_text())['runs'] if x['seed']==r['case'])
        gains=np.array(row['actuator_gain_upper'])[-2:];expected=[wheel_bound(w,g) for w,g in zip(r['pre_wheel_speed_rad_s'],gains)]
        np.testing.assert_allclose(expected,r['wheel_command_bounds_Nm'],atol=1e-12,rtol=1e-12)
        learned.append(dict(seed=r['seed'],case=r['case'],wheel_command_to_bound_ratio=r['wheel_command_to_bound_ratio'],independent_envelope_passed=True))
    failing=[r for r in records if not r['success']];passing=[r for r in records if r['success']]
    summary=dict(classical_trajectories=12,classical_success=len(passing),classical_failures=len(failing),
        classical_failed_yaw_events_with_fast_unloaded_wheel=sum(r['first_yaw_failure'] is not None and r['first_yaw_failure']['fast_side_sampled_normal_zero'] for r in failing),
        successful_classical_trajectories_with_above_rated_wheel_speed=sum(r['above_rated_moving_samples']>0 for r in passing),independent_envelope_trajectories_checked=12,learned_event_bounds_checked=12)
    runner.write(out/'round215_direction_review.json',dict(verified=True,round=215,previous_turn='progress',summary=summary,strong_references=records,learned_envelope_checks=learned,
        input_sha256=inputs,source_sha256=runner.sha(__file__),hardware_profile_sha256=runner.sha(root/'wheelleg_ppo/tools/hardware_profile.py'),native_controller_sha256=runner.sha(root/'wheelleg_warp/native/controller.py'),
        source_admission_sha256=runner.sha(out/'source_admission.json'),runner_contract_sha256=runner.sha(out/'runner_contract.json'),pair_review_sha256=runner.sha(out/'round213_pair_review.json'),
        decision='Continue public-proprioception support/speed/torque-envelope feasibility identification;do not retrain or revive pointzero/wheelbox/parking benefit. Simple above-rated rule is not automatically a safe or novel controller.',
        next='216 qualify a timing-aligned wheel-speed/body-motion discrepancy and motor-headroom proxy from existing raw39 sensor fields,including successful negative controls and declared delay.217 derive distinguishable speed-yaw coordination and same-information strong analytical comparison before any new physics budget.218 source/contract only if justified;219 bounded qualification only after admission;220 deepreview/cleanup. Full contribution/formal5/freshID-OOD/statistics/manuscript remains required.',
        limits='Strong reference12 trajectories on6 seen cases; not paired bitwise to learned39 and no causal intervention. Sampled zero normal not geometric flight. Above-rated is motor derating,not slip or support detector. Scalar v=Romega is invalid without rolling/contact/leg-motion assumptions. Full controller includes lateral force/internal dynamics; no single-wheel impossibility theorem.',
        cleanup=json.loads((out/'round215_cleanup.json').read_text()),new_training_or_evaluations=0,candidate_benefit_reopened=False,formal5_admitted=False,entire_goal_complete=False))
    print('PASS215 independent motor curve and strong-reference direction audit',summary,flush=True)


if __name__=='__main__':run()
