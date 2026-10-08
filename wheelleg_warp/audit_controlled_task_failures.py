"""Offline first attitude-limit witnesses for the same8failed tasks; no new runs."""
import json
import numpy as np
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_task_pair_v1'


def run():
    assert not (OUT/'round287_failure_mechanism.json').exists()
    c=json.loads((OUT/'completion.json').read_text())
    review=json.loads((OUT/'task_review.json').read_text())
    assert c['verified'] and c['evaluations']==328 and not review['engineering_continue_gate_passed']
    assert review['completion_sha256']==sha(OUT/'completion.json')
    witnesses=[];failed={}
    for entry in c['records']:
        if entry['panel']!='controlled':continue
        path=OUT/entry['path'];assert sha(path)==entry['sha256']
        rows=json.loads(path.read_text())['runs']
        for row in rows:
            if row['success']:continue
            assert not row['scenario']['relative_attitude']
            assert row['physical_safety_passed'] and row['design_joint_passed']
            failed.setdefault(entry['arm'],[]).append(row['seed'])
            file=path.parent/row['complete_trace']['path'];assert sha(file)==row['complete_trace']['sha256']
            with np.load(file,allow_pickle=False) as z:
                t=z['trace'];col={str(n):i for i,n in enumerate(z['columns'])}
                assert t.shape[0]==row['physical_steps'] and np.isfinite(t).all()
                att=t[:,[col[n] for n in ('roll','pitch','yaw')]]
                crossings=np.flatnonzero(np.any(abs(att)>np.deg2rad(5.),axis=1));assert len(crossings)
                i=int(crossings[0]);h=float(t[i,col['height_command']])
                mean=float(np.mean(t[i,[col['left_reference'],col['right_reference']]]))
                requested=float(t[i,col['roll_offset_requested']]);selected=float(t[i,col['roll_offset_selected']])
                lower=float(row['geometric_limit_m']);upper=.38
                room=max(0.,min(mean-lower,upper-mean))
                expected=float(np.clip(requested,-room,room))
                np.testing.assert_allclose(selected,expected,rtol=0,atol=1e-12)
                np.testing.assert_allclose(mean,h,rtol=0,atol=1e-12)
                desired_mean_interval=[lower+abs(requested),upper-abs(requested)]
                height_interval=[h-row['height_tolerance_m'],h+row['height_tolerance_m']]
                full_requested_feasible=max(desired_mean_interval[0],height_interval[0])<=min(desired_mean_interval[1],height_interval[1])
                speeds=t[i,[col['pre_wheel_speed_left'],col['pre_wheel_speed_right']]]
                normal=t[i,[col['left_normal_N'],col['right_normal_N']]]
                commands=t[i,[col['command_left'],col['command_right']]]
                bounds=t[i,[col['pre_command_bound_left'],col['pre_command_bound_right']]]
                ratio=abs(commands)/bounds
                zero=np.flatnonzero(t[:,col['speed_command']]==0.)
                moving_cross=bool(t[i,col['speed_command']]!=0.)
                witnesses.append(dict(arm=entry['arm'],seed=row['seed'],height=h,
                    step=i+1,pre_s=float(t[i,col['pre_s']]),post_s=float(t[i,col['post_s']]),
                    first_exceeded_axes=[n for n,v in zip(('roll','pitch','yaw'),att[i]) if abs(v)>np.deg2rad(5.)],
                    attitude_deg=np.rad2deg(att[i]).tolist(),during_moving_command=moving_cross,
                    public_command=float(t[i,col['speed_command']]),
                    requested_offset_m=requested,selected_offset_m=selected,available_symmetric_room_m=room,
                    desired_mean_interval_for_full_requested_offset=desired_mean_interval,
                    allowed_height_tracking_interval=height_interval,
                    full_requested_offset_feasible_with_height_tolerance=bool(full_requested_feasible),
                    pre_wheel_speed_rad_s=speeds.tolist(),pre_normal_N=normal.tolist(),
                    wheel_command_Nm=commands.tolist(),speed_dependent_command_bounds_Nm=bounds.tolist(),
                    command_to_bound_ratio=ratio.tolist(),any_exact_zero_normal=bool(np.any(normal==0.)),
                    any_wheel_near_command_bound=bool(np.any(ratio>=1.-1e-6)),
                    physical_design_passed=True,complete_trace_path=str(file.relative_to(ROOT)),
                    complete_trace_sha256=sha(file)))
    assert len(witnesses)==16 and len(failed['old_B0'])==len(failed['map_B0'])==8
    assert failed['old_B0']==failed['map_B0']
    prior=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reference_role_probe_v1/round183_pair_review.json'
    assert prior.exists()
    atomic_json(OUT/'round287_failure_mechanism.json',dict(
        verified=True,round=287,auditor_sha256=sha(__file__),task_review_sha256=sha(OUT/'task_review.json'),
        witnesses=witnesses,same_failed8=failed['old_B0'],new_simulation=0,new_training_samples=0,
        old_consistent_pair_review_sha256=sha(prior),
        finding='Currentfixedmean symmetriclegoffset room is effectivelyzero atupperheight andtinyatlowerheight. Sixyawfirst failures include unloadedwheel andspeed-bound;twoforwardupper cases firstcrossroll withbothwheels loaded. Same categories persistwithmap.',
        limits='Firstrecorded witnesses are observations,not unique causal attribution orfull friction/controllability certificate. Meaninterval is length-coordinate algebra,not a terrain-height/bodyroll exact model. Requested0.035 offset may itself be infeasible within0.02height tolerance; do notclaim simplychangingmean solves it.',
        stopping='Do not revive closedfixed consistent_pair/fixed damping/governor/continuousmap orrefitgains fromthese points. Needcoupled roll-yaw/load/reference hypothesis andpast-counterexample audit before anynewmethod orPPO.',
        next='288 compare these witnesses with existing consistent_pair/yaw-support counterexamples and derive measurable coupled mechanism;baseline numericdifferences stayunresolved. No new simulation admitted.'))
    print('PASS28716firstlimit witnesses,samefailed8,offsetroom algebra;0simulation/learning',flush=True)


if __name__=='__main__':
    run()
