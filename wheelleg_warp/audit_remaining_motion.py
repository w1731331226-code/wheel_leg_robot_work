"""Timing of all strict-pair residual failures and all21 prefix divergences."""
import json
import numpy as np
import run_parking_withdrawal as runner
from review_nom_yaw_filter import full_flags


def first_over(x,limit):
    hit=np.flatnonzero(np.abs(x)>limit)
    return int(hit[0]) if len(hit) else None


def run():
    assert first_over(np.array([0.,5.,5.1]),5)==2 and first_over(np.array([0.,-5.1]),5)==1
    assert first_over(np.array([0.,5.]),5) is None
    runner.verify();out=runner.OUT;root=runner.ROOT;f=out/'round213_pair_review.json';review=json.loads(f.read_text())
    assert review['verified'] and review['summary']['paired_records']==39
    inputs={str(f.relative_to(root)):runner.sha(f)};remaining=[];divergences=[]
    def data(seed,condition,case,kind):
        result=out/'runs'/str(seed)/condition/'batch_0/result.json'
        assert runner.sha(result)==review['input_sha256'][str(result.relative_to(root))]
        inputs[str(result.relative_to(root))]=runner.sha(result)
        row=next(r for r in json.loads(result.read_text())['runs'] if r['seed']==case)
        file=result.parent/row[kind]['path'];assert runner.sha(file)==row[kind]['sha256']
        inputs[str(file.relative_to(root))]=row[kind]['sha256']
        with np.load(file,allow_pickle=False) as z:return row,{k:z[k] for k in z.files}
    for pair in review['case_pairs']:
        seed,case=pair['seed'],pair['case']
        row,b=data(seed,'parking_request_withdrawal',case,'complete_trace');t=b['trace'];cols={str(k):i for i,k in enumerate(b['columns'])}
        if pair['recorded_prefix_exact'] and not row['success']:
            assert not row['scenario']['relative_attitude'] and row['target_leg_m']==.115
            np.testing.assert_allclose(np.rad2deg(np.max(np.abs(t[:,[cols['roll'],cols['pitch'],cols['yaw']]]),axis=0)),row['peak_deg'],atol=1e-12,rtol=0)
            i=first_over(t[:,cols['yaw']],np.deg2rad(5));assert i is not None
            g=pair['trigger_step']-1
            load=t[:,cols['left_target_normal_N']]+t[:,cols['right_target_normal_N']]
            hit=np.flatnonzero(load>1e-6)
            remaining.append(dict(seed=seed,case=case,scenario=row['scenario'],flags=full_flags(row),first_yaw_failure_step=i+1,
                first_yaw_failure_post_s=float(t[i,cols['post_s']]),parking_trigger_step=g+1,
                failure_before_parking=i<g,current_command=float(t[i,cols['speed_command']]),
                yaw_deg=float(np.rad2deg(t[i,cols['yaw']])),roll_deg=float(np.rad2deg(t[i,cols['roll']])),
                first_original_target_load_pre_s=float(t[hit[0],cols['pre_s']]) if len(hit) else None,
                target_normal_N=float(load[i]),residual_lambda=float(t[i,cols['original_lambda']]),
                wheel_normal_N=t[i,[cols['left_normal_N'],cols['right_normal_N']]].tolist(),
                recorded_wheel_slip_m_s=t[i,[cols['left_slip'],cols['right_slip']]].tolist(),
                minimum_lambda_before_failure=float(t[:i+1,cols['original_lambda']].min()),
                filtered_virtual_residual=t[i,47:53].tolist(),accepted_wheel_residual_Nm=t[i,35:37].tolist(),
                pre_commanded_hip_Nm=b['pre'][i,-6:-2].tolist(),
                wheel_command_Nm=t[i,37:39].tolist(),wheel_command_bounds_Nm=t[i,58:60].tolist(),
                pre_wheel_speed_rad_s=t[i,56:58].tolist(),wheel_command_to_bound_ratio=(np.abs(t[i,37:39])/t[i,58:60]).tolist(),
                physical_pass=row['physical_safety_passed'],design_pass=row['design_joint_passed']))
        elif not pair['recorded_prefix_exact']:
            _,a=data(seed,'original',case,'complete_trace');_,pa=data(seed,'original',case,'parking_trace');_,pb=data(seed,'parking_request_withdrawal',case,'parking_trace')
            g=pair['trigger_step']-1
            def first_diff(x,y):
                n=min(g,len(x),len(y));hit=np.flatnonzero(np.any(x[:n]!=y[:n],axis=1))
                return int(hit[0]) if len(hit) else None
            post=first_diff(a['post'],b['post']);requests=first_diff(pa['trace'][:,5:11],pb['trace'][:,5:11])
            trace=first_diff(a['trace'],t)
            d=dict(seed=seed,case=case,first_post_difference_step=post+1 if post is not None else None,
                   first_trace_difference_step=trace+1 if trace is not None else None,first_request_difference_step=requests+1 if requests is not None else None,
                   parking_trigger_step=g+1,post_difference_precedes_request=(post<requests if post is not None and requests is not None else None))
            if post is not None:
                changes=np.flatnonzero(a['post'][post]!=b['post'][post])
                d['first_changed_post_columns']=changes.tolist()
                d['first_post_max_abs_difference']=float(np.max(np.abs(a['post'][post]-b['post'][post])))
                d['requested_identical_at_first_post_difference']=bool(np.array_equal(pa['trace'][post,5:11],pb['trace'][post,5:11]))
            divergences.append(d)
    assert len(remaining)==12 and len(divergences)==21
    summary=dict(strict_remaining_failures=12,unique_strict_failed_cases=len({r['case'] for r in remaining}),
        first_yaw_failure_before_parking=sum(r['failure_before_parking'] for r in remaining),
        physical_design_pass=sum(r['physical_pass'] and r['design_pass'] for r in remaining),
        first_yaw_failure_lambda_one=sum(r['residual_lambda']==1 for r in remaining),
        first_yaw_failure_lambda_values=[r['residual_lambda'] for r in remaining],
        post_difference_before_request=sum(r['post_difference_precedes_request'] is True for r in divergences),
        requests_equal_at_first_post_difference=sum(r.get('requested_identical_at_first_post_difference',False) for r in divergences))
    runner.write(out/'round214_remaining_motion.json',dict(verified=True,round=214,summary=summary,remaining_strict_cases=remaining,prefix_divergences=divergences,
        input_sha256=inputs,source_sha256=runner.sha(__file__),runner_contract_sha256=runner.sha(out/'runner_contract.json'),new_training_or_evaluations=0,
        limits='All12 remaining failures within predeclared strict18 and all21 excluded pairs. Target-normal snapshots are not full tangential friction or loaded traversal certificates. Lambda1 is not unlimited authority or proof of controllability. First recorded divergence cannot uniquely identify CUDA/solver/root cause;warmstarts unrecorded. No relaxed prefix or new method/gate.',
        next='215 deep review whether active-motion yaw/support/control-authority identification has a distinguishable hypothesis before new learning. Parking-only branch cannot address failures that already cross5deg before its trigger;keep closed benefit/formal5. Full contribution/strongbaselines/freshID-OOD/statistics/manuscript remain open.'))
    print('PASS214 offline motion/prefix evidence',summary,flush=True)


if __name__=='__main__':run()
