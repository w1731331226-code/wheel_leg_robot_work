"""Offline request-to-motor audit;does not rescue the failed fixed candidates."""
import json
import numpy as np
import run_coordination_qualification as runner


def wheel_delivery(raw_nom,bounds,total):
    base=np.clip(np.clip(raw_nom,-4.5,4.5),-bounds,bounds)
    base=base.astype(np.float32).astype(float)  # Original Nom explicitly quantizes BEFORE correction.
    command=np.clip(base+total,-bounds,bounds).astype(np.float32).astype(float)
    return command,base


def run():
    command,base=wheel_delivery(np.array([[.7,.4]]),np.array([[.4,.4]]),np.array([[-.1,.5]]))
    np.testing.assert_allclose(command,[[.3,.4]],atol=2e-8,rtol=0);np.testing.assert_allclose(base,[[.4,.4]],atol=2e-8,rtol=0)
    runner.verify();out=runner.OUT;root=runner.ROOT;f=out/'qualification_review.json';review=json.loads(f.read_text())
    assert review['verified'] and not any(v['passed'] for v in review['gates'].values())
    inputs={str(f.relative_to(root)):runner.sha(f)};completion=json.loads((out/'completion.json').read_text());records=[]
    for job in completion['records']:
        if job['condition']=='B0':continue
        result=out/job['path'];assert runner.sha(result)==job['sha256'];inputs[str(result.relative_to(root))]=job['sha256']
        for row in json.loads(result.read_text())['runs']:
            values={}
            for kind in ('complete_trace','coordination_trace'):
                z=result.parent/row[kind]['path'];assert runner.sha(z)==row[kind]['sha256'];inputs[str(z.relative_to(root))]=row[kind]['sha256']
                with np.load(z,allow_pickle=False) as a:values[kind]=a['trace']
            t=values['complete_trace'];c=values['coordination_trace'];assert len(t)==len(c)==row['physical_steps']
            np.testing.assert_array_equal(t[:,:2],c[:,:2])
            target=np.clip(c[:,6:12]+c[:,12:18]+c[:,18:24],-1,1);total=c[:,24:30]
            expected,base=wheel_delivery(t[:,31:33],t[:,58:60],total[:,4:6])
            np.testing.assert_array_equal(expected,t[:,37:39])
            # The inherited header still points to old shared-reference storage.
            np.testing.assert_array_equal(t[:,54:56],c[:,10:12])
            actual_change=t[:,33:35]-base
            np.testing.assert_array_equal(t[:,33:35],np.clip(base+total[:,4:6],-t[:,58:60],t[:,58:60]))
            moving=c[:,2]!=0;valid=np.flatnonzero(moving);update=c[:,5]==1
            norm=np.sum(abs(target-total),axis=1)
            records.append(dict(condition=job['condition'],panel=job['panel'],case=row['seed'],success=row['success'],
                moving_samples=int(moving.sum()),gamma_min_moving=float(c[moving,3].min()) if len(valid) else None,
                requested_motion_minus_public_max_m_s=float(np.max(abs(c[moving,4]-c[moving,2]))) if len(valid) else None,
                maximum_unbounded_motion_delta_Nm=float(np.max(abs(c[:,12:18]))),maximum_damping_request_Nm=float(np.max(abs(c[:,18:24]))),
                combined_target_minus_applied_L1_max_Nm=float(norm.max()),combined_request_peak_Nm=float(np.max(abs(total))),
                actual_wheel_change_max_Nm=float(np.max(abs(actual_change))),
                requested_wheel_correction_max_Nm=float(np.max(abs(total[:,4:6]))),
                motor_envelope_reconstruction_exact=True,inherited_header_is_original_pool=True,
                query_error2_updates=int(np.sum(update & np.any(c[:,30:32]==2,axis=1)))))
    assert len(records)==328
    summary={}
    for condition in ('Bomega','Cgamma'):
        rows=[r for r in records if r['condition']==condition]
        summary[condition]=dict(trajectories=len(rows),motor_command_reconstruction_exact=sum(r['motor_envelope_reconstruction_exact'] for r in rows),
            nonzero_actual_wheel_change=sum(r['actual_wheel_change_max_Nm']>0 for r in rows),
            maximum_actual_wheel_change_Nm=max(r['actual_wheel_change_max_Nm'] for r in rows),maximum_target_applied_L1_gap_Nm=max(r['combined_target_minus_applied_L1_max_Nm'] for r in rows),
            minimum_moving_gamma=min(r['gamma_min_moving'] for r in rows if r['gamma_min_moving'] is not None),
            query_error2_updates=sum(r['query_error2_updates'] for r in rows))
    runner.write(out/'round224_request_delivery.json',dict(verified=True,round=224,summary=summary,records=records,input_sha256=inputs,source_sha256=runner.sha(__file__),runner_contract_sha256=runner.sha(out/'runner_contract.json'),new_training_or_evaluations=0,
        findings='All328 motor commands reconstruct exactly from actual combined pool+original Nom+envelope. The inherited fulltrace nominal-correction header records ORIGINAL shared pool,not the hook-replaced combined pool:usecoordination log for candidate applied requests. No result/gate/raw field overwritten.',
        limits='Combined target-applied gap includes box clipping and temporal slew,not attributable to one constraint alone. Nonzero execution is not correct yaw direction,adequate authority or causal task benefit. All fixed gates remain failed. Future learned method still requires distinctive mechanism/strong comparisons/freshtraining andOOD.',
        next='225 deepreview/cleanup:close fixed heuristic chain;decide a distinguishable dynamics/support authority identification or theory-backed mechanism,notbudget/gain rescue.'))
    print('PASS224 actual motor reconstructions',summary,flush=True)


if __name__=='__main__':run()
