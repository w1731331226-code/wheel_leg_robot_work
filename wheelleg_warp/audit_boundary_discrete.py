"""Existing trace only: discrete hinge update, guard timing and viability gap."""
from pathlib import Path
import json
import numpy as np
from review_yaw_sector import sha,ROOT

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/leg_cone_v1'


def run():
    review=json.loads((OUT/'review.json').read_text());assert review['verified'] and not review['continue_gate']
    registration=json.loads((OUT/'registration.json').read_text());completion=json.loads((OUT/'completion.json').read_text())
    assert sha(OUT/'registration.json')==review['registration_sha256'] and sha(OUT/'completion.json')==review['completion_sha256']
    dt=float(np.float32(.0005));summary={};remaining=[];total=0
    for record in completion['records']:
        label=record['label'];path=OUT/(label+'_trace.npz');assert sha(path)==record['trace_sha256']
        rows=json.loads((OUT/(label+'.json')).read_text())['runs'];z=np.load(path,allow_pickle=False)
        maximum_error=0.;max_ratio=0.;fused_mismatches=0;split_mismatches=0
        for i,row in enumerate(rows):
            t=z['trace'][z['offsets'][i]:z['offsets'][i+1]]
            q=t[:,2:6];v=t[:,6:10];postq=t[:,10:14];postv=t[:,14:18]
            exact=q+dt*postv;fused=exact.astype(np.float32).astype(float)
            split=(q.astype(np.float32)+(np.float32(.0005)*postv.astype(np.float32))).astype(float)
            ulp=np.maximum(abs(np.spacing(exact.astype(np.float32))).astype(float),np.finfo(np.float32).tiny)
            error=abs(postq-exact);budget=.5*ulp+np.finfo(np.float32).eps*abs(dt*postv)+1e-12
            assert np.all(error<=budget),'Observed hinge update exceeds roundoff-only envelope'
            maximum_error=max(maximum_error,float(error.max()));max_ratio=max(max_ratio,float((error/ulp).max()))
            fused_mismatches+=int(np.sum(fused!=postq));split_mismatches+=int(np.sum(split!=postq));total+=q.size
            crossing=np.flatnonzero(t[:,47]<0)
            if label.endswith('_cone') and len(crossing):
                k=int(crossing[0]);j=int(abs(postq[k]).argmax());sign=float(np.sign(postq[k,j]))
                effective_a=(postv[:,j]-v[:,j])/dt
                velocity_term=sign*dt*v[k,j];acceleration_term=sign*dt*(postv[k,j]-v[k,j])
                signed_rounding=sign*(postq[k,j]-exact[k,j]);margin=1.4-sign*q[k,j]
                assert np.isclose(velocity_term+acceleration_term+signed_rounding-margin,sign*postq[k,j]-1.4,rtol=0,atol=1e-12)
                guard=(abs(q[:,j])+np.maximum(0,np.sign(q[:,j])*v[:,j])*.02>=1.4)&(np.sign(q[:,j])*t[:,18+j]<0)
                starts=np.flatnonzero(guard&np.r_[True,~guard[:-1]])
                before=starts[starts<=k]
                first=int(before[0]) if len(before) else None;last=int(before[-1]) if len(before) else None
                nearby=effective_a[max(0,k-40):k+1]*sign
                remaining.append(dict(label=label,case=row['seed'],joint=j,time_s=float(t[k,1]),
                    pre_margin_rad=float(margin),pre_outward_velocity_rad_s=float(sign*v[k,j]),post_outward_velocity_rad_s=float(sign*postv[k,j]),
                    retrospective_effective_outward_acceleration_rad_s2=float(sign*effective_a[k]),
                    velocity_displacement_rad=float(velocity_term),velocity_change_displacement_rad=float(acceleration_term),rounding_displacement_rad=float(signed_rounding),
                    first_overshoot_rad=float(sign*postq[k,j]-1.4),maximum_episode_overshoot_rad=-row['min_active_design_margin_rad'],
                    original_guard_active=bool(guard[k]),first_guard_start_s=float(t[first,1]) if first is not None else None,
                    last_guard_start_before_crossing_s=float(t[last,1]) if last is not None else None,
                    prior20ms_observed_outward_acceleration_max=float(nearby.max()),
                    actor_direction_Nm=float(sign*t[k,24+j]),nominal_direction_Nm=float(sign*t[k,18+j]),
                    note='Acceleration here is reconstructed after the step, not an available prediction or validated robust bound.'))
        summary[label]=dict(max_hinge_integration_error_rad=maximum_error,max_error_in_position_ulps=max_ratio,
            fused_float32_prediction_mismatches=fused_mismatches,split_float32_prediction_mismatches=split_mismatches)
        print('VERIFIED',label,summary[label],flush=True)
    assert len(remaining)==4 and total==review['physics_samples']*4
    source=ROOT/'.venv/lib/python3.10/site-packages/mujoco_warp/_src/forward.py'
    report=dict(verified=True,joint_updates=total,hinge_update='q_next=round_float32(q+dt*v_next), semi-implicit/implicit velocity update; active hinge coordinates only, not free-joint quaternion integration.',
        trace_summary=summary,remaining_crossings=remaining,trace_review_sha256=sha(OUT/'review.json'),installed_forward_source_sha256=sha(source),reviewer_sha256=sha(__file__),
        discrete_condition=dict(formula='For both s=+-1: s*(q+dt*v)+dt^2*A_s+roundoff_bound <=1.4, with A_s >= s*a_effective',
            assumptions='A_s bounds the signed next-step effective acceleration for the actual coupled/contact/actuation state and candidate control; a torque sign constraint does not supply it.',
            warning='Retrospective finite differences or this development-sample maximum cannot be relabelled as a guaranteed prospective disturbance/acceleration bound.'),
        conclusion='All observed active-hinge position differences fit the stated arithmetic error envelope. Four strict design failures remain; at three first crossings, pre velocity is inward yet next velocity is outward. Ignoring acceleration at a20ms kinematic guard and ignoring rounding are separate from physically preserving the state domain.',
        next='Obtain a bounded common Nom/Actor response/viability model with uncertainty and execution timing evidence before any new filter or learning; keep the original state gate and stopped candidate.',
        training_updates=0,physics_episodes=0,old_gate_or_final_used=False)
    (OUT/'discrete_audit_123.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PASS existing hinge updates',total,'remaining crossings',len(remaining),flush=True)


if __name__=='__main__':run()
