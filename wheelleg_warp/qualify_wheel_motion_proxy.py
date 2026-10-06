"""Offline time-aligned subset qualification with successful negative controls."""
import json
import numpy as np
import run_parking_withdrawal as runner
import wheel_motion_proxy as probe
from native.terrain import HeightTerrainScenario,model


def run():
    probe.unit();p=runner.verify();out=runner.OUT;root=runner.ROOT
    f=out/'round215_direction_review.json';review=json.loads(f.read_text());assert review['verified']
    assert runner.sha(root/'wheelleg_warp/review_motion_authority_direction.py')==review['source_sha256']
    inputs={str(f.relative_to(root)):runner.sha(f)};results=[];rated=probe.hw.MOTOR_RATED_RPM*2*np.pi/60
    for ref in review['strong_references']:
        case=ref['case'];label=ref['law']
        names=[n for n in review['input_sha256'] if n.endswith(f'/case_{case}.npz') and f'/controlled/{label}/' in n]
        assert len(names)==1;file=root/names[0];assert runner.sha(file)==review['input_sha256'][names[0]];inputs[names[0]]=runner.sha(file)
        with np.load(file,allow_pickle=False) as z:t=z['trace'];cols={str(k):i for i,k in enumerate(z['columns'])};post=z['post'];sizes=z['state_sizes']
        cpu=model(HeightTerrainScenario(**ref['scenario']));assert sizes.tolist()==[cpu.nq,cpu.nv,cpu.nu]
        ids=[cpu.nq+int(cpu.joint(n).dofadr[0]) for n in ('wheel1','wheel2')]
        np.testing.assert_array_equal(t[:,41:43],post[:,ids])
        q=post[:,:cpu.nq];yaw=np.arctan2(2*(q[:,3]*q[:,6]+q[:,4]*q[:,5]),1-2*(q[:,5]**2+q[:,6]**2))
        vx=np.cos(yaw)*post[:,cpu.nq]+np.sin(yaw)*post[:,cpu.nq+1]
        np.testing.assert_allclose(vx,t[:,44],atol=1e-12,rtol=0)
        # Reconstruct ONLY the three fields consumed by the proxy,not the whole Actor packet.
        history=np.zeros((len(t)+1,39),np.float32);history[1:,6]=t[:,44];history[1:,20:22]=t[:,41:43]
        decisions=np.arange(0,len(t)+1,40,dtype=int);actual_delay=int(round(ref['scenario']['delay_ms']*2))
        actual,indices=probe.delivered(history,decisions,actual_delay)
        assert np.all(indices<=decisions) and (len(decisions)<2 or decisions[1]==40)
        moving=np.r_[False,t[:,cols['speed_command']]!=0][indices]
        variations=[]
        for delay in (0,10,20,40):
            value,idx=probe.delivered(history,decisions,delay)
            variations.append(dict(assumed_delay_substeps=delay,max_discrepancy_change_m_s=float(np.max(np.abs(value['wheel_body_discrepancy_m_s']-probe.delivered(history,decisions,0)[0]['wheel_body_discrepancy_m_s']))),
                oldest_history_age_ms=float(np.max(decisions-idx)*.5)))
        results.append(dict(law=label,case=case,success=ref['success'],recorded_delay_substeps=actual_delay,actor_decision_endpoints=len(decisions),
            required_fields_only=[6,20,21],max_abs_moving_discrepancy_m_s=float(np.max(np.abs(actual['wheel_body_discrepancy_m_s'][moving]))),
            min_moving_nominal_capacity_Nm=float(np.min(actual['nominal_capacity_Nm'][moving])),
            above_rated_moving_actor_samples=int(np.any(np.abs(history[indices,20:22])>rated,axis=1)[moving].sum()),synthetic_delay_sensitivity=variations))
    assert len(results)==12 and sum(r['success'] for r in results)==4
    runner.write(out/'round216_proxy_qualification.json',dict(verified=True,round=216,records=results,input_sha256=inputs,
        source_sha256={f'wheelleg_warp/{n}':runner.sha(root/'wheelleg_warp'/n) for n in ('wheel_motion_proxy.py','qualify_wheel_motion_proxy.py')},
        observation_environment_sha256=runner.sha(root/'wheelleg_warp/native/environment.py'),route_source_sha256=runner.sha(root/'wheelleg_warp/route_state.py'),hardware_profile_sha256=runner.sha(root/'wheelleg_ppo/tools/hardware_profile.py'),
        actual_zero_delay_trajectories=12,successful_negative_controls=4,new_training_or_evaluations=0,
        allowed_claim='Discrepancy and nominal speed-dependent capacity computable from time-aligned unnormalized raw39 fields6/20/21 plus public constants. No contact/truth/arrival/hidden mass consumed.',
        limits='Reconstructed required subset,not saved actual full39/RMS-clipped observations. All actual trajectories delay0;delay5/10/20ms are offline time-shift fixtures,not delayed-plant validation. Data must not mix current state32:38 with delayed32 history for this proxy. Capacity is nominal torque availability,not net residual headroom or real torque bound;Nom command/gain uncertainty needed for those. Romega-vx is not true slip/support because pitch/yaw/leg motion/contact orientation are ignored. Diagnostic pre-integration force precedes post packet by0.5ms;truthnormal is excluded from proxy.',
        next='217 derive same-information speed-yaw coordination with dynamic/motor/contact assumptions and strong analytic comparison; do not threshold-fit these12 or call sensorproxy a novelty. Full contribution/formal5/freshID-OOD/statistics/manuscript still needed.'))
    print('PASS216 raw39 required-subset/20ms endpoints,12actualdelay0/4successfulcontrols;syntheticdelay only,0 new evaluations',flush=True)


if __name__=='__main__':run()
