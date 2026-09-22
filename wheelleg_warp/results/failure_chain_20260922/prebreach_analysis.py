"""Read archived traces only: pre-failure actuator feasibility and command lag."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/'summary.json').read_text());rows=[]
for r in sorted(summary['runs'],key=lambda x:(x['policy'],x['seed'])):
    path=HERE/f"{r['policy']}_{r['seed']}.npz"
    with np.load(path,allow_pickle=False) as data:
        t=data['trace'];c={name:i for i,name in enumerate(data['columns'])}
    assert np.isfinite(t).all()
    get=lambda name:t[:,c[name]]
    first=lambda mask:float(get('end_s')[np.flatnonzero(mask)[0]]) if mask.any() else None
    start=r['first_target_contact_s'];end=r['first_attitude_failure_s'] or start+.2
    window=(get('end_s')>=start)&(get('end_s')<=end)
    fraction=lambda mask:float(mask[window].mean())
    requested=t[:,47:53];executed=t[:,53:59];base=t[:,41:47];bound=t[:,65:71]
    limits=np.ones_like(requested)
    np.divide(bound-base,requested,out=limits,where=requested>0)
    np.divide(-bound-base,requested,out=limits,where=requested<0)
    feasible=np.clip(limits.min(axis=1),0,1)
    hypothetical=base+feasible[:,None]*requested
    assert np.all(np.abs(hypothetical)<=bound+1e-7)
    np.testing.assert_allclose(executed,requested*get('lambda')[:,None],atol=1e-8,rtol=0)
    invalid=get('base_infeasible')>0
    pair_saturated=(np.abs(np.abs(base[:,4:6])-bound[:,4:6])<1e-7).all(axis=1)&(base[:,4]*base[:,5]>0)
    axes=[]
    if r['first_attitude_failure_s'] is not None:
        row=t[np.searchsorted(get('end_s'),end)]
        axes=[k for k in ('roll_error','pitch_error','yaw') if abs(row[c[k]])>np.deg2rad(5)]
    ndim=3 if r['policy']=='source' else 6
    target=t[:,23:23+ndim];filtered=t[:,29:29+ndim]
    wheel=[]
    for k in ([2] if ndim==3 else [4,5]):
        large=(abs(target[:,k])>.05)&(abs(filtered[:,k])>.05)
        opposite=large&(target[:,k]*filtered[:,k]<0)
        wheel.append(dict(channel=k,opposite_fraction=fraction(opposite),both_above_point05_fraction=fraction(large)))
    rows.append(dict(policy=r['policy'],seed=r['seed'],trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        success=r['info']['success'],failure_axis=axes,window_start_s=start,window_end_s=end,
        left_first_target_contact_s=first(get('left_target_contacts')>0),right_first_target_contact_s=first(get('right_target_contacts')>0),
        invalid_base_fraction=fraction(invalid),reclaimable_fraction=fraction(invalid&(feasible>1e-6)),
        blocked_after_clipped_base_fraction=fraction(feasible<=1e-6),same_sign_wheel_saturation_fraction=fraction(pair_saturated),
        command_filter_lag_fraction=fraction(np.any(abs(target-filtered)>1e-6,axis=1)),wheel_command_vs_filter=wheel,
        requested_rms=np.sqrt((requested[window]**2).mean(axis=0)).tolist(),
        executed_rms=np.sqrt((executed[window]**2).mean(axis=0)).tolist(),
        hypothetical_feasible_rms=np.sqrt(((requested*feasible[:,None])[window]**2).mean(axis=0)).tolist()))
result=dict(runs=rows,new_simulation=False,policy_or_control_changed=False,
    provenance='Original failure_chain traces: source M3 and prior virtual6 endpoint, NOT the later early-termination candidate.',
    windows='Failures: first target contact through first 5deg breach inclusive; successes: fixed first 0.2s. Different durations; not matched success/failure statistics.',
    interpretation='Reclaimable means torque-envelope feasibility only, not stable or useful control. Opposite target/filter sign is lag, not proof either is the correct stabilizing action. Mapping guards still required.',
    analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
destination=HERE/'prebreach_analysis.json'
temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temporary.replace(destination)
print('PASS: eight archived traces; hypothetical torque bounds and actual residual mapping verified')
