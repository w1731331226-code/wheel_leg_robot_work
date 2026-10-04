"""Offline check of the one allowed nominal wheel-coordinate correction."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

p=Path(__file__).resolve().parent;root=p.parents[3]
sys.path[:0]=[str(root/'wheelleg_warp'),str(root/'wheelleg_ppo/tools')]
from train_height_comparison import summary
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
pre=json.loads((p/'preregistration.json').read_text());r=json.loads((p/'result.json').read_text());rows=r['runs']
for field,source in [('verifier_sha256','test_wheel_deadline.py'),('deadline_harness_sha256','test_deadline_reference.py'),('harness_sha256','test_lateral_intervention.py')]:
    assert pre[field]==r[field]==sha(root/'wheelleg_warp'/source)
protocol=root/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json'
assert pre['protocol_sha256']==r['protocol_sha256']==sha(protocol)
assert all(sha(root/name)==value for name,value in json.loads(protocol.read_text())['source_sha256'].items())
assert pre['nominal_wheel_body_offset_m']==.075 and len(rows)==36
assert r['training_updates']==0 and r['old_gate_or_final_replayed'] is False
z=np.load(p/'reference_trace.npz',allow_pickle=False);post=np.load(p/'trace.npz',allow_pickle=False)
assert len(z['offsets'])==37 and z['offsets'][-1]==len(z['trace']) and np.isfinite(z['trace']).all()
assert list(z['columns'])==['start_s','body_x_m','body_y_m','observed_yaw_rad','observed_yaw_rate_rad_s','observed_command_m_s','desired_yaw_rad','bounded_action_wheel_diff']
candidate=pre['candidate'];assert candidate['roll_gain']==0
maximum_rounding_error=0.
for i,row in enumerate(rows):
    assert row['condition']==pre['conditions'][i] and row['scenario']==pre['cases'][i]['scenario'] and row['seed']==pre['cases'][i]['seed']
    assert row['reason']=='completed' and row['physical_safety_passed'] and row['design_joint_passed']
    assert row['physical_evidence_steps']==row['physical_steps']>0
    a=z['trace'][z['offsets'][i]:z['offsets'][i+1]];b=post['trace'][post['offsets'][i]:post['offsets'][i+1]]
    assert len(a)==len(b)==row['episode']['l']
    np.testing.assert_allclose(a[:,0],np.arange(len(a))*.02,atol=1e-9,rtol=0)
    c=row['condition'];target_body_progress=1.55-.05-c['direction']*.075
    remaining=np.maximum(target_body_progress-c['direction']*a[:,1],.05)
    mathematical=c['arm']*np.clip(np.arctan2(-c['direction']*a[:,2],remaining),-np.pi/36,np.pi/36)*(abs(a[:,5])>.05)
    rounded_x=(a[:,1].astype(np.float32)+.075).astype(np.float64)
    runtime_remaining=np.maximum(1.5-c['direction']*rounded_x,.05)
    desired=c['arm']*np.clip(np.arctan2(-c['direction']*a[:,2],runtime_remaining),-np.pi/36,np.pi/36)*(abs(a[:,5])>.05)
    error=float(np.max(abs(mathematical-desired)));maximum_rounding_error=max(maximum_rounding_error,error)
    # |d atan2(y,r)/dr| <=1/(2r) <=10 for the registered r>=0.05m floor.
    assert error<=10*float(np.max(abs(rounded_x-(a[:,1]+.075))))+1e-15
    np.testing.assert_allclose(desired,a[:,6],atol=1e-15,rtol=0)
    pd=-(candidate['kp']*a[:,3].astype(np.float32)+candidate['kd']*a[:,4].astype(np.float32))
    action=np.clip((pd+(.4+candidate['kp'])*desired)/.3,-1,1).astype(np.float32)
    np.testing.assert_array_equal(action,a[:,7]);np.testing.assert_array_equal(a[:,7],b[:,3])
    assert np.max(abs(a[:,7]))<=1 and np.max(abs(a[:,6]))<=np.pi/36
groups={}
for arm in [0,1,-1]:
    selected=[row for row in rows if row['condition']['arm']==arm]
    assert summary(selected)==r['groups'][str(arm)]
    groups[str(arm)]=dict(**r['groups'][str(arm)],by_direction={str(d):dict(total=6,
        successes=sum(row['success'] for row in selected if row['condition']['direction']==d)) for d in [1,-1]},
        peak_actual_yaw_deg=max(row['peak_deg'][2] for row in selected))
accepted=groups['1']['success_count']==12
report=dict(verified=True,accepted=accepted,episodes=36,groups=groups,
    maximum_float32_coordinate_rounding_error_rad=maximum_rounding_error,
    precision_audit='Runtime addition x+.075 is float32 before direction promotes it to float64. Exact commands reconstructed; ideal-coordinate discrepancy bounded by registered distance floor, no tolerance widening or physics rerun.',
    preregistered_acceptance='All12 correct-reference conditions under original complete task gates, both directions',
    decision='Do not promote the nominal coordinate correction. Close this bounded deadline-reference branch; no gain or reference-bound search.' if not accepted else 'Fixed constructed conditions passed; generalization and same-information learning remain unproven.',
    next='Audit unchanged dense/terminal reward after irreversible breaches on saved public traces, then register a matched reward-timing mechanism intervention if warranted; not another deadline parameter sweep.',
    limits='Six constructed direction/offset conditions per arm repeated twice; ideal odometry and known step geometry, not independent tests or old-training-distribution causal proof.',
    artifact_sha256={name:sha(p/name) for name in ['preregistration.json','result.json','trace.npz','reference_trace.npz']},verifier_sha256=sha(Path(__file__)))
(p/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(groups,ensure_ascii=False))
print('PASS command/reference reconstruction and original source/evidence/summaries; registered acceptance',accepted)
