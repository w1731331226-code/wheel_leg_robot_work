"""Independent offline reconstruction of registered deadline-reference commands."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

p=Path(__file__).resolve().parent;root=p.parents[3]
sys.path[:0]=[str(root/'wheelleg_warp'),str(root/'wheelleg_ppo/tools')]
from train_height_comparison import summary
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
pre=json.loads((p/'preregistration.json').read_text());result=json.loads((p/'result.json').read_text())
assert pre['verifier_sha256']==result['verifier_sha256']==sha(root/'wheelleg_warp/test_deadline_reference.py')
assert pre['harness_sha256']==result['harness_sha256']==sha(root/'wheelleg_warp/test_lateral_intervention.py')
frozen=root/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json'
assert pre['protocol_sha256']==result['protocol_sha256']==sha(frozen)
assert all(sha(root/name)==value for name,value in json.loads(frozen.read_text())['source_sha256'].items())
assert result['training_updates']==0 and result['old_gate_or_final_replayed'] is False
rows=result['runs'];assert len(rows)==36
z=np.load(p/'reference_trace.npz',allow_pickle=False);trace=z['trace'];offsets=z['offsets']
after=np.load(p/'trace.npz',allow_pickle=False)
assert len(offsets)==37 and offsets[-1]==len(trace) and np.isfinite(trace).all()
assert list(z['columns'])==['start_s','body_x_m','body_y_m','observed_yaw_rad','observed_yaw_rate_rad_s','observed_command_m_s','desired_yaw_rad','bounded_action_wheel_diff']
candidate=pre['candidate'];assert candidate['roll_gain']==0
for i,row in enumerate(rows):
    assert row['condition']==pre['conditions'][i] and row['scenario']==pre['cases'][i]['scenario'] and row['seed']==pre['cases'][i]['seed']
    assert row['reason']=='completed' and row['physical_safety_passed'] and row['design_joint_passed']
    assert row['physical_evidence_steps']==row['physical_steps']>0
    a=trace[offsets[i]:offsets[i+1]];b=after['trace'][after['offsets'][i]:after['offsets'][i+1]]
    assert len(a)==len(b)==row['episode']['l']
    np.testing.assert_allclose(a[:,0],np.arange(len(a))*.02,atol=1e-9,rtol=0)
    c=row['condition'];remaining=np.maximum(1.5-c['direction']*a[:,1],.05)
    desired=c['arm']*np.clip(np.arctan2(-c['direction']*a[:,2],remaining),-np.pi/36,np.pi/36)*(abs(a[:,5])>.05)
    np.testing.assert_allclose(desired,a[:,6],atol=1e-15,rtol=0)
    # Production calculation uses float32 observed yaw/rate before adding float64 reference.
    pd=-(candidate['kp']*a[:,3].astype(np.float32)+candidate['kd']*a[:,4].astype(np.float32))
    action=np.clip((pd+(.4+candidate['kp'])*desired)/.3,-1,1).astype(np.float32)
    np.testing.assert_array_equal(action,a[:,7]);np.testing.assert_array_equal(a[:,7],b[:,3])
    assert np.max(abs(a[:,7]))<=1 and np.max(abs(a[:,6]))<=np.pi/36
groups={}
for arm in [0,1,-1]:
    selected=[row for row in rows if row['condition']['arm']==arm]
    assert summary(selected)==result['groups'][str(arm)]
    groups[str(arm)]=dict(**result['groups'][str(arm)],by_direction={str(d):dict(total=6,
        successes=sum(row['success'] for row in selected if row['condition']['direction']==d)) for d in [1,-1]},
        maximum_actual_yaw_peak_deg=max(row['peak_deg'][2] for row in selected))
prior=root/'wheelleg_warp/results/paper_recovery_20261004/wheel_entry_2khz_checked'
prior_z=np.load(prior/'wheel_trace.npz',allow_pickle=False)
prior_rows=json.loads((prior/'result.json').read_text())['runs'];wheel_offset=[]
for direction in [1,-1]:
    i=next(i for i,row in enumerate(prior_rows) if row['condition']==dict(arm=0,direction=direction,offset_m=0.,repeat=0))
    t=prior_z['trace'][prior_z['offsets'][i]:prior_z['offsets'][i+1]]
    wheel_offset.append(dict(direction=direction,initial_wheel_minus_body_x_m=float(t[0,5]-t[0,2])))
report=dict(verified=True,episodes=36,groups=groups,prior_geometry_probe=wheel_offset,
    prior_trace_sha256=sha(prior/'wheel_trace.npz'),
    decision='Partial constructed task restoration only: correct reference reverse6/6, forward2/6. Original B1 and wrong sign each4/12. No general controller promotion or yaw-score superiority.',
    next='Audit body-to-wheel longitudinal offset and nominal yaw feedback deadband before changing the deadline; do not sweep gains or weaken original success gates.',
    limits='Six geometry/direction/offset conditions per arm repeated twice, not12 independent random cases; ideal odometry and public geometry; no new RL training or independent evaluation.',
    verifier_sha256=sha(Path(__file__)),artifact_sha256={name:sha(p/name) for name in ['preregistration.json','result.json','trace.npz','reference_trace.npz']})
(p/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['groups'],ensure_ascii=False))
print('PASS source hashes, original physical/design evidence and summaries, all recorded action/reference reconstruction')
