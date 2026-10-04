"""Read-only round92 checks and body-position diagnostics; no physics replay."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from train_height_comparison import summary
from native.terrain import HeightTerrainScenario,model
import mujoco

p=Path(__file__).resolve().parent
pre=json.loads((p/'preregistration.json').read_text())
result=json.loads((p/'result.json').read_text())
digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert pre['verifier_sha256']==result['verifier_sha256']==digest(ROOT/'wheelleg_warp/test_lateral_intervention.py')
assert pre['protocol_sha256']==result['protocol_sha256']==digest(ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json')
protocol=json.loads((ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json').read_text())
assert all(digest(ROOT/name)==sha for name,sha in protocol['source_sha256'].items())
assert result['training_updates']==0 and result['old_gate_or_final_replayed'] is False
rows=result['runs'];assert len(rows)==36==result['additional_public_diagnostic_episodes']
z=np.load(p/'trace.npz',allow_pickle=False);tr=z['trace'];offsets=z['offsets']
assert list(z['columns'])==['time_s','body_x_m','body_y_m','bounded_action_wheel_diff']
assert len(offsets)==37 and offsets[0]==0 and offsets[-1]==len(tr) and np.isfinite(tr).all()
assert np.max(abs(tr[:,3]))<=1 and np.all(np.diff(offsets)>0)
details=[]
for i,row in enumerate(rows):
    assert row['seed']==pre['cases'][i]['seed'] and row['scenario']==pre['cases'][i]['scenario']
    assert row['condition']==pre['conditions'][i] and row['reason']=='completed'
    assert row['physical_evidence_steps']==row['physical_steps']>0
    assert row['physical_safety_passed'] and row['design_joint_passed']
    t=tr[offsets[i]:offsets[i+1]]
    assert np.all(np.diff(t[:,0])>0) and abs(t[-1,0]-row['duration_s'])<1e-9
    assert abs(t[-1,2]-row['final_cross_track_m'])<1e-12
    crossing=np.flatnonzero(row['condition']['direction']*t[:,1]>=1.55)
    assert len(crossing)
    details.append(dict(seed=row['seed'],**row['condition'],success=row['success'],
        terrain_mask=row['touched_terrain_contact_mask'],body_y_at_x155_m=float(t[crossing[0],2]),
        final_body_y_m=float(t[-1,2]),peak_yaw_deg=row['peak_deg'][2]))
for arm in [0,1,-1]:
    selected=[r for r in rows if r['condition']['arm']==arm]
    assert summary(selected)==result['groups'][str(arm)]
    assert sum(r['success'] for r in selected)==4
    assert all(r['success']==(r['condition']['offset_m']==0) for r in selected)
m=model(HeightTerrainScenario(**pre['cases'][0]['scenario']))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,m.keyframe('stand').id);mujoco.mj_forward(m,d)
tile=m.geom('terrain_00');halfwidth=float(tile.size[1])
wheelhalf=float(m.geom('wheel_collide_L').size[1])
wheelcenters=[float(d.geom_xpos[m.geom('wheel_collide_'+s).id,1]) for s in ['L','R']]
report=dict(verified=True,episodes=36,physical_and_design_passed=36,groups=result['groups'],
    geometry=dict(tile_halfwidth_m=halfwidth,wheel_halfwidth_m=wheelhalf,nominal_wheel_y_m=wheelcenters,
        nominal_any_overlap_body_corridor_m=halfwidth+wheelhalf-max(abs(y) for y in wheelcenters)),
    details=details,scope='50Hz body-position proxy at x=1.55m, not exact wheel entry/contact timing; nominal upright corridor is not a dynamic collision proof',
    decision='Reject this fixed feedback law as a task-restoration result; correct sign reduces lateral error but all offset episodes still miss one wheel terrain evidence. Next inspect actual wheel trajectories and correction deadline before new control design.',
    verifier_sha256=digest(Path(__file__)))
(p/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS 36 completed episodes, physical/design evidence, bounded actions, source/protocol hashes and recomputed summaries')
print(json.dumps(report['geometry'],ensure_ascii=False))
