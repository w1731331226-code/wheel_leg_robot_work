"""Offline checks and a scientific figure from measured wheel trajectories."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent;root=p.parents[3]
sys.path[:0]=[str(root/'wheelleg_warp'),str(root/'wheelleg_ppo/tools')]
from train_height_comparison import summary
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
reg=json.loads((p/'telemetry_registration.json').read_text());v=json.loads((p/'wheel_verification.json').read_text())
assert v['source_sha256']==reg['source_sha256']==sha(root/'wheelleg_warp/diagnose_wheel_entry.py')
assert v['experiment_sha256']==reg['experiment_sha256']==sha(root/'wheelleg_warp/test_lateral_intervention.py')
r=json.loads((p/'result.json').read_text());rows=r['runs']
frozen=root/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json'
assert r['protocol_sha256']==sha(frozen)
assert all(sha(root/name)==value for name,value in json.loads(frozen.read_text())['source_sha256'].items())
z=np.load(p/'wheel_trace.npz',allow_pickle=False);tr=z['trace'];offsets=z['offsets']
assert np.isfinite(tr).all() and len(offsets)==37 and offsets[-1]==v['physics_samples']==len(tr)
assert len(rows)==36 and list(z['columns'])==reg['columns'] and np.all(tr[:,0]==1)
findings={}
for arm in [0,1,-1]:
    selected=[row for row in rows if row['condition']['arm']==arm]
    assert summary(selected)==r['groups'][str(arm)]
    separation=[]
    for i,row in enumerate(rows):
        if row['condition']['arm']!=arm:continue
        t=tr[offsets[i]:offsets[i+1]];detail=v['details'][i]
        assert len(t)==row['physical_steps']==row['physical_evidence_steps']
        np.testing.assert_allclose(t[:,1],np.arange(1,len(t)+1)*.0005,rtol=0,atol=1e-9)
        assert detail['seed']==row['seed'] and detail['condition']==row['condition']
        assert row['physical_safety_passed'] and row['design_joint_passed']
        for side in range(2):
            x=t[:,5+3*side];y=t[:,6+3*side]
            overlap=(row['condition']['direction']*x+t[:,11]>=1.55)&(row['condition']['direction']*x-t[:,11]<=2.05)
            measured=float((.16+t[:,12]-abs(y))[overlap].max())
            assert measured==detail['wheels'][side]['max_lateral_overlap_during_longitudinal_overlap_m']
            if row['condition']['offset_m'] and not (int(t[-1,14])&(4<<side)):
                assert measured<0 and not (t[:,13].astype(int)&(4<<side)).any()
                separation.append(-measured)
    assert len(separation)==8
    findings[str(arm)]=dict(missed_wheels=8,min_gap_over_entire_longitudinal_overlap_m=min(separation),
        max_gap_over_entire_longitudinal_overlap_m=max(separation),success_count=r['groups'][str(arm)]['success_count'])
fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
for ax,direction in zip(axes,[1,-1]):
    for arm,label in [(0,'Original B1'),(1,'Correct lateral feedback'),(-1,'Wrong-sign control')]:
        i=next(i for i,row in enumerate(rows) if row['condition']==dict(arm=arm,direction=direction,offset_m=.12,repeat=0))
        t=tr[offsets[i]:offsets[i+1]]
        ax.plot(direction*t[:,8],1000*(abs(t[:,9])-.16-t[:,12]),label=label,linewidth=1.4)
    ax.axhline(0,color='black',linestyle='--',linewidth=.8)
    ax.axvspan(1.55,2.05,color='grey',alpha=.15,label='Step extent')
    ax.set(xlim=(0,2.7),ylim=(-20,190),xlabel='Right wheel progress (m)',title=f'{"Forward" if direction==1 else "Reverse"}\nInitial body offset +120 mm')
    ax.grid(alpha=.2)
axes[0].set_ylabel('Right tire lateral gap to step (mm)')
axes[1].legend(fontsize=8,loc='upper left')
fig.tight_layout();fig.savefig(p/'wheel_clearance.png',dpi=180);plt.close(fig)
report=dict(verified=True,episodes=36,physics_samples=len(tr),groups=findings,
    controlled_result='Correct feedback reduces wheel-path separation, but every offset case still geometrically misses one wheel at all scored 0.5ms steps through longitudinal AABB overlap.',
    limits='Same development conditions replayed; not new independent test, not bitwise reproducibility, not continuous-time proof, not an optimal-control impossibility result. Native contact bits do not certify positive normal force.',
    next='Derive bounded heading/path correction with an explicit contact deadline from public geometry, before any new RL training; retain original evaluation thresholds and compare same-information classical controls.',
    analysis_sha256=sha(Path(__file__)),artifact_sha256={name:sha(p/name) for name in
        ['preregistration.json','telemetry_registration.json','result.json','trace.npz','wheel_trace.npz','wheel_verification.json']})
(p/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
