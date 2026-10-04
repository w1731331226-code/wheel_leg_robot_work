"""Offline telemetry checks and all-case displacement distribution; no fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parent;root=p.parents[4]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
pre=json.loads((p/'registration.json').read_text());result=json.loads((p/'result.json').read_text())
assert pre['source_sha256']==result['source_sha256']==sha(root/'wheelleg_warp/trace_reward_contact.py')
assert result['replayed_episodes']==192 and result['policy_updates']==result['normalization_updates']==0
metrics={};paired={};hashes={};fig,ax=plt.subplots(figsize=(6.5,3.6))
for arm in ['original','potential']:
    r=json.loads((p/(arm+'.json')).read_text());rows=r['runs'];z=np.load(p/(arm+'_trace.npz'),allow_pickle=False)
    assert len(rows)==96 and list(z['columns'])==pre['columns'] and z['offsets'][-1]==len(z['trace'])
    before_center_yaw=[]
    for i,row in enumerate(rows):
        assert row['seed']==pre['cases'][i]['seed'] and row['scenario']==pre['cases'][i]['scenario']
        t=z['trace'][z['offsets'][i]:z['offsets'][i+1]]
        assert np.isfinite(t).all() and len(t)==row['episode']['l'] and np.all(np.diff(t[:,0])>0)
        assert np.max(abs(t[:,6:9]))<=1 and t[-1,0]==row['duration_s']
        assert float(abs(t[:,2]).max())==row['max_abs_body_y_m']
        assert int(t[-1,4])==(row['touched_contact_mask']|row['touched_terrain_contact_mask'])
        phase=(t[:,0]>=2)&(np.sign(row['scenario']['speed'])*t[:,1]<row['scenario']['center'])
        assert phase.any();before_center_yaw.append(float(np.mean(t[phase,3])*180/np.pi))
    requested=[v for v in rows if v['required_terrain_contact_mask']]
    groups={name:[v for v in requested if bool(v['terrain_passed'])==passed] for name,passed in [('contact_passed',True),('contact_missing',False)]}
    adjusted=np.array([np.sign(v['scenario']['speed'])*v['body_y_at_center_progress_m'] for v in rows])
    metrics[arm]=dict(summary=r['summary'],mean_max_abs_y_m=r['mean_max_abs_body_y_m'],mean_abs_y_at_center_m=r['mean_abs_y_at_center_m'],
        mean_direction_adjusted_center_y_m=float(adjusted.mean()),positive_direction_adjusted_cases=int(np.sum(adjusted>0)),
        mean_pre_center_yaw_deg=float(np.mean(before_center_yaw)),positive_pre_center_yaw_cases=sum(v>0 for v in before_center_yaw),
        terrain_requested_cases=len(requested),groups={name:dict(n=len(vals),
            mean_max_abs_y_m=float(np.mean([v['max_abs_body_y_m'] for v in vals])) if vals else None,
            mean_abs_y_at_center_m=float(np.mean([abs(v['body_y_at_center_progress_m']) for v in vals])) if vals else None) for name,vals in groups.items()})
    values=np.sort([abs(v['body_y_at_center_progress_m']) for v in rows]);ax.step(values*1000,np.arange(1,97)/96,where='post',label=arm)
    paired[arm]={v['seed']:v for v in rows}
    for suffix in ['.json','_trace.npz']:hashes[arm+suffix]=sha(p/(arm+suffix))
assert paired['original'].keys()==paired['potential'].keys()
ax.set(xlabel='Absolute root-body lateral position at center progress (mm)',ylabel='Empirical fraction of 96 cases',ylim=(0,1.02))
ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(p/'lateral_position.png',dpi=180);plt.close(fig)
out=dict(verified=True,metrics=metrics,lost_body_case_count=sum(paired['original'][i]['success'] and not paired['potential'][i]['success'] for i in paired['original']),
    scope='All96 cases, largest-decline seed selected after data. Root-body proxy at signedx>=scenario.center, not exact wheel entry/contact clearance or universal corridor bound; contact timing sampled20ms. Critic predictions not MC targets, deterministic replay not stochastic-policy calibration.',
    conclusion='The degraded controller accumulates larger lateral position error across this fixed-policy pair. Contact-passed and missing groups still overlap; body proxy is not a geometric contact proof or causal learning-mechanism test.',
    original_pilot_disposition_unchanged=True,input_sha256=hashes,source_sha256=sha(Path(__file__)))
(p/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(metrics,ensure_ascii=False))
