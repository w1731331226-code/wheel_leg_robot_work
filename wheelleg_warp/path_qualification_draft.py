"""Separate sampled centre-lane prerequisite, not a replacement task score."""
import json
import numpy as np
from task_mode_recorder import OUT, COL
from review_yaw_sector import sha
from score_reference_learning import load_rows
from dashboard.live_env import atomic_json

I={c:i for i,c in enumerate(COL)}


def conditions(trace,box,side,direction):
    t=np.asarray(trace); R=np.asarray(box['rotation']); h=np.asarray(box['size']); position=np.asarray(box['position'])
    if (t.ndim!=2 or t.shape[1]!=len(COL) or not len(t) or not np.isfinite(t).all()
        or side not in ('left','right') or direction not in (-1,1) or R.shape!=(3,3)
        or h.shape!=(3,) or position.shape!=(3,) or np.any(h<=0) or not np.isfinite(h).all() or not np.isfinite(position).all()
        or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1.,atol=1e-6)):
        raise ValueError('Invalid trace/box/side/direction')
    p=(t[:,[I[f'pre_{side}_{a}'] for a in 'xyz']]-box['position'])@R
    q=(t[:,[I[f'post_{side}_{a}'] for a in 'xyz']]-box['position'])@R
    x0=direction*p[:,0]; x1=direction*q[:,0]; eps=1e-7
    inside0=abs(x0)<=h[0]+eps; inside1=abs(x1)<=h[0]+eps; slab=inside0|inside1
    entering=np.flatnonzero(x0<-h[0]-eps); exiting=np.flatnonzero(x1>h[0]+eps)
    before_after=bool(len(entering) and len(exiting) and entering[0]<exiting[-1])
    lateral=bool(np.all(abs(p[inside0,1])<=h[1]+eps) and np.all(abs(q[inside1,1])<=h[1]+eps))
    in_span=np.flatnonzero(slab); adequate=len(in_span)>=2
    # Every interior segment of each visit needs consecutive recorded physics steps.
    consecutive=slab[1:]&slab[:-1]
    dense=bool(np.all(np.diff(t[:,I['step']])[consecutive]==1))
    geometric=before_after and adequate and dense and lateral
    return dict(before_entry_and_after_exit=before_after,interior_samples=int(slab.sum()),sampled_lane_bound_passed=lateral,
        interior_density_passed=dense,geometry_prerequisite=geometric,
        complete_loaded_rollover_qualified=None,
        limitation='Necessary sampled centre-lane geometry only; not whole-tyre footprint, continuous-time proof, top-load/airborne mode or final task success.')


def unit():
    b=dict(position=[0,0,0],rotation=np.eye(3).tolist(),size=[1,.035,.01],geom=3)
    t=np.zeros((5,len(COL))); t[:,I['step']]=np.arange(1,6)
    t[:,I['pre_left_x']]=[-1.1,-.9,0,.9,1.05]; t[:,I['post_left_x']]=[-1.05,-.8,.1,.95,1.1]
    assert conditions(t,b,'left',1)['geometry_prerequisite']
    a=t.copy(); a[:,I['pre_left_y']]=a[:,I['post_left_y']]=.06; assert not conditions(a,b,'left',1)['geometry_prerequisite']
    a=t.copy(); a[2:,I['step']]+=10; assert not conditions(a,b,'left',1)['interior_density_passed']
    a=t.copy(); a[:,[I['pre_left_x'],I['post_left_x']]]*=-1; assert conditions(a,b,'left',-1)['geometry_prerequisite']
    assert conditions(t,b,'left',1)['complete_loaded_rollover_qualified'] is None


def run():
    unit(); proposal=json.loads((OUT/'proposal.json').read_text()); completion=json.loads((OUT/'completion.json').read_text())
    rows=[]; inputs={}
    for job in completion['records']:
        file=OUT/job['path']; assert sha(file)==job['sha256']; result=load_rows(file,proposal['cases'])
        gf=file.parent/'geometry.json'; assert sha(gf)==job['geometry_sha256']; g=json.loads(gf.read_text());inputs[str(file.relative_to(OUT))]=sha(file); inputs[str(gf.relative_to(OUT))]=sha(gf)
        for w,r in enumerate(result['runs']):
            if r['scenario']['terrain']!='legacy':
                rows.append(dict(controller=job['label'],case=r['seed'],original_success=r['success'],status='mixed_event/path contract not implemented')); continue
            tf=file.parent/r['task_mode_trace']['path'];assert sha(tf)==r['task_mode_trace']['sha256'];inputs[str(tf.relative_to(OUT))]=sha(tf)
            with np.load(tf,allow_pickle=False) as data:assert data['columns'].tolist()==COL; trace=data['trace']
            side='left' if r['scenario']['height_l'] else 'right';assert len(g['boxes'][w])==1
            rows.append(dict(controller=job['label'],case=r['seed'],original_success=r['success'],**conditions(trace,g['boxes'][w][0],side,int(np.sign(r['scenario']['speed'])))))
    assert len(rows)==205
    summary={label:dict(original_success=sum(r['original_success'] for r in rows if r['controller']==label),
        legacy_geometry_prerequisite=sum(r.get('geometry_prerequisite',False) for r in rows if r['controller']==label),
        original_success_and_legacy_geometry=sum(r['original_success'] and r.get('geometry_prerequisite',False) for r in rows if r['controller']==label)) for label in [j['label'] for j in completion['records']]}
    atomic_json(OUT/'path_qualification_draft_audit.json',dict(verified=True,status='draft necessary geometry only; user task choice pending',summary=summary,rows=rows,input_sha256=inputs,source_sha256=sha(__file__),new_evaluations=0,training_updates=0,
        limits='No new final-success label; loaded top passage/duty, tyre clearance/contact distances, temporal gaps and mixed-event mapping still need an explicit protocol. Original metrics immutable. Do not tune requirements by candidate performance.'))
    print('PASS draft geometry checks and205 original-score preservation',summary,flush=True)


if __name__=='__main__': run()
