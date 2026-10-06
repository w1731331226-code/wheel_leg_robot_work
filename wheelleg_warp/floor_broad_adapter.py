"""Clean-only broad role recording: retain terrain/delay, allow empty targets."""
import json
from pathlib import Path
import numpy as np
import reference_role_probe as roles

rec=roles.rec
OUT=rec.ROOT/'wheelleg_warp/results/paper_recovery_20261004/floor_broad_qualification_v1'
CAPACITY=65537  # Reject at constructor if this32.768s diagnostic allocation is too small.


def zero_table(cases,std):
    if std!=0 or not cases:raise ValueError('Broad adapter is clean-only and needs cases')
    return np.zeros((len(cases),CAPACITY),np.float32)


def geometry(raw):
    position=raw.data.geom_xpos.numpy();rotation=raw.data.geom_xmat.numpy();size=raw.model.geom_size.numpy()
    ids=raw.ids.numpy();windows=[];manifest=[]
    for w,scene in enumerate(raw.scenarios):
        boxes=[];ends=[];direction=np.sign(scene.speed)
        for j in range(int(ids[13]),int(ids[16])+1):
            half=size[w if len(size)>1 else 0,j];p=position[w,j];R=rotation[w,j];extent=abs(R)@half
            if p[2]+extent[2]<=0:continue
            boxes.append(dict(geom=j,position=p.tolist(),rotation=R.tolist(),size=half.tolist()))
            ends.extend([direction*p[0]-extent[0],direction*p[0]+extent[0]])
        windows.append([min(ends)-.2,max(ends)+.2] if ends else [-1e30,1e30]);manifest.append(boxes)
    return np.asarray(windows),manifest


def batches(cases):
    groups={}
    for i,c in enumerate(cases):groups.setdefault(c['scenario']['solver_iterations'],[]).append((i,c))
    return [rows[k:k+20] for rows in groups.values() for k in range(0,len(rows),20)]


def instrument(cases,mode,arm,directory=None):
    if not cases or len(cases)>20 or len({c['scenario']['solver_iterations'] for c in cases})!=1:
        raise ValueError('Use nonempty homogeneous solver batches of at most20 worlds')
    # Frozen recorder logs numeric noise seeds even in clean mode. These tags
    # affect filenames/metadata only; scenario and external regression ID stay intact.
    legacy=json.loads((OUT/'proposal.json').read_text())['panels']['legacy']
    names={c['seed']:87500000+i for i,c in enumerate(legacy)}
    recording=[dict(c,seed=names[c['seed']]) if isinstance(c['seed'],str) else c for c in cases]
    table=roles.noise.noise_table;geo=rec.old.geometry
    roles.noise.noise_table=zero_table;rec.old.geometry=geometry
    try:raw=roles.instrument(recording,mode,arm,0.,directory)
    finally:roles.noise.noise_table=table;rec.old.geometry=geo
    deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
    assert CAPACITY>deadline*40
    np.testing.assert_array_equal(raw.param.numpy()[:,4],np.rint([c['scenario']['delay_ms']*2 for c in cases]))
    assert raw.history.shape[1]==41
    raw._broad_contract=dict(verified=True,cases=[c['seed'] for c in cases],recording_ids=[c['seed'] for c in recording],arm=arm,
        capacity=CAPACITY,actor_deadline=deadline,physics_deadline=deadline*40,
        original_delay_steps=raw.param.numpy()[:,4].astype(int).tolist(),history_shape=list(raw.history.shape),
        terrain_kinds=[c['scenario']['terrain'] for c in cases],noise_std=0.,
        no_target_worlds=[i for i,b in enumerate(geometry(raw)[1]) if not b],
        semantics='Only clean zero-table provider and empty-target geometry handling changed. Sensor/history/control default interfaces restored outside construction; source role behaviour unchanged.')
    if directory is not None:
        meta=json.loads((directory/'geometry.json').read_text())
        meta['recording_case_ids']=meta['case_ids'];meta['case_ids']=[c['seed'] for c in cases]
        rec.atomic_json(directory/'geometry.json',meta)
        rec.atomic_json(directory/'broad_contract.json',raw._broad_contract)
    return raw


def freeze():
    from test_floor_broad_adapter import check
    assert not (OUT/'source_contract.json').exists();p=json.loads((OUT/'proposal.json').read_text())
    parent=rec.ROOT/p['parent_source_contract'];assert rec.sha(parent)==p['parent_source_contract_sha256']
    sources=dict(json.loads(parent.read_text())['source_sha256']);assert all(rec.sha(rec.ROOT/n)==v for n,v in sources.items())
    check()
    for name in ('floor_broad_adapter.py','test_floor_broad_adapter.py'):
        sources['wheelleg_warp/'+name]=rec.sha(rec.ROOT/'wheelleg_warp'/name)
    rec.atomic_json(OUT/'source_contract.json',dict(verified=True,proposal_sha256=rec.sha(OUT/'proposal.json'),
        source_sha256=sources,unit_sha256=rec.sha(OUT/'unit.json'),new_evaluations=0,training_updates=0,
        status='Clean broad source admitted;312old rows conditional reuse by identity;344runner not frozen/run.'))
    print('SOURCE ADMITTED clean9terrain/delay/named28;0 new physics evaluation',flush=True)


if __name__=='__main__':freeze()
