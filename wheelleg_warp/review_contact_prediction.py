"""Reconstruct every frozen holdout prediction, descriptor and inclusion score."""
import json
from itertools import product
import numpy as np
from review_yaw_sector import ROOT,sha
from probe_contact_uncertainty import construct,step
from check_contact_response import extract_current
from native.terrain import HeightTerrainScenario,model
from dashboard.live_env import atomic_json

BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004'
DATA=BASE/'contact_state_holdout_v2'
OUT=BASE/'contact_state_prediction_v1'


def run():
    p=json.loads((OUT/'registration.json').read_text());c=json.loads((OUT/'completion.json').read_text());reg=json.loads((DATA/'registration.json').read_text());original=json.loads((DATA/'collection_completion.json').read_text())
    assert p['states']==c['states']==576 and c['cpu_corner_steps']==p['cpu_corner_steps']==9216 and c['cpu_actual_geometry_steps']==p['cpu_actual_geometry_steps']==576
    assert p['training_updates']==c['training_updates']==p['new_gpu_steps']==0 and not p['old_gate_or_final_used']
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items()) and c['registration_sha256']==sha(OUT/'registration.json')
    assert sha(DATA/'collection_review.json')==p['collection_review_sha256'] and sha(DATA/'collection_completion.json')==p['collection_completion_sha256']
    bounds=[[7.,7.5],[.6,1.],[.6,1.],[-.03,.03]];assert p['bounds']==bounds and p['corners']==[list(v) for v in product(*bounds)]
    reserve=float(np.spacing(np.float32(1.4)));assert p['numerical_reserve_rad']==reserve
    assert c['records']==json.loads((OUT/'completed_jobs.json').read_text())['records']
    assert [r['label'] for r in c['records']]==[r['label'] for r in original['records']]
    records=[]
    for job,old in zip(c['records'],original['records']):
        assert sha(OUT/job['path'])==job['sha256'];sp=DATA/'collection'/f'{job["label"]}_states.npz';assert sha(sp)==old['states_sha256'];z=np.load(sp,allow_pickle=False);ids=z['ids']
        batch=json.loads((OUT/job['path']).read_text())['records'];assert len(batch)==job['states']==len(z['state'])==144
        for i,(r,s) in enumerate(zip(batch,z['state'])):
            world=int(z['world'][i]);scene=HeightTerrainScenario(**reg['cases'][world]['scenario'])
            assert (r['label'],r['index'],r['world'],r['case'],r['event'],r['time_s'])==(job['label'],i,world,reg['cases'][world]['seed'],int(z['event'][i]),s[49])
            descriptor=extract_current(scene,s[:17]);assert descriptor==r['contacts']
            corners=np.array([step(construct(s[:17],descriptor,vector),s,ids) for vector in p['corners']]);np.testing.assert_allclose(corners,r['corner_q'],rtol=0,atol=1e-12)
            lower=corners.min(0);upper=corners.max(0);np.testing.assert_array_equal(lower,r['lower_q']);np.testing.assert_array_equal(upper,r['upper_q'])
            oracle=step(model(scene),s,ids);gpu=s[68:85][ids[:4]];np.testing.assert_allclose(oracle,r['oracle_q'],rtol=0,atol=1e-12);np.testing.assert_array_equal(gpu,r['gpu_q'])
            assert r['interval_safe']==bool(np.all(lower-reserve>=-1.4)&np.all(upper+reserve<=1.4))
            for name,q in [('gpu',gpu),('oracle',oracle)]:
                assert r[name+'_inside']==bool(np.all(q>=lower-reserve)&np.all(q<=upper+reserve))
                excess=max(0.,float(np.max(lower-reserve-q)),float(np.max(q-upper-reserve)))
                assert np.isclose(excess,r[name+'_excess_rad'],rtol=0,atol=1e-15)
            records.append(r)
        print('VERIFIED',job['label'],len(batch),flush=True)
    groups={}
    for key in c['groups']:
        label,kind=key.split('/');rows=[r for r in records if (label=='all' or label==r['label']) and (kind=='all' or (r['event']==2)==(kind=='contact_creation'))]
        assert rows
        groups[key]=dict(states=len(rows),gpu_misses=sum(not r['gpu_inside'] for r in rows),oracle_misses=sum(not r['oracle_inside'] for r in rows),
            gpu_false_safe=sum(r['interval_safe'] and max(abs(v) for v in r['gpu_q'])>1.4 for r in rows),oracle_false_safe=sum(r['interval_safe'] and max(abs(v) for v in r['oracle_q'])>1.4 for r in rows),
            max_gpu_excess_rad=max(r['gpu_excess_rad'] for r in rows),max_oracle_excess_rad=max(r['oracle_excess_rad'] for r in rows),max_corner_width_rad=max(max(u-l for u,l in zip(r['upper_q'],r['lower_q'])) for r in rows))
    assert len(records)==576 and groups==c['groups'] and c['development_gate']==(groups['all/all']['gpu_misses']==groups['all/all']['oracle_misses']==0)
    misses=[dict(label=r['label'],case=r['case'],event=r['event'],index=r['index'],gpu_excess_rad=r['gpu_excess_rad'],oracle_excess_rad=r['oracle_excess_rad']) for r in records if not(r['gpu_inside'] and r['oracle_inside'])]
    atomic_json(OUT/'review.json',dict(verified=True,states=576,reconstructed_corner_steps=9216,reconstructed_oracle_steps=576,groups=groups,misses=misses,development_gate=c['development_gate'],registration_sha256=sha(OUT/'registration.json'),completion_sha256=sha(OUT/'completion.json'),reviewer_sha256=sha(__file__),training_updates=0,
        interpretation='All finite new-state scores rebuilt with unchanged current-contact/corner model. Misses reject this proposed envelope; even zero misses would not prove continuous-domain safety or controller admission.'))
    print('PASS',groups['all/all'],flush=True)


if __name__=='__main__':run()
