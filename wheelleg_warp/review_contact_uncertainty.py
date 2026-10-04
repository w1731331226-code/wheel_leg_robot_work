"""Independent corner/interior reconstruction; no continuous-domain certificate."""
from pathlib import Path
from dataclasses import replace
from itertools import product
import json
import numpy as np
from review_yaw_sector import ROOT,sha
from probe_contact_uncertainty import construct,step
from native.terrain import HeightTerrainScenario,model

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/contact_uncertainty_v1'
DATA=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_response_v1'
CONTACT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/contact_response_v1'


def run():
    p=json.loads((OUT/'registration.json').read_text());r=json.loads((OUT/'result.json').read_text());contacts=json.loads((CONTACT/'result.json').read_text());prior=json.loads((DATA/'registration.json').read_text());ledger=json.loads((DATA/'completion.json').read_text())
    assert r['verified_producer'] and r['cpu_prediction_steps']==648 and r['cpu_oracle_validation_steps']==216 and p['training_updates']==0
    assert p['source_sha256']==sha(ROOT/'wheelleg_warp/probe_contact_uncertainty.py') and p['contact_review_sha256']==sha(CONTACT/'review.json') and not p['old_gate_or_final_used']
    assert r['registration_sha256']==sha(OUT/'registration.json')
    bounds=np.array([[7,7.5],[.6,1],[.6,1],[-.03,.03]],float);assert p['bounds']==bounds.tolist() and p['corners']==[list(v) for v in product(*bounds.tolist())]
    interior=(bounds[:,0]+np.random.default_rng(129001).uniform(.05,.95,(8,4))*(bounds[:,1]-bounds[:,0])).tolist();assert p['interiors']==interior
    reserve=float(np.spacing(np.float32(1.4)));assert p['numerical_reserve_rad']==reserve
    states={}
    for rec in ledger['records']:
        path=DATA/(rec['label']+'_states.npz');assert sha(path)==rec['states_sha256'];z=np.load(path,allow_pickle=False)
        for i,state in enumerate(z['state']):states[(rec['label'],i)]=(state,z['ids'],int(z['world'][i]),int(z['event'][i]))
    assert len(r['records'])==len(states)==27;seen=set();allchecks=[]
    for rec in r['records']:
        key=(rec['label'],rec['index']);assert key not in seen;seen.add(key);state,ids,world,event=states[key];assert rec['event']==event
        descriptor=next(x['contacts'] for x in contacts['records'] if x['label']==key[0] and x['index']==key[1])
        values=np.array([step(construct(state[:17],descriptor,vector),state,ids) for vector in p['corners']])
        np.testing.assert_allclose(values,rec['corner_q'],rtol=0,atol=1e-12);lower=values.min(axis=0);upper=values.max(axis=0)
        np.testing.assert_allclose(lower,rec['lower_q'],rtol=0,atol=1e-12);np.testing.assert_allclose(upper,rec['upper_q'],rtol=0,atol=1e-12)
        assert len(rec['interior_checks'])==8
        scene=HeightTerrainScenario(**prior['cases'][world]['scenario'])
        for j,c in enumerate(rec['interior_checks']):
            vector=interior[j];assert c['index']==j and c['parameter_vector']==vector
            predicted=step(construct(state[:17],descriptor,vector),state,ids)
            actual=step(model(replace(scene,mass=vector[0],mu_l=vector[1],mu_r=vector[2],drive_difference=vector[3])),state,ids)
            np.testing.assert_allclose(predicted,c['predicted_q'],rtol=0,atol=1e-12);np.testing.assert_allclose(actual,c['oracle_q'],rtol=0,atol=1e-12)
            assert np.isclose(abs(predicted-actual).max(),c['contact_oracle_difference_rad'],rtol=0,atol=1e-12)
            assert c['model_inside_raw_corner']==bool(np.all(predicted>=lower-1e-12)&np.all(predicted<=upper+1e-12))
            assert c['oracle_inside_reserved_corner']==bool(np.all(actual>=lower-reserve)&np.all(actual<=upper+reserve));allchecks.append(c)
        gpu=state[68:85][ids[:4]];np.testing.assert_array_equal(gpu,rec['observed_gpu_q'])
        assert rec['gpu_inside_reserved_corner']==bool(np.all(gpu>=lower-reserve)&np.all(gpu<=upper+reserve))
        print('VERIFIED interval',key,flush=True)
    summary=dict(interior_checks=len(allchecks),model_corner_misses=sum(not c['model_inside_raw_corner'] for c in allchecks),oracle_reserved_corner_misses=sum(not c['oracle_inside_reserved_corner'] for c in allchecks),
        observed_gpu_corner_misses=sum(not rec['gpu_inside_reserved_corner'] for rec in r['records']),max_contact_oracle_difference_rad=max(c['contact_oracle_difference_rad'] for c in allchecks),
        max_corner_width_rad=max(max(np.asarray(rec['upper_q'])-np.asarray(rec['lower_q'])) for rec in r['records']))
    assert summary==r['summary']
    report=dict(verified=True,cpu_predictions_reconstructed=648,cpu_oracles_reconstructed=216,states=27,summary=summary,
        registration_sha256=sha(OUT/'registration.json'),result_sha256=sha(OUT/'result.json'),reviewer_sha256=sha(__file__),
        scope='All16 public corners and8 prespecified interiors per state rebuilt with fixed current-contact descriptors. Actual parameter labels never choose a predictor corner.',
        limits='No independent state holdout or proof that continuous parameter extrema occur at corners. PositionULP reserve is arithmetic, not contact/model uncertainty. Zero finite misses do not admit deployment, learning or recursive state safety.')
    (OUT/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))


if __name__=='__main__':run()
